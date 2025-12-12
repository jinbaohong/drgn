# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
XArrays
-------

The ``drgn.helpers.linux.xarray`` module provides helpers for working with the
`XArray <https://docs.kernel.org/core-api/xarray.html>`_ data structure from
:linux:`include/linux/xarray.h`.

.. note::

    XArrays were introduced in Linux 4.20 as a replacement for `radix trees`_.
    To make it easier to work with data structures that were changed from a
    radix tree to an XArray (like ``struct address_space::i_pages``), drgn
    treats XArrays and radix trees interchangeably in some cases.

    Specifically, :func:`~drgn.helpers.linux.xarray.xa_load()` is equivalent to
    :func:`~drgn.helpers.linux.radixtree.radix_tree_lookup()`, and
    :func:`~drgn.helpers.linux.xarray.xa_for_each()` is equivalent to
    :func:`~drgn.helpers.linux.radixtree.radix_tree_for_each()`, except that
    the radix tree helpers assume ``advanced=False``. (Therefore,
    :func:`~drgn.helpers.linux.xarray.xa_load()` and
    :func:`~drgn.helpers.linux.xarray.xa_for_each()` also accept a ``struct
    radix_tree_root *``, and
    :func:`~drgn.helpers.linux.radixtree.radix_tree_lookup()` and
    :func:`~drgn.helpers.linux.radixtree.radix_tree_for_each()` also accept a
    ``struct xarray *``.)

    Additionally, old-style radix trees (Linux < 4.7) with explicit height
    fields in the root structure are also supported.
"""

from typing import Generator, Iterator, Optional, Tuple

from _drgn import _linux_helper_xa_load
from drgn import NULL, IntegerLike, Object, cast

__all__ = (
    "xa_for_each",
    "xa_is_value",
    "xa_is_zero",
    "xa_load",
    "xa_to_value",
)


_XA_ZERO_ENTRY = 1030  # xa_mk_internal(257)


def _xa_is_node(entry_value: int) -> bool:
    return (entry_value & 3) == 2 and entry_value > 4096


class _XAIteratorNode:
    def __init__(self, node: Object, index: int) -> None:
        self.slots = node.slots
        self.shift = node.shift.value_()
        self.index = index
        self.next_slot = 0


class _OldRadixTreeIteratorNode:
    """Iterator node for old-style radix trees (Linux < 4.7) without shift member."""

    __slots__ = ("node", "slots", "height", "shift", "index", "next_slot")

    def __init__(self, node: Object, height: int, shift: int, index: int) -> None:
        self.node = node
        self.slots = node.slots
        self.height = height
        self.shift = shift
        self.index = index
        self.next_slot = 0


# Old-style radix tree constants (Linux < 4.7)
_RADIX_TREE_INDIRECT_PTR = 1
_RADIX_TREE_EXCEPTIONAL_ENTRY = 2

# For kernel 4.4:
# RADIX_TREE_MAX_PATH = DIV_ROUND_UP(64, 6) = 11
# RADIX_TREE_HEIGHT_SHIFT = RADIX_TREE_MAX_PATH + 1 = 12
# RADIX_TREE_HEIGHT_MASK = (1 << 12) - 1 = 0xFFF
_RADIX_TREE_HEIGHT_MASK = 0xFFF


def _radix_tree_is_indirect_ptr(ptr_value: int) -> bool:
    """Check if pointer has RADIX_TREE_INDIRECT_PTR bit set."""
    return bool(ptr_value & _RADIX_TREE_INDIRECT_PTR)


def _get_radix_tree_map_params(root: Object) -> Tuple[int, int, int]:
    """
    Get radix tree parameters from kernel.
    Returns (MAP_SHIFT, MAP_SIZE, MAP_MASK)

    :param root: ``struct radix_tree_root *``
    """
    try:
        node_type = root.prog_.type("struct radix_tree_node")
        members = node_type.members
        if members is not None:
            for member in members:
                if member.name == "slots":
                    map_size = member.type.length
                    if map_size is not None:
                        map_shift = (map_size - 1).bit_length()
                        if (1 << map_shift) != map_size:
                            map_shift = 6  # Fallback
                            map_size = 64
                        map_mask = map_size - 1
                        return map_shift, map_size, map_mask
    except Exception:
        pass
    # Default: CONFIG_BASE_SMALL=n -> shift=6, size=64
    return 6, 64, 63


def _is_old_radix_tree(root: Object) -> bool:
    """
    Check if this kernel uses old-style radix trees (Linux < 4.7).

    Old-style radix trees have 'path' member in radix_tree_node but no 'shift'.

    :param root: ``struct radix_tree_root *``
    """
    prog = root.prog_
    try:
        node_type = prog.type("struct radix_tree_node")
        node_type.member("shift")
        # Has shift - newer radix tree
        return False
    except LookupError:
        try:
            node_type = prog.type("struct radix_tree_node")
            node_type.member("path")
            # Has path but no shift - old-style
            return True
        except LookupError:
            return False


def _xa_for_each_old_radix_tree(
    root: Object,
) -> Generator[Tuple[int, Object], None, None]:
    """
    Handle old-style radix trees (Linux < 4.7) with height encoded in node->path.

    These radix trees have:
    - root->rnode: pointer to root node (with RADIX_TREE_INDIRECT_PTR bit if internal)
    - node->path: encodes height in lower bits (height & RADIX_TREE_HEIGHT_MASK)
    - node->slots[]: child pointers or data entries

    IMPORTANT: Only root->rnode has the indirect bit set. Child node pointers
    in slots[] do NOT have the indirect bit - they are direct pointers.

    :param root: ``struct radix_tree_root *``
    """
    prog = root.prog_
    node_type = prog.type("struct radix_tree_node *")
    map_shift, map_size, map_mask = _get_radix_tree_map_params(root)

    rnode = root.rnode.read_()
    rnode_value = rnode.value_()

    # Empty tree
    if rnode_value == 0:
        return

    # Check if it's an indirect pointer (has internal nodes)
    if not _radix_tree_is_indirect_ptr(rnode_value):
        # Direct pointer - single entry at index 0
        yield 0, cast("void *", rnode)
        return

    # Get actual node pointer (clear indirect bit from root's rnode only)
    root_node = Object(prog, node_type, rnode_value & ~_RADIX_TREE_INDIRECT_PTR)

    # Get height from node->path (NOT from root->height!)
    height = root_node.path.value_() & _RADIX_TREE_HEIGHT_MASK

    if height == 0:
        return

    # Start traversal
    stack = [_OldRadixTreeIteratorNode(root_node, height, (height - 1) * map_shift, 0)]

    while stack:
        current = stack[-1]

        # Exhausted this node's slots
        if current.next_slot >= map_size:
            stack.pop()
            continue

        slot = current.slots[current.next_slot].read_()
        slot_value = slot.value_()

        # Calculate index for this slot
        index = current.index + (current.next_slot << current.shift)
        current.next_slot += 1

        # Skip empty slots
        if slot_value == 0:
            continue

        if current.height == 1:
            # Leaf level - these are data entries
            # Skip exceptional entries (bit 1 set) - caller can use
            # include_exceptional parameter if needed
            yield index, cast("void *", slot)
        else:
            # Internal node - descend
            # IMPORTANT: In kernel 4.4, internal node pointers do NOT have
            # the indirect bit! Only root->rnode has it.
            child_node = Object(prog, node_type, slot_value)
            stack.append(
                _OldRadixTreeIteratorNode(
                    child_node, current.height - 1, current.shift - map_shift, index
                )
            )


def _xa_load_old_radix_tree(root: Object, index: int) -> Object:
    """
    Look up entry in old-style radix tree (Linux < 4.7).

    :param root: ``struct radix_tree_root *``
    :param index: Entry index
    :return: ``void *`` found entry, or ``NULL`` if not found
    """
    prog = root.prog_
    map_shift, map_size, map_mask = _get_radix_tree_map_params(root)

    # Get root node pointer
    rnode = root.rnode.read_()
    rnode_value = rnode.value_()

    # Empty tree
    if rnode_value == 0:
        return NULL(prog, "void *")

    # Check if it's an indirect pointer (has internal nodes)
    if not _radix_tree_is_indirect_ptr(rnode_value):
        # Direct pointer - single entry at index 0
        if index == 0:
            return cast("void *", rnode)
        return NULL(prog, "void *")

    # Get actual node pointer (clear indirect bit)
    node_type = prog.type("struct radix_tree_node *")
    node = Object(prog, node_type, rnode_value & ~_RADIX_TREE_INDIRECT_PTR)

    # Get height from node->path
    height = node.path.value_() & _RADIX_TREE_HEIGHT_MASK

    # Calculate max index for this height
    max_index = (1 << (height * map_shift)) - 1 if height > 0 else 0
    if index > max_index:
        return NULL(prog, "void *")

    shift = (height - 1) * map_shift

    while height > 0:
        slot_idx = (index >> shift) & map_mask
        slot = node.slots[slot_idx].read_()
        slot_value = slot.value_()

        if slot_value == 0:
            return NULL(prog, "void *")

        shift -= map_shift
        height -= 1

        if height == 0:
            # Reached leaf level - return data
            return cast("void *", slot)

        # Continue descent - slot is a node pointer
        # IMPORTANT: internal slots do NOT have indirect bit set in 4.4!
        node = Object(prog, node_type, slot_value)

    return NULL(prog, "void *")


def xa_load(xa: Object, index: IntegerLike, *, advanced: bool = False) -> Object:
    """
    Look up the entry at a given index in an XArray.

    >>> entry = xa_load(inode.i_mapping.i_pages.address_of_(), 2)
    >>> cast("struct page *", entry)
    *(struct page *)0xffffed6980306f40 = {
        ...
    }

    :param xa: ``struct xarray *``
    :param index: Entry index.
    :param advanced: Whether to return nodes only visible to the XArray
        advanced API. If ``False``, zero entries (see :func:`xa_is_zero()`)
        will be returned as ``NULL``.
    :return: ``void *`` found entry, or ``NULL`` if not found.
    """
    prog = xa.prog_

    # Check if this is a radix_tree_root (not xarray) and if it's old-style
    try:
        xa.xa_head  # XArray has xa_head member
    except AttributeError:
        # It's a radix_tree_root - check if old-style
        if _is_old_radix_tree(xa):
            return _xa_load_old_radix_tree(xa, int(index))

    # Use the C helper for XArrays and newer radix trees
    entry = _linux_helper_xa_load(xa, index)
    if not advanced and entry.value_() == _XA_ZERO_ENTRY:
        return NULL(prog, "void *")
    return entry


def xa_for_each(xa: Object, *, advanced: bool = False) -> Iterator[Tuple[int, Object]]:
    """
    Iterate over all of the entries in an XArray.

    >>> for index, entry in xa_for_each(inode.i_mapping.i_pages.address_of_()):
    ...     print(index, entry)
    ...
    0 (void *)0xffffed6980356140
    1 (void *)0xffffed6980306f80
    2 (void *)0xffffed6980306f40
    3 (void *)0xffffed6980355b40

    :param xa: ``struct xarray *``
    :param advanced: Whether to return nodes only visible to the XArray
        advanced API. If ``False``, zero entries (see :func:`xa_is_zero()`)
        will be skipped.
    :return: Iterator of (index, ``void *``) tuples.
    """
    prog = xa.prog_

    def should_yield(entry_value: int) -> bool:
        return entry_value != 0

    # This handles four cases:
    #
    # 1. XArrays.
    # 2. Radix trees since Linux kernel commit f8d5d0cc145c ("xarray: Add
    #    definition of struct xarray") (in v4.20) redefined them in terms of
    #    XArrays. These reuse the XArray structures and are close enough to
    #    case 1 that the same code handles both.
    # 3. Radix trees before that commit (Linux 4.7-4.19). These are similar to
    #    cases 1 and 2, but they have different type and member names, use
    #    different flags in the lower bits (see Linux kernel commit 3159f943aafd
    #    ("xarray: Replace exceptional entries") (in v4.20)), and represent
    #    sibling entries differently (see Linux kernel commit 02c02bf12c5d
    #    ("xarray: Change definition of sibling entries") (in v4.20)).
    # 4. Old-style radix trees (Linux < 4.7) with explicit height field in
    #    root structure and no shift member in nodes. These use
    #    RADIX_TREE_INDIRECT_PTR (bit 0) for internal node marking.
    try:
        entry = xa.xa_head.read_()
    except AttributeError:
        entry = xa.rnode
        node_type = entry.type_
        entry = cast("void *", entry)

        # Check if this is an old-style radix tree (Linux < 4.7)
        # by checking if radix_tree_node has 'path' member (which encodes height)
        # instead of 'shift' member
        try:
            # Try to access shift - if it exists, this is a newer radix tree
            node_struct_type = prog.type("struct radix_tree_node")
            node_struct_type.member("shift")
            # Has shift member - this is case 3 (Linux 4.7-4.19)
        except LookupError:
            # No shift member - check for path member (old-style, Linux < 4.7)
            try:
                node_struct_type.member("path")
                # Has path member but no shift - this is case 4 (old-style)
                yield from _xa_for_each_old_radix_tree(xa)
                return
            except LookupError:
                # Neither shift nor path - unexpected, fall through to case 3
                pass

        # Return > 0 if radix_tree_is_internal_node(), < 0 if
        # is_sibling_entry(), and 0 otherwise.
        def is_internal(slots: Optional[Object], entry_value: int) -> int:
            if (entry_value & 3) == 1:
                # slots must be a reference object, so address_ is never None.
                if slots is not None and (
                    slots.address_ <= entry_value < slots[len(slots)].address_  # type: ignore[operator]
                ):
                    return -1
                else:
                    return 1
            return 0

        # entry_to_node()
        def to_node(entry_value: int) -> Object:
            return Object(prog, node_type, entry_value - 1)

    else:
        node_type = prog.type("struct xa_node *")

        # Return > 0 if xa_is_node(), < 0 if xa_is_sibling(), and 0 otherwise.
        def is_internal(slots: Optional[Object], entry_value: int) -> int:
            if _xa_is_node(entry_value):
                return 1
            elif (entry_value & 3) == 2 and entry_value < 256:
                return -1
            else:
                return 0

        # xa_to_node()
        def to_node(entry_value: int) -> Object:
            return Object(prog, node_type, entry_value - 2)

        if not advanced:
            # We're intentionally redefining should_yield() for this case.
            def should_yield(entry_value: int) -> bool:  # noqa: F811
                return entry_value != 0 and entry_value != _XA_ZERO_ENTRY

    entry_value = entry.value_()
    internal = is_internal(None, entry_value)
    if internal > 0:
        stack = [_XAIteratorNode(to_node(entry_value), 0)]
    else:
        if internal == 0 and should_yield(entry_value):
            yield 0, entry
        return

    while stack:
        node = stack[-1]
        if node.next_slot >= len(node.slots):
            stack.pop()
            continue

        entry = node.slots[node.next_slot].read_()
        entry_value = entry.value_()

        index = node.index + (node.next_slot << node.shift)
        node.next_slot += 1

        internal = is_internal(node.slots, entry_value)
        if internal > 0:
            stack.append(_XAIteratorNode(to_node(entry_value), index))
        elif internal == 0 and should_yield(entry_value):
            yield index, entry


def xa_is_value(entry: Object) -> bool:
    """
    Return whether an XArray entry is a value.

    See :func:`xa_to_value()`.

    :param entry: ``void *``
    """
    return (entry.value_() & 1) != 0


def xa_to_value(entry: Object) -> Object:
    """
    Return the value in an XArray entry.

    In addition to pointers, XArrays can store integers between 0 and
    ``LONG_MAX``. If :func:`xa_is_value()` returns ``True``, use this to get
    the stored integer.

    >>> entry = xa_load(xa, 9)
    >>> entry
    (void *)0xc9
    >>> xa_is_value(entry)
    True
    >>> xa_to_value(entry)
    (unsigned long)100

    :param entry: ``void *``
    :return: ``unsigned long``
    """
    return cast("unsigned long", entry) >> 1


def xa_is_zero(entry: Object) -> bool:
    """
    Return whether an XArray entry is a "zero" entry.

    A zero entry is an entry that was reserved but is not present. These are
    only visible to the XArray advanced API, so they are only returned by
    :func:`xa_load()` and :func:`xa_for_each()` when ``advanced = True``.

    >>> entry = xa_load(xa, 10, advanced=True)
    >>> entry
    (void *)0x406
    >>> xa_is_zero(entry)
    True
    >>> xa_load(xa, 10)
    (void *)0

    :param entry: ``void *``
    """
    return entry.value_() == _XA_ZERO_ENTRY
