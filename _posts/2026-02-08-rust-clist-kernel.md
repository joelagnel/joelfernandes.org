---
layout: post
title: "CList: Iterating C Linked Lists from Rust in the Kernel"
date: 2026-02-08
---

The Linux kernel's doubly-circular linked list (`struct list_head`) is quite common. When writing Rust code that interfaces with C subsystems, we need a way to iterate over these lists safely. This post explains CList, a Rust abstraction for C linked lists I am [proposing](https://lore.kernel.org/all/20260120204303.3229303-1-joelagnelf@nvidia.com/) that uses const generics to create a new dynamic type (a feature similar to C++ [non-type template parameters](https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter)).

## The Problem

Rust in the kernel already has [linked list abstractions](https://github.com/torvalds/linux/blob/master/rust/kernel/list.rs), but they don't fit into the ownership model we need for interfacing with linked lists created by C. When C code creates and manages a linked list, Rust code can't claim ownership - it can only borrow and iterate. The existing Rust list types assume Rust owns the nodes, which doesn't work when interfacing with C.

My use case is the GPU buddy allocator: C code allocates memory blocks and links them together. Rust driver code needs to iterate over those blocks without taking ownership or modifying the list structure.

## The container_of Pattern in C

C's `list_for_each_entry` macro uses `container_of` to convert a `list_head` pointer to a pointer to the containing structure:

```c
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))
```

This requires knowing the byte offset of the `list_head` field within the containing struct. In Rust, we can compute this at compile time with `offset_of!` and propagate it through the type system using const generics. Unlike with C, we can do this once when we define the CList type dynamically and don't have to do it every time we iterate!

## Const Generic Offset Technique

The `CList` type carries the offset as a const generic parameter:

```rust
pub struct CList<'a, T, const OFFSET: usize> {
    head: &'a CListHead,
    _phantom: PhantomData<&'a T>,
}
```

When you create a `CList`, the `clist_create!` macro computes the offset and bakes it into the type at compile time:

{% raw %}
```rust
macro_rules! clist_create {
    ($head:expr, $rust_type:ty, $c_type:ty, $($field:tt).+) => {{
        const OFFSET: usize = ::core::mem::offset_of!($c_type, $($field).+);
        $crate::clist::CList::<$rust_type, OFFSET>::from_raw($head)
    }};
}
```
{% endraw %}

Your new dynamic CList type can also provide a custom iterator. No need to call container_of every time (and say a prayer you got the offset right!).

First, your generic CList type creates a new generic iterator using the same const generic:

```rust
impl<'a, T, const OFFSET: usize> CList<'a, T, OFFSET> {
    pub fn iter(&self) -> CListIter<'a, T, OFFSET> {
        CListIter {
            head_iter: CListHeadIter {
                current_head: self.head,
                list_head: self.head,
            },
            _phantom: PhantomData,
        }
    }
}
```

Then every time you iterate, it uses OFFSET to perform the container_of equivalent:

```rust
impl<'a, T, const OFFSET: usize> Iterator for CListIter<'a, T, OFFSET> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let head = self.head_iter.next()?;
        // The const generic OFFSET is used here to compute the item pointer
        Some(unsafe { &*head.as_raw().byte_sub(OFFSET).cast::<T>() })
    }
}
```

## Why Const Generics Matter

The offset is computed once at compile time and baked into the type. Type safety: you can't accidentally use the wrong offset due to bugs in your iterator body. The iterator created for you automatically does it. This is a key Rust principle - some problems can be moved into the type system in a way that reduces programmer bugs by not having to write code that might be error prone.

## Usage Example

This is how I am using it in Rust in the kernel. A clist_create macro is provided which automatically does the creation of the CList. You just have to pass it the container type and it'll do the rest. The CList creation macro is defined as follows. It just needs to know the C and Rust container types, a pointer to the head of the list, and the name of the list_head field:

```rust
clist_create!(head_ptr, RustType, CType, link_field)
```

And here is the full real world example:

```rust
// C struct with embedded list_head (from gpu_buddy allocator)
#[repr(C)]
struct gpu_buddy_block {
    link: bindings::list_head,
    // ... other fields
}

// Rust wrapper
#[repr(transparent)]
struct Block(Opaque<gpu_buddy_block>);

// AllocatedBlocks holds a CListHead that C code populates
struct AllocatedBlocks {
    list: CListHead,
    buddy: Arc<Mutex<GpuBuddy>>,
}

// Iterate over blocks from C-allocated list
let list = unsafe { clist_create!(allocated.list.as_raw(), Block, gpu_buddy_block, link) };
for block in list.iter() {
    // process each block
}
```

## Design Constraints

The current implementation is read-only. Mutable iteration would require more complex safety reasoning. C code must not modify the list while Rust is iterating.

That's by design. For nova-core's memory management, we just need to iterate over buddy allocator blocks that C code creates. Mutation can come later with more careful ownership work. Perhaps we can also make this new module play well with the existing list.rs code.

## Review Feedback

The review feedback has been positive. Special thanks to Alice Ryhl, Alexandre Courbot, Gary Guo, and Zhi Wang who shaped this work. I am also particularly impressed by the speed of review from the Linux kernel Rust community. They are quite dedicated to this project and that is quite refreshing to see.

## What's Next

CList is part of a [larger memory management infrastructure](https://lore.kernel.org/all/20260120204303.3229303-1-joelagnelf@nvidia.com/) for the nova-core GPU driver that I am working on. Quite exciting and it's coming together nicely: GPU buddy allocator, page tables, virtual memory management, and more. See the RFC v6 patches for the full implementation.
