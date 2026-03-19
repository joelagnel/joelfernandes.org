---
layout: post
title: "Rust Macros That Are Unsafe Are Not Lint Friendly"
date: 2026-03-19
---

When a Rust macro needs to perform unsafe operations internally, you have a design choice: does the macro itself contain the `unsafe` block, or does the caller write it? It sounds like a small API decision, but it has surprising consequences for Clippy's lint system. I ran into this while upstreaming [`CList`](https://lore.kernel.org/all/20260317201710.712702-2-joelagnelf@nvidia.com/), a Rust abstraction for iterating C linked lists in the Linux kernel, and the resulting LKML discussion surfaced several interesting perspectives.

This post walks through the problem, the three Clippy lints that interact (sometimes poorly) with unsafe macros, the different approaches proposed by kernel Rust developers, and test programs I wrote to study the behavior.

## The CList Macro

CList provides a way for Rust code to iterate over C's `struct list_head` linked lists. The `clist_create!` macro computes the byte offset of the `list_head` field at compile time via const generics and creates a typed `CList` object:

{% raw %}
```rust
#[macro_export]
macro_rules! clist_create {
    (unsafe { $head:ident, $rust_type:ty, $c_type:ty, $($field:tt).+ }) => {{
        // Compile-time check that field path is a `list_head`.
        // SAFETY: `p` is a valid pointer to `$c_type`.
        let _: fn(*const $c_type) -> *const $crate::bindings::list_head =
            |p| unsafe { &raw const (*p).$($field).+ };

        // Calculate offset and create `CList`.
        const OFFSET: usize = ::core::mem::offset_of!($c_type, $($field).+);
        // SAFETY: The caller of this macro is responsible for ensuring safety.
        unsafe { $crate::interop::list::CList::<$rust_type, OFFSET>::from_raw($head) }
    }};
}
```
{% endraw %}

The caller uses it like this:

```rust
// SAFETY: `head` is valid and initialized, items are `SampleItemC` with
// embedded `link` field, and `Item` is `#[repr(transparent)]` over `SampleItemC`.
let list = clist_create!(unsafe { head, Item, SampleItemC, link });
```

Notice the `unsafe` keyword inside the macro arguments. This is not a real Rust `unsafe` block. It's a token pattern matched by the macro's `(unsafe { ... })` arm. The idea is to make the call site *look* unsafe, signaling to the reader that safety obligations exist. The actual `unsafe` blocks live inside the macro expansion.

This pattern creates problems with three Clippy lints.

## The Three Clippy Lints

### 1. `unnecessary_safety_comment`

Clippy requires `// SAFETY:` comments only before real `unsafe` blocks. Since the `unsafe` in `clist_create!(unsafe { ... })` is just a matched token, Clippy doesn't see any unsafe block at the call site. If you write a `// SAFETY:` comment (as you should for any unsafe operation), Clippy complains:

```
error: statement has unnecessary safety comment
  --> src/test_error.rs:23:5
   |
23 |     let _a = create_from_raw!(unsafe { p });
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
help: consider removing the safety comment
  --> src/test_error.rs:22:8
   |
22 |     // SAFETY: `p` points to a valid i32.
   |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

This is a false positive. The macro *does* perform unsafe operations. The caller *should* document why the preconditions are satisfied. But Clippy tells them to delete the comment.

Alice Ryhl (Google) [flagged this](https://lkml.org/lkml/2026/3/18/1229) during review of v13. Miguel Ojeda had [previously identified the same issue](https://lkml.org/lkml/2026/3/12/1852) on v12, noting that "the supposed to be `unsafe` block does not count as one for Clippy."

### 2. `undocumented_unsafe_blocks`

Inside the macro, the `unsafe` blocks need `// SAFETY:` comments too. But when the macro is expanded at a call site, Clippy checks whether the *expanded* code has safety comments in the right positions. The comment placement that looks correct inside the macro definition can end up in the wrong position after expansion.

Alexandre Courbot (NVIDIA) [hit this](https://lkml.org/lkml/2026/3/19/1111) when using `clist_create!` in the GPU buddy allocator:

```
warning: unsafe block missing a safety comment
  --> rust/kernel/interop/list.rs:335:17
   |
335 |     |p| unsafe { &raw const (*p).$($field).+ };
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

The fix was to move the `// SAFETY:` comment to directly above the closure (on the `|p|` line) rather than above the `let` binding.

### 3. `macro_metavars_in_unsafe`

This lint catches macros that expand caller-provided expressions (`$expr`) inside an `unsafe` block. The concern is that the caller can sneak arbitrary code into an unsafe context without writing `unsafe` themselves. Using `:ident` instead of `:expr` for `$head` avoids this, since an identifier is just a name and cannot contain arbitrary unsafe code.

This is why `clist_create!` uses `$head:ident` rather than `$head:expr`. As I noted in the [thread](https://lkml.org/lkml/2026/3/18/2206): "Not doing `:ident` was causing clippy errors."

Alice Ryhl [pointed out](https://lkml.org/lkml/2026/3/18/1229) this forces callers to put the pointer in a local variable first, rather than passing an expression like `&raw mut my_struct.head` directly. A minor ergonomic cost, but one that avoids a legitimate safety concern.

## The Approaches

The LKML discussion produced four distinct approaches to this problem. Here they are, with test programs demonstrating each.

### Approach 1: `SAFETY*:` Comment Hack (Miguel's Temporary Fix)

Miguel Ojeda [suggested](https://lkml.org/lkml/2026/3/12/1852) using `// SAFETY*:` instead of `// SAFETY:` as a temporary workaround. Clippy's lint only triggers on the exact string `SAFETY:`, so the asterisk dodges the false positive while still communicating intent to human readers.

```rust
// SAFETY*: `p` points to a valid i32 on the stack.
let _a = create_from_raw!(unsafe { p });
```

This compiles cleanly with Clippy. Miguel was clear this is a [stopgap](https://lkml.org/lkml/2026/3/18/2197): "I suggested it as a temporary thing we could do if we want to use that 'fake `unsafe` block in macro matcher' pattern more and more."

Danilo Krummrich (Red Hat) [agreed](https://lkml.org/lkml/2026/3/19/1202) this was acceptable for now: "if this is what we want to do in such cases, we should probably document it somewhere in the coding guidelines."

### Approach 2: Remove Fake-Unsafe, Let Caller Write Real `unsafe` (Gary's Approach)

Gary Guo [proposed](https://lkml.org/lkml/2026/3/19/1029) the most radical simplification: drop the `unsafe` keyword from the macro entirely and don't wrap the internal unsafe operations. The caller is forced to write a real `unsafe { }` block, and all Clippy lints work perfectly:

```rust
macro_rules! create_from_raw_no_fake {
    ($ptr:ident) => {
        deref_ptr($ptr)   // unsafe call -- NOT wrapped
    };
}

// SAFETY: `p` points to a valid i32 on the stack.
let _b = unsafe { create_from_raw_no_fake!(p) };
```

The compiler enforces that the call site is inside an `unsafe` block (because the macro expands to an unsafe function call), and Clippy correctly associates the `// SAFETY:` comment with it.

Gary's reasoning: "I am not sure why the macro should have unsafe keyword in it, rather than just being `clist_create(a, b, c, d)` and just have user write unsafe."

Danilo [pushed back](https://lkml.org/lkml/2026/3/19/1114): if the macro doesn't wrap its internal unsafe operations, "then the calls within the macro are not justified individually." Gary [clarified](https://lkml.org/lkml/2026/3/19/1141): since the only safety justification is "the caller of this macro is responsible for ensuring safety" (which defers entirely to the caller anyway), there's no loss. For macros that expand to *items* (like `impl_device_context_deref!`), the caller *can't* write an outer `unsafe {}`, so the fake-unsafe pattern is the only option. But for expression-position macros like `clist_create!`, the caller can.

### Approach 3: Outer `unsafe` Block in Macro Expansion (Joel's Revised Approach)

I proposed [reworking the macro](https://lkml.org/lkml/2026/3/18/2168) so the `unsafe` keyword is part of the expansion rather than just a pattern match. The macro still matches `unsafe { ... }` for visual clarity, but emits an outer `unsafe { }` that wraps the entire body:

```rust
macro_rules! create_from_raw_outer {
    (unsafe { $ptr:ident }) => (
        // SAFETY: The caller of this macro ensures `$ptr` is valid.
        unsafe { {
            deref_ptr($ptr)
        } }
    );
}

// SAFETY: `p` points to a valid i32 on the stack.
let _a = create_from_raw_outer!(unsafe { p });
```

This keeps the `unsafe` visual marker at the call site while making the safety comments work correctly. Alex Courbot [liked this approach](https://lkml.org/lkml/2026/3/19/1111): "it preserves the expected use of `SAFETY:` without that confusing `*`."

### Approach 4: Upstream Clippy Fix (Future)

Miguel noted that the right long-term fix is to teach Clippy to recognize the "fake unsafe block in macro matcher" pattern: "if we plan to use the pattern more, then I am happy to ask upstream if it would make sense for Clippy to recognize it." He also pointed to the [Rust-for-Linux wishlist](https://github.com/Rust-for-Linux/linux/issues/354) for proper `unsafe` macro support in the language itself, which would let individual macro arms be marked as unsafe.

## Studying the Behavior

To verify these behaviors independently of the kernel build system, I wrote test programs using vanilla Rust and Clippy. All tests below use `rustc 1.93.0` / `clippy 0.1.93`. Note: the kernel uses nightly toolchains where `macro_metavars_in_unsafe` is enabled via `-W clippy::all`; on stable, you may need to enable it explicitly.

### Test: Fake `unsafe` Blocks Are Invisible to Clippy

{% raw %}
```rust
#![deny(clippy::unnecessary_safety_comment)]
#![deny(clippy::undocumented_unsafe_blocks)]

unsafe fn deref_ptr(p: *const i32) -> i32 {
    // SAFETY: Caller guarantees `p` is valid.
    unsafe { *p }
}

macro_rules! create_from_raw {
    (unsafe { $ptr:ident }) => {{
        // SAFETY: The caller ensures `$ptr` is valid.
        unsafe { deref_ptr($ptr) }
    }};
}

fn main() {
    let x: i32 = 42;
    let p = &x as *const i32;

    // SAFETY: `p` points to a valid i32.          <-- Clippy ERROR
    let _a = create_from_raw!(unsafe { p });
}
```
{% endraw %}

Running `clippy-driver --edition 2024 -D clippy::unnecessary_safety_comment`:

```
error: statement has unnecessary safety comment
  --> test_error.rs:23:5
   |
23 |     let _a = create_from_raw!(unsafe { p });
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

**Clippy does not see `unsafe` in the macro matcher as a real unsafe block.** The safety comment is rejected.

### Test: `SAFETY*:` Workaround Compiles Clean

```rust
    // SAFETY*: `p` points to a valid i32. (Clippy workaround)
    let _a = create_from_raw!(unsafe { p });
```

Compiles with zero warnings. The asterisk makes Clippy ignore the comment while preserving the documentation intent.

### Test: Caller-Written `unsafe` Works Perfectly

```rust
macro_rules! create_from_raw_no_fake {
    ($ptr:ident) => {
        deref_ptr($ptr)
    };
}

fn main() {
    let x: i32 = 42;
    let p = &x as *const i32;

    // SAFETY: `p` points to a valid i32 on the stack.
    let _b = unsafe { create_from_raw_no_fake!(p) };
}
```

Compiles clean. Clippy correctly sees the real `unsafe {}` block and accepts the `// SAFETY:` comment.

### Test: Outer `unsafe` Wrapper Also Works

```rust
macro_rules! create_from_raw_outer {
    (unsafe { $ptr:ident }) => (
        // SAFETY: The caller ensures `$ptr` is valid.
        unsafe { {
            deref_ptr($ptr)
        } }
    );
}

fn main() {
    let x: i32 = 42;
    let p = &x as *const i32;

    // SAFETY: `p` points to a valid i32 on the stack.
    let _a = create_from_raw_outer!(unsafe { p });
}
```

Also compiles clean. The real `unsafe {}` in the expansion makes Clippy happy, and the caller can write their own `// SAFETY:` on the outer real unsafe.

## Where Things Stand

For CList v13, we went with the `SAFETY*:` workaround plus constraining `$head` to `:ident`. Danilo is picking up the patch as a dependency for the GPU buddy allocator. The longer-term fix depends on either:

1. Gary's approach gaining consensus (drop fake-unsafe for expression-position macros),
2. Upstream Clippy learning to recognize the macro-matcher pattern, or
3. Rust itself gaining [unsafe macro support](https://github.com/Rust-for-Linux/linux/issues/354) as a language feature.

The core tension is between two valid goals: macros should visually signal that they involve unsafe operations (the fake-unsafe pattern), and Clippy's lint infrastructure should be able to verify safety documentation (which requires real `unsafe` blocks). Right now, you can't have both.

For anyone writing Rust macros that wrap unsafe operations, the practical takeaway: **use `:ident` over `:expr` for arguments that end up inside `unsafe` blocks**, and be aware that Clippy's safety-comment lints don't see token-level `unsafe` in macro matchers. If your macro expands to an expression (not an item), consider letting the caller write the real `unsafe {}` block.

---

*LKML thread: [\[PATCH v13 1/1\] rust: interop: Add list module for C linked list interface](https://lore.kernel.org/all/20260317201710.712702-2-joelagnelf@nvidia.com/)*

*Related Rust-for-Linux issue: [#354 - Unsafe macros](https://github.com/Rust-for-Linux/linux/issues/354)*
