---
layout: post
published: true
title: "Single-stepping the kernel's C code"
comments: true
category: linux
---
Recently, I have had to use the GNU debugger (gdb) connected to a Qemu instance
of a RISC-V processor to step through some kernel code.

Turns out that the Linux kernel is compiled with gcc `-O2` flag for
optimizations it needs during the build. This causes several problems for a
debugger.  One of them is that the gdb command `info registers` will show
values as `<optimized out>`. Another issue is that single-stepping will make
the debugger jump back and forth across lines of code.

To circumvent this issue, I ended up with a hack that works well. I don't claim
this recommended or correct, but it makes it through the build and gdb works
fine. In my debugging, I have wanted to single-step through scheduler code in
the `__schedule` kernel function. For this purpose, all I have to do is add the
following to `kernel/sched/Makefile`.

```
CFLAGS_REMOVE_core.o := -O2
CFLAGS_core.o := -O0
```

This works brilliantly! What do you think? Let me know in the comments.

Some more tips:

* `CONFIG_DEBUG_INFO` is needed to ensure kernel has debug symbols for gdb to
  load, and ofcourse `CONFIG_DEBUG_KERNEL`.
* `CONFIG_FRAME_POINTER` should be enabled to ensure stack unwinding,
  backtraces work correctly.
* `CONFIG_GDB_SCRIPTS` is a bunch of useful gdb scripts automatically load when a
  vmlinux is gdb'd.

