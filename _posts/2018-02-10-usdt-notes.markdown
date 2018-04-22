---
layout: post
title: "USDT for reliable Userspace event tracing"
date: 2018-02-10 23:22:39 -0800
comments: true
categories: 
---
Userspace program when compiled to native can place tracepoints in them using a USDT (User Statically Defined Tracing), the details of the tracepoints (such as address and arguments) are placed as a note in the ELF binary for other tools to interpret. This at first seems much better than using Uprobes directly on userspace functions, since the latter not only needs symbol information from the binary but is also at the mercy of compiler optimizations and function inlining. In Android we directly write into `tracing_mark_write` for userspace tracepoints. While this has its advantages (simplicity), it also means that the only way to "process" the trace information for ALL tracing usecases is by reading back the trace buffer to userspace and processing them offline, thus needing a full user-kernel-user round trip. Further its not possible to easily create triggers on these events (dump the stack or stop tracing when an event fires, for example). Uprobes event tracing on the other hands gets the full benefit that ftrace events do. Also all userspace emitted trace data is a string which has to be parsed and post-processed. USDT seems a nice way to solve some of these problems, since we can then use the user-provided data and create Uprobes at the correct locations, and process these on the fly using BCC without things ever hitting the trace buffer or returning to userspace.

This simple article is based on some notes I made while playing with USDT probes. To build a C program with an SDT probe, you can just include `sdt.h` and `sdt-config.h` from the Ubuntu systemtap-sdt-devel package, which works for both arm64 and x86.

C program can be as simple as:
```
#include "sdt.h"
int main() {
	DTRACE_PROBE("test", "probe1");
}
```
One compiling this, a new `.note.stapsdt` ELF section is created, which can be read by:
`readelf -n <bin-path>`
```
Displaying notes found in: .note.stapsdt
  Owner                 Data size	Description
  stapsdt              0x00000029	NT_STAPSDT (SystemTap probe descriptors)
    Provider: "test"
    Name: "probe1"
    Location: 0x0000000000000664, Base: 0x00000000000006f4, Semaphore: 0x0000000000000000
    Arguments: 
```
Here there are no arguments, however we can use `DTRACE_PROBE2` to pass more, for example for 2 of them:
```
int main() {
	int a = 1;
	int b = 2;

	DTRACE_PROBE2("test", "probe1", a, b);
}
```
The `readelf` tool now reads:
```
Displaying notes found in: .note.stapsdt
  Owner                 Data size	Description
  stapsdt              0x00000040	NT_STAPSDT (SystemTap probe descriptors)
    Provider: "test"
    Name: "probe1"
    Location: 0x0000000000000672, Base: 0x0000000000000704, Semaphore: 0x0000000000000000
    Arguments: -4@-4(%rbp) -4@-8(%rbp)
```

Notice how the arguments show exactly how to access the parameters at the location. In this case, we know the arguments are on the stack and at momention offsets from the base pointer.

Compiling with `-f-omit-frame-pointer` shows the following in `readelf`:
```
Displaying notes found in: .note.stapsdt
  Owner                 Data size	Description
  stapsdt              0x00000040	NT_STAPSDT (SystemTap probe descriptors)
    Provider: "test"
    Name: "probe1"
    Location: 0x0000000000000670, Base: 0x0000000000000704, Semaphore: 0x0000000000000000
    Arguments: -4@-4(%rsp) -4@-8(%rsp)
```

Without the base pointer, the compiler relies on the stack pointer for the arguments. However notice that even though the C program is identical to the previous example, the "arguments" in the note section of the ELF has changed. This dynamic nature is one of the key reasons why SDT probes are so much better than say using `perf probe` directly to install Uprobes in the wild. With some help userspace and the compiler, we have more reliable information to access arguments without needing any DWARF debug info.

Compiling with ARM64 gcc shows a slightly different output for the Arguments. Note that that Arguments is just a string which can be post processed by tools to fetch the probe data.
```
Displaying notes found in: .note.stapsdt
  Owner                 Data size	Description
  stapsdt              0x0000003f	NT_STAPSDT (SystemTap probe descriptors)
    Provider: "test"
    Name: "probe1"
    Location: 0x000000000000077c, Base: 0x0000000000000820, Semaphore: 0x0000000000000000
    Arguments: -4@[sp, 12] -4@[sp, 8]
```

USDT limitations as I see it:

- No information about types is stored. This is kind of sad, since now inorder to know what do with the values, one needs more information. These tracepoints were used with DTrace and SystemTap and turns out the scripts that probe these tracepoints are where the type information is stored or assumed. Uprobes tracer supports "string" but without knowing that the USDT is a string value, there's no way a Uprobe can be created on it, since all the stap note tells us is that there's a pointer there (who knows if its a 64-bit integer or a character pointer, for example).

- Argument names are also not stored. This means arguments have to be in the same order in the debug script as they are in the program being debugged.

It seems with a little bit of work, both these things can be added. Does that warrant a new section or can the stapsdt section be augment without causing breakage of existing tools? I don't know yet.

SDT parsing logic
-----------------
BCC has a USDT parser written to extract probes from the ELF. Read `parse_stapsdt_note` in `src/cc/bcc_elf.c` in the BCC tree for details.

Dynamic programming languages
-----------------------------
Programs that are interpretted can't provide this information ahead of time. The [libstapsdt](https://github.com/sthima/libstapsdt#how-it-works) tries to solve this by [creating a shared library on the fly](https://github.com/sthima/libstapsdt/blob/6045277309ff0425322bed5e71393ce5c8fa1344/src/libstapsdt.c#L89) and linking to it from the dynamic program. This seems a bit fragile but appears to have users. There are wrappers in Python and Nodejs. [Check this article](https://medium.com/sthima-insights/we-just-got-a-new-super-power-runtime-usdt-comes-to-linux-814dc47e909f) for more details.

Open question I have:
* Do any existing Linux tools handle USDT strings? Uprobe tracer does support strings, so the infrastructure seems to be there. I didn't see any hints of this in [BCC](src/cc/usdt/usdt_args.cc). Neither does [libstapsdt](https://github.com/sthima/libstapsdt/blob/6045277309ff0425322bed5e71393ce5c8fa1344/src/libstapsdt.h#L14) seem to have this.

Other ideas
-----------
* Creating Uprobes on the fly when a process is loaded: Ideally speaking, if the ELF note section had all the information that the kernel needed, then we could create the Uprobe events for the uprobe trace events at load time and keep them disabled without needing userspace to do anything else. This seems crude at first, but in the "default" case, it would still have almost no-overhead. This does mean that all the information Uprobe tracing needs will have to be stored in the note section. The other nice thing about this is, you no longer need to know the PID of all processes with USDTs in them. EDIT: This idea is flawed. Uprobes are created before a process is loaded AFAIU now, using the binary path of the executable and libraries. What's more useful is to maintain a cache of all executables in the file system, and their respective instrumentation points. Then on boot up, perhaps we can create all necessary uprobes from early userspace.

* For dynamic languages, `libstapsdt` seems great, but it feels a bit hackish since it creates a temporary file for the stub. Perhaps uprobes can be created after the temporary file is dlopen'ed and then the file can be unlinked if it hasn't been already, so that there aren't any more references to the temporary file in the file system. Such a temporary file could also probably be in a RAM based file system perhaps.

References
----------
1. [Brendan Gregg's USDT ftrace page](http://www.brendangregg.com/blog/2015-07-03/hacking-linux-usdt-ftrace.html)
2. [Uprobe tracer in the kernel](https://www.kernel.org/doc/Documentation/trace/uprobetracer.txt)
3. [USDT for dynamic languages](https://medium.com/sthima-insights/we-just-got-a-new-super-power-runtime-usdt-comes-to-linux-814dc47e909f)
4. [Sasha Goldstein's "Next Generation Linux Tracing With BPF" article](https://dzone.com/articles/next-generation-linux-tracing)
5. [SystemTap SDT implementation](https://sourceware.org/systemtap/wiki/UserSpaceProbeImplementation)







