---
layout: post
title: "Getting YouCompleteMe working for kernel development with vim"
comments: true
categories: [vim, productivity]
---
[YCM](https://github.com/Valloric/YouCompleteMe) is a pretty neat tool for
speedy kernel development in vim. Especially when you don't want to write a
whole bunch of code, and then deal with 100s of compiler errors. It is better
to fix the errors on the spot, as you write the code.  That's exactly what YCM
is best at since it constantly builds the code on the fly. The other great
thing about it among other things being, it will show you function prototypes
and so forth as you type, so you can call functions correctly with the right
set of parameters and types. YCM uses [clangd](https://clangd.llvm.org/) under
the hood which understands code navigation, helps with code completion, and so
forth. `clangd` is used by a whole plehtora of IDEs and tools. I remember using
it when I tried to get `vscode` working with ChromeOS source navigation (which
is actually a bigger project in terms of lines of code, than the Linux
kernel!).

Interestingly YCM does not use `cscope` even if it is available, and they do
not want to support it. Further, they reluctantly support `ctags`. Since I
don't use `ctags`, I am not sure how having tags available changes YCM
behavior and how it works with `clangd`. But that is something worth trying.

Getting YCM working with vim is pretty easy, but requires a few steps.

First you have to build a `compile_commands.json` file in the kernel root. To
do so, run:
```
bear -- make -j99 CC=clang
```
Note the `CC` variable passed to Make. It has to be `clang`, otherwise YCM
throws a tonne of errors when it is in use. I believe that is because clangd
incorrectly passes gcc-specific options to clang, which are not all supported.

Following
[this](https://stackoverflow.com/questions/30180064/how-to-setup-youcompleteme-for-kernel-and-device-driver-development)
stackoverflow post shows how to avoid some issues. This post is how I learnt to
use `bear`. Just use `bear` and ignore all other posts or articles that ask you
to create a `.ycm_extra_conf.py` file. I did not need to do that at all.

I faced the following issues:
1. First, there might some warnings in the code that arise when clangd/clang
try to build your code on the fly. In case these warnings are not legit, it is
best to ignore them. For that the `g:ycm_filter_diagnostics` vim variable can
be defined.  The above post shows an example.

2. Editing header files may not work well: This happens because header files
cannot really be "built". Further, sometimes header files require other header
files to be included, which may be done in the actual C file that includes the
header but not the header itself. However, vim/YCM does not know about that.
This issue can simply be fixed by including other header dependencies into the
header file being edited.

3. Kernel config macro dependencies: If a build has not happened yet, or you are
writing C code that depends on a new CONFIG option, a build may not have
happened yet, so the `autoconf` headers may not yet be available. This causes
YCM to not build those sections of code. A quick fix might be to add something
like this in the sources:
```
#ifndef CONFIG_FOO_BAR
#define CONFIG_FOO_BAR
#endif
```
Similar tricks can be employed to satisfy macros such as `IS_ENABLED(CONFIG_FOO_BAR)`.

Hope this helps, do you have any other tips or ideas to use YCM better? If so,
let me know in the comments!
