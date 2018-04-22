---
layout: post
title: "Design of fork followed by exec in Linux"
date: 2014-04-22 23:06:11 -0500
comments: true
categories: 
---
A lot of folks ask *why do you have to do fork and then an exec, to execute a new program?* and *why can't it be done in one step?*, or *why does fork [create a copy-on-writed address space](http://man7.org/linux/man-pages/man2/fork.2.html), to only have it thrown away later when you do an exec?*. So I decided do a small write up about this topic.
 
On a separate note, firstly it is important to remember that `fork` is not used for threading, its primary use is to create a separate process, that is a child of the parent process that called `fork`.

Normally one might think that doing a `fork` and separate `exec` can be combined in one step, and it probably should be. But there are applications of maintaining this separation. Here's a [post that explains](http://stackoverflow.com/questions/1345320/applications-of-fork-system-call) why you might need to do just a `fork` call. Summarizing the post, you may need to setup some initial data and "fork" a bunch of workers. All these works are supposed to execute in *their own* address space and share *only the initial data*. In this case, copy-on-write is extremely useful since the initial data can be shared in physical memory and forking this way would be extremely cheap. The kernel marks all these shared pages as read only, and makes writable copies of shared data when they are written to.

There is a small overhead if `fork` is followed immediately by an `exec` system call, since the copy-on-write shared address space is of no use and is thrown away anyway. Combining both the `fork`, `exec` in this case might might have some advantages, reducing this overhead.

Linux Implementation of Copy-on-write (COW) for shared Virtual Memory Areas
---------------------------------------------------------------------------
Some of this COW code that executes on a fork can be found in `mm/memory.c`. There is an is_cow function to detect if a virtual memory area (a region of virtual memory, see `/proc/self/maps`) is copy-on-write.

``` c
static inline bool is_cow_mapping(vm_flags_t flags)
{
        return (flags & (VM_SHARED | VM_MAYWRITE)) == VM_MAYWRITE;
}
```
A VMA (Virtual Memory Area) is a contiguous segment of virtual memory belonging to a particular process. Every VMA has a bunch of VM_ flags associated with it. `VM_MAYWRITE`, relevant to the above code, is used to mark that a mapped region can be changed to writable by mprotect system call. It is possible that a memory region is initially readonly and the user wants to make it writable. `VM_MAYWRITE` gives that permission. Note that if if the kernel doesn't set `VM_MAYWRITE`, then the region is automatically not COW because there is no question of writing to it.

When a memory mapping is created via the mmap system call, and if `MAP_SHARED` is passed in flags, the `VM_SHARED` bit is set for the VMA and as a result the region is not copy-on-write (The above is_cow_mapping function returns false). By definition, shared memory regions are just that - shared. So no need copy-on-write. In other words, If the VMA is a shared mapping or is a read only mapping, then it isn't a COW mapping.

Let's take the example of mapping a file using mmap,

By default in the kernel on VMA creation, the VMA flags is set to `VM_SHARED = 0` and `VM_MAYWRITE = 1`. Now if mmap is asked it to create a shared mapping of a file by passing it `MAP_SHARED` flag, for example, that can be shared with other processes that are being forked, then the `VM_SHARED` bit is set to 1 for that VMA. Additionally if the file is opened in read only mode, then `VM_MAYWRITE` is set to 1. This has the effect of making is_cow_mapping return false. Ofcourse, the shared mapping doesn't need to be a COW.

On the other hand, if `MAP_PRIVATE` is passed in the flags to mmap, then `VM_SHARED` bit is set to 0, and `VM_MAYWRITE` remains at 1 (regardless of whether the file is read or write opened, since writes will not be carried to the underlying file). This makes is_cow_mapping return true. Indeed, private mappings should be copy-on-write enabled.

You can see the code I'm talking about [conveniently here](http://lxr.free-electrons.com/source/mm/mmap.c#L1284).

The important point here is that every mapping is either a COW mapping or not a COW mapping. During the `clone` system call which is called by `fork` library call internally, if the `CLONE_VM` flag is not passed to `clone` as is the case internally within `fork`, then all the VMA mappings of the parent process are copied to the child, including the page table entries. In this case, any writes to COW mappings should trigger a copy on write. The main thing to note is the children *inherit* the COW property of all the copied VMA mappings of its parent and don't need to be explictly marked as COW.

However, If `CLONE_VM` is passed, then the VMAs are not copied and the memory descriptor of the child and the parent process are the same, in this case the child and parent share the same address space and are thus are threads. [See for yourself](http://lxr.free-electrons.com/source/kernel/fork.c#L879). COW or no COW doesn't matter here.

So here's a question for you, For N `clone` system calls with `!CLONE_VM` passed for spawning N threads, we can just create as many VMA copies as we want each time, the COW mappings will take care of themselves. Right? Almost! There's more work... the *physical pages* of both the original VMA and the copy VMA have to be marked as read-only. That's the only way Copy-on-write of those will be triggered by the CPU when those pages are written to. Here's the code in [copy_one_pte](http://lxr.free-electrons.com/source/mm/memory.c#L849) that sets this up:
``` c
         /*
          * If it's a COW mapping, write protect it both
          * in the parent and the child
          */
         if (is_cow_mapping(vm_flags)) {
                 ptep_set_wrprotect(src_mm, addr, src_pte);
                 pte = pte_wrprotect(pte);
         }
``` 
There you go, now when the COW memory region is written to, a page fault happens, and the page fault handler knows that the VMA of the faulting page is a COW and that's what triggered the page fault. It can then create a copy of the page and restart the faulting instruction, this time removing the write protection if there aren't any others sharing the VMA. So in short, fork+exec can be expensive if you had done lots of `fork` calls on a process with a lot of large files. Since all this copying business is wasted on doing a subsequent `exec` system call.

There is one optimization however, why should you have to do this marking for pages that are not physically present in memory? Those will fault anyway. So the above code is *not run* if the page is not present, nicely done by checking for `!pte_present(pte)` to be true before the preceding code.

Please share any comments you may have in the comments section.
