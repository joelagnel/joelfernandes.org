---
layout: post
title: "ARMv8: flamegraph and NMI support"
date: 2016-12-31 22:29:26 -0700
comments: true
categories: 
---

Non-maskable interrupts (NMI) is a really useful feature for debugging, that hardware can provide. Unfortunately ARM doesn't provide an out-of-the-box NMI interrupt mechanism. This post shows a flamegraph issue due to missing NMI support, and the upstream work being done to simulate NMI in ARMv8.

Some great Linux kernel features that rely on NMI to work properly are:

* Backtrace from all CPUs: A number of places in the kernel rely on dumping the stacks of all CPUs at the time of a failure to determine what was going on. Some of them are [Hung Task detection](http://lxr.free-electrons.com/source/kernel/hung_task.c), [Hard/soft lockup detector](http://lxr.free-electrons.com/source/Documentation/lockup-watchdogs.txt) and spinlock debugging code.

* Perf profiling and flamegraphs: To be able to profile code that runs in interrupt handlers, or in sections of code that disable interrupts, Perf relies on NMI support in the architecture. [flamegraphs](http://www.brendangregg.com/FlameGraphs/cpuflamegraphs.html) are a great visual representation of perf profile output. Below is a flamegraph I generated from perf profile output, that shows just what happens on an architecture like ARMv8 with missing NMI support. Perf is using maskable-interrupts on this platform for profiling:

{% img /images/nmi/flamegraph.png %}

As you can see in the area of the flamegraph where the arrow is pointed, a large amount of time is spent in `_raw_spin_unlock_irqrestore`. It can baffle anyone looking at this data for the first time, and make them think that most of the time is spent in the unlock function. What's actually happenning is because perf is using a maskable interrupt in ARMv8 to do its profiling, any section of code that disables interrupts will not be see in the flamegraph (not be profiled). In other words perf is unable to peek into sections of code where interrupts are disabled. As a result, when interrupts are reenabled during the `_raw_spin_unlock_irqrestore`, the perf interrupt routine then kicks in and records the large number of samples that elapsed in the interrupt-disable section but falsely accounts it to the _raw_spin_unlock_restore function during which the perf interrupt got a chance to run. Hence the flamegraph anomaly. It is indeed quite sad that ARM still doesn't have a true NMI which perf would love to make use of.

BUT! [Daniel Thompson](https://lkml.org/lkml/2016/8/19/583) has been hard at work trying to simulate Non-maskable interrupts on ARMv8. The idea is [based on using interrupt priorities](/misc/arm-irq-priortization-white-paper.pdf) and is the subject of the rest of this post.

NMI Simulation using priorities
-------------------------------
To simulate an NMI, Daniel creates 2 groups of interrupts in his patchset. One group is for all 'normal' interrupts, and the other for non-maskable interrupts (NMI). Non-maskable interrupts are assigned a higher priority than the normal interrupt group. Inorder to 'mask' interrupts in this approach, Daniel replaces the regular interrupt masking scheme in the kernel which happens at the CPU-core level, with setting of the interrupt controller's PMR (priority mask register). When the PMR is set to a certain value, only interrupts which have a higher priority than what's in the PMR will be signaled to a CPU core, all other interrupts will be silenced (masked). By using this technique, it is possible to mask normal interrupts while keeping the NMI unmasked all the time.

Just how does he do this? So, a small primer on interrupts in the ARM world.
ARM uses the GIC [Generic interrupt controller](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dai0176c/ar01s03s01.html) to prioritize and route interrupts to CPU cores. GIC interrupt priorties go from 0 to 255. 0 being highest and 255 being the lowest. By default, the kernel [assigns priority 0xa0 (192)](http://lxr.free-electrons.com/source/include/linux/irqchip/arm-gic.h?v=4.8#L57) to all interrupts. He changes this default priority from 0xa0 to 0xc0 (you'll see why).
He then defines what values of PMR would be consider as "unmasked" vs "masked". Masked is 0xb0 and unmasked is 0xf0. This results in the following priorities (greater numbers are lower priority).
```
0xf0 (240 decimal)  (11110000 binary) - Interrupts Unmasked (enabled)
0xc0 (192 decimal)  (11000000 binary) - Normal interrupt priority
0xb0 (176 decimal)  (10110000 binary) - Interrupts masked   (disabled)
0x80 (128 decimal)  (10000000 binary) - Non-maskable interrupts
```
In this new scheme, when interrupts are to be masked (disabled), the PMR is set to 0xf0 and when they are unmasked (enabled), the PMR is set to 0xb0. As you can see, setting the PMR to 0xb0 indeed masks normal interrupts, because 0xb0(PMR) < 0xc0(Normal), however non-maskable interrupts still stay unmasked as 0x80(NMI) < 0xb0(PMR). Also notice that inorder to mask/unmask interrupts, all that needs to be done is flip bit 7 in the PMR (0xb0 -> 0xf0). Daniel largely uses Bit 7 as the mask bit in the patchset.

Quirk 1: Saving of the PMR context during traps
-----------------------------------------------
Its suggested in the patchset that during traps, the priority value set in the PMR needs to be saved because it may change during traps. To facilitate this, Daniel found a dummy bit in the PSTATE register (PSR). During any exception, Bit 7 of of the PMR is saved into a PSR bit (he calls it the G bit) and restores it on return from the exception. Look at the changes to `kernel_entry` macro in the set for this code.

Quirk 2: Ack of masked interrupts
---------------------------------
Note that interrupts are masked before the GIC interrupt controller code can even identify the source of the interrupt. When the GIC code eventually runs, it is tasked with identifying the interrupt source. It does so by reading the `IAR` register. This read also has the affecting of "Acking" the interrupt - in other words, telling the GIC that the kernel has acknowledged the interrupt request for that particular source. Daniel points out that, because the new scheme uses PMR for interrupt masking, its no longer possible to ACK interrupts without first unmasking them (by resetting the PMR) so he temporarily resets PMR, does the `IAR` read, and restores it. Look for the code in `gic_read_iar_common` in his patchset to handle this case.

Open questions I have
---------------------
- Where in the patchset does Daniel mask NMIs once an NMI is in progress, or is this even needed?

Future work
-----------
Daniel has tested his patchset only on the foundation model yet, but it appears that the patch series with modifications should work on the newer Qualcomm chipsets that have the necessary GIC (Generic interrupt controller) access from the core to mess with IRQ priorities. Also, currently Daniel has only implemented CPU backtrace, more work needs to be done for perf support which I'll look into if I can get backtraces working properly on real silicon first.


