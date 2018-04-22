---
layout: page
title: "technical resources"
date: 2014-04-25 13:29
comments: true
sharing: true
footer: true
---
#### My Presentations
[ELCE 2016: Debugging Real-Time issues in Linux](/technical-resources/elce2016-debug-rt.pdf)

#### Low-level, ASM
* [Tim Bird's Ftrace function-graph paper](http://elinux.org/images/0/0c/Bird-LS-2009-Measuring-function-duration-with-ftrace.pdf)

#### Memory Management
##### Ordering and Barriers
Basics

- [Paul M's paper on Why memory barriers](http://joelagnel.github.com/papers/whymb.2010.07.23a.pdf)
- [Linux DMA basics (ARM perspective)](http://infocenter.arm.com/help/topic/com.arm.doc.dai0228a/index.html)
- [ARM A8  architecture  - memory subsystem](http://www.arm.com/files/pdf/A8_Paper.pdf)
- [Leif L. at ELC - more barrier references at end of slide](http://elinux.org/images/f/fa/Software_implications_memory_systems.pdf)
- [Leif L. on arm.com blogs](http://community.arm.com/groups/processors/blog/2011/03/22/memory-access-ordering--an-introduction)
- [LWN on ACCESS_ONCE](http://lwn.net/Articles/508991/)
- [C11 atomic variables and the kernel](http://lwn.net/Articles/586838/)
- [C11 memory models](http://en.cppreference.com/w/cpp/atomic/memory_order)
- [C11 memory models - gcc wiki](http://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync)
- [SMP Primer for Android](http://developer.android.com/training/articles/smp.html)
- [Effect of Write Combining in L1 Buffers](http://mechanical-sympathy.blogspot.com/2011/07/write-combining.html)
- [Shared Memory Consistency Models:A Tutorial](http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-95-7.pdf)

Advanced

* [Paul M's article explaining the memory modeling tools - validating memory barriers](http://lwn.net/Articles/470681/)
* [Paper on POWER memory model, similar to ARM](http://www.cl.cam.ac.uk/~pes20/ppc-supplemental/pldi105-sarkar.pdf)

Linking Loading, shared libraries

* [Hands-on dissection of PLT and GOT](https://www.technovelty.org/linux/plt-and-got-the-key-to-code-sharing-and-dynamic-libraries.html)

Misc

* [ACP (Accelerator Coherency Port) for coherent access by ARM-external Masters](http://www.googoolia.com/downloads/papers/sadri_fpgaworld_ver2.pdf)

#### Electronics
* [NAND flash internals - great primer](http://vjswami.com/2013/11/11/a-primer-on-flash-and-a-look-into-the-challenges-of-designing-an-all-flash-array/)
* [NAND flash Organization - lots of info](http://www.lostcircuits.com/mambo//index.php?option=com_content&task=view&id=72&Itemid=46&limit=1&limitstart=4)

##### Super capacitors
* [Good video on Super Cap + Boost Converter circuits](http://www.instructables.com/id/The-Forever-Rechargeable-VARIABLE-Super-Capacitor-/)
* [Capacitor Basics](http://www.schoolphysics.co.uk/age16-19/Electricity%20and%20magnetism/Electrostatics/text/Capacitor_charge_and_discharge/index.html)

#### Drones
* [Linux and UAVs, a great overview of some of the projects](https://www.linux.com/news/embedded-mobile/mobile-linux/773231-linux-takes-to-the-skies-in-drones)
