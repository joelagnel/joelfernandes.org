---
layout: post
title: "Understanding critical section behavior with ftrace output"
date: 2016-03-12 11:08:38 -0800
comments: true
published: false
categories: 
---

The ftrace irqsoff, preempt and preemptirqsoff tracers are amazing tools identify paths in Linux kernel code that block tasks from getting CPU-time. Essentially during these paths, the scheduler has its hands tied behind its back. There are mainly 2 cases where this can happen.

Local Interrupts disabled (masked):
All CPU cores have a bit in a 'flags' or status register, that is used to mask interrupts on that particular CPU. For x86 architectures, its in a [FLAGS register](https://en.wikipedia.org/wiki/FLAGS_register) called the [IF bit](https://en.wikipedia.org/wiki/Interrupt_flag). With this bit set, timer interrupts wont be received without which the scheduler wont get a chance to run and schedule something else.

Preemption disabled:
Preemption is often disabled in the kernel when acquiring spinlocks. We went over the mechanics of spinlocks [previously](http://www.linuxinternals.org/blog/2014/05/07/spinlock-implementation-in-linux-kernel/). There are other cases where preepmtion may be disabled as well. With preemption turned off, the scheduler will not preempt an existing task and prevent giving some other task a chance to run.

Both these cases introduce latency to runnable tasks and so they are bad for predictablity and real-timeness. Disabling interrupts is worse than turning off preemption because this also has the added effect of introducing latency to other interrupts.

The tracers I just mentioned are called latency tracers and they trace the total latency and optionally the functions that are executed between when irqs are turned off and on, or when preempt is turned off and on. They are [documented well](https://www.kernel.org/doc/Documentation/trace/ftrace.txt) so I wont go over them here.

One thing I want to mention is about the preemptirqsoff tracer, it starts when when *either* preempt *or* irqs are turned off, and stops tracing only when *both* preemption and irqs are turned back on.
This makes the preemptirqsoff tracer excellent in tracing kernel code paths that are blocking a task because regardless of whether preemption or irqs causing the latency, it treats both these cases identical or interchangeable. Borrowing from the amazing documentation, the following example illustrates this:

```
Knowing the locations that have interrupts disabled or
preemption disabled for the longest times is helpful. But
sometimes we would like to know when either preemption and/or
interrupts are disabled.

Consider the following code:

    local_irq_disable();
    call_function_with_irqs_off();
    preempt_disable();
    call_function_with_irqs_and_preemption_off();
    local_irq_enable();
    call_function_with_preemption_off();
    preempt_enable();

The irqsoff tracer will record the total length of
call_function_with_irqs_off() and
call_function_with_irqs_and_preemption_off().

The preemptoff tracer will record the total length of
call_function_with_irqs_and_preemption_off() and
call_function_with_preemption_off().

But neither will trace the time that interrupts and/or
preemption is disabled. This total time is the time that we can
not schedule. To record this time, use the preemptirqsoff
tracer.
```

It appears though that function tracing in preempt tracers is a bit broken and I [posted a fix](https://lkml.org/lkml/2016/3/12/24) for a bug which prevent tracing functions which were called while interrupts were enabled and preemption is disabled. So in the above example, call_function_with_preemption_off wouldn't be traced. It still tracks the max latency and reports it, but it would omit the tracing of this function.

Coming back to the original topic of understanding critical section kernel behavior, the beautiful ftrace output for latency tracers show you a set of flags on the left beside every function traced which illustrates what is going on with preemption and interrupts in the traced function. Here I will go over a set of trace outputs and show you some interesting things.


