---
layout: post
published: true
title: "RCU-preempt: What happens on a context switch"
comments: true
category: [linux, kernel, rcu, scheduler]
---
Note: This article requires knowledge of RCU (read copy update) basics and its
different flavors.

RCU's main algorithm is to detect when it is free to reclaim objects that RCU
readers no longer need. The "RCU-sched" flavor of RCU does this by just
disabling preemption across the read section. So any time any of the CPUs is
not running in a preempt disabled section (such as with preemption off, or
interrupts off), then the CPU is said to be in a "quiescent state" (QS). Once
all CPUs reach a QS after the reclaimer filed a claim to release an object, the
object can be safely released. The time from when the request for RCU to
release an object to when RCU says its Ok to release it, is called the grace
period.

RCU-sched is kind of a big hammer, having readers disable preemption can have
poor performance effects. After all, read sections are expected to be light in
RCU. It can also effect real-time response of applications.

For this reason, preemptible RCU came about (also called RCU-preempt).
Obviously in this flavor, RCU reader sections can get preempted to run
something else.

A [recent discussion](https://www.mail-archive.com/linux-kernel@vger.kernel.org/msg1682346.html)
on LKML clarified to me that "preempted to run something else" not only covers
involuntary preemption but also voluntarily sleeping. This design is because,
with `PREEMPT_RT` kernels, "rt" version of spinlocks are actually mutexes that
can put the RCU reader to sleep.

So coming back to the point of this article, I want to go over what happens on
a context-switch. When the scheduler is called, we end up in `__schedule`
function. Here in the beginning `rcu_note_context_switch` is called with the
`preempt` parameter. The `preempt` parameter indicates if task blocked with
help of `schedule()` or if it was a kernel path (such as return from interrupts
or system calls) that called into the scheduler to preempt the currently
running task.

`rcu_note_context_switch` first calls `rcu_preempt_note_context_switch` for
RCU-preempt to take note. Lets discuss this function.

First note that the RCU-preempt flavor does warn you if you voluntarily sleep inside
an RCU read side section. I'm not sure how the "RT-spinlock" for RT kernels
doesn't get this warning. Probably they delete this warning in PREEMPT_RT
patchset, idk. The warning is `WARN_ON_ONCE(!preempt &&
t->rcu_read_lock_nesting > 0);`. But seems pretty clear to me a non-RT kernel
would scream with this warning if an RCU-preempt read section went to sleep.
Getting preempted is Ok but not voluntary sleeping according to this code! (see
side note in last para)

If the task being preempted is in a read-side RCU section, then (and only then)
it calls `rcu_preempt_ctxt_queue`. Here the task being preempted is added to a
list of blocked tasks. The reason why we need to add it is, RCU-preempt has 2
perspectives of Quiescent state (QS). Recall, a QS is reached whenever an
entity is not blocking the current grace period (GP). RCU-preempt considers 2
entity perspectives: Either the task, or the CPU. In the RCU-preempt world, if
a task that is currently in an RCU read section gets preempted, then the CPU
has reached a QS because it is no longer running the RCU-read section that is
blocking the GP. But now, the task has reached a non-QS (It is blocking the
GP). This list basically indicates this fact. If there are blocking tasks, then
the GP cannot complete even though the CPU reports its QS. [Paul Mckenney explains this here](https://lkml.org/lkml/2018/5/4/632). The other benefit of having a list of tasks is that preempted RCU read sections can be boosted. Paul Mckenney again came to the rescue to [explain this to me](https://lkml.org/lkml/2018/5/4/659).

Finally, you see that `rcu_preempt_note_context_switch` does report a QS. This
is because if the task was in a read section, it has just been added to the
blocked task list. If its not, then we just reached a QS for the CPU. Either
way we entered a CPU QS. So is recorded with a call to `rcu_preempt_qs();`.

Please go through the [Expedited GP document](https://www.kernel.org/doc/Documentation/RCU/Design/Expedited-Grace-Periods/Expedited-Grace-Periods.html) which also explains some of the RCU-preempt behaviors.

Side note: At the moment, I don't immediately see why by blocking in a RCU-preempt
section shouldn't be allowed.  Since we're tracking blocked tasks the same way
as preempted tasks, it should be possible to handle them the same way. They
both cause a CPU QS and a task non-QS to be entered, they both need priority
boosting. Perhaps the warning should be removed? Let me know your feedback in
the comments.
