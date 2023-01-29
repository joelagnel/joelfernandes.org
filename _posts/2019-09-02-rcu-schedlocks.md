---
layout: post
title: "Making sense of scheduler deadlocks in RCU"
comments: true
categories: rcu, scheduler, locking
---
Note: At the time of this writing, it is kernel v5.3 release. RCU moves fast
and can change in the future, so some details in this article may be obsolete.

The RCU subsystem and the task scheduler are inter-dependent. They both depend
on each other to function correctly. The scheduler has many data structures
that are protected by RCU. And, RCU may need to wake up threads to perform
things like completing grace periods and callback execution. One such case
where RCU does a wake up and enters the scheduler is
`rcu_read_unlock_special()`.

Recently Paul McKenney consolidated RCU flavors. What does this mean?

Consider the following code executing in CPU 0:
```
preempt_disable();
rcu_read_lock();
rcu_read_unlock();
preempt_enable();
```
And, consider the following code executing in CPU 1:
```
a = 1;
synchronize_rcu();  // Assume synchronize_rcu
                    // executes after CPU0's rcu_read_lock
b = 2;
```
CPU 0's execution path shows 2 flavors of RCU readers, one nested into another.
The `preempt_{disable,enable}` pair is an `RCU-sched` flavor RCU reader
section, while the `rcu_read_{lock,unlock}` pair is an `RCU-preempt` flavor RCU
reader section.

In older kernels (before v4.20), CPU 1's `synchronize_rcu()`  could return
_after_ CPU 0's `rcu_read_unlock()` but before CPU 0's `preempt_enable()`. This
is because `synchronize_rcu()` only needs to wait for the "RCU-preempt" flavor
of the RCU grace period to end.

In newer kernels (v4.20 and above), the RCU-preempt and RCU-sched flavors have
been consolidated. This means CPU 1's `synchronize_rcu()` is guaranteed to wait
for both of CPU 1's `rcu_read_unlock()` and `preempt_enable()` to complete.

Now, lets get a bit more detailed. That `rcu_read_unlock()` most likely does
very little. However, there are cases where it needs to do more, by calling
`rcu_read_unlock_special()`. One such case is if the reader section was
preempted. A few more cases are:

* The RCU reader is blocking an expedited grace period, so it needed to report
  a quiescent state quickly.
* The RCU reader is blocking a grace period for too long (~100 jiffies on my
  system, that's the default but can be set with
  `rcutree.jiffies_till_sched_qs` parameter).

In all these cases, the `rcu_read_unlock()` needs to do more work. However,
care must be taken when calling `rcu_read_unlock()` from the scheduler, that's
why this article on scheduler deadlocks.

One of the reasons `rcu_read_unlock_special()` needs to call into the scheduler
is priority de-boosting:  A task getting preempted in the middle of an RCU
read-side critical section results in blocking the completion of the critical
section and hence could prevent current and future grace periods from ending.
So the priority of the RCU reader may need to be boosted so that it gets enough
CPU time to make progress, and have the grace period end soon. But it also
needs to be de-boosted after the reader section completes. This de-boosting
happens by calling of the `rcu_read_unlock_special()` function in the outer
most `rcu_read_unlock()`.

What could go wrong with the scheduler using RCU? Let us see this in action.
Consider the following piece of code executed in the scheduler:
```
  reader()
	{
		rcu_read_lock();
		do_something();     // Preemption happened
                /* Preempted task got boosted */
		task_rq_lock();     // Disables interrupts
                rcu_read_unlock();  // Need to de-boost
		task_rq_unlock();   // Re-enables interrupts
	}
```
Assume that the `rcu_read_unlock()` needs to de-boost the task's priority. This
may cause it to enter the scheduler and cause a deadlock due to recursive
locking of RQ/PI locks.

Because of these kind of issues, there has traditionally been a rule that RCU
usage in the scheduler must follow:
```
“Thou shall not hold RQ/PI locks across an rcu_read_unlock() if thou not
holding it or disabling IRQ across both both the rcu_read_lock() +
rcu_read_unlock().”
```
More on this rule can be read [here as well](https://lwn.net/Articles/453002/).

Obviously, acquiring RQ/PI locks across the whole `rcu_read_lock()` and
`rcu_read_unlock()` pair would resolve the above situation. Since preemption
and interrupts are disabled across the whole `rcu_read_lock()` and
`rcu_read_unlock()` pair; there is no question of task preemption.

Anyway, the point is `rcu_read_unlock()` needs to be careful about scheduler
wake-ups; either by avoiding calls to `rcu_read_unlock_special()` altogether
(as is the case if interrupts are disabled across the entire RCU reader), or by
detecting situations where a wake up is unsafe. Peter Ziljstra says there's no
way to know when the scheduler uses RCU, so "generic" detection of the unsafe
condition is a bit tricky.

Now with RCU consolidation, the above situation actually improves. Even if the
scheduler RQ/PI locks are not held across the whole read-side critical sectoin,
but just across that of the `rcu_read_unlock()`, then that itself may be enough
to prevent a scheduler deadlock. The reasoning is: during the
`rcu_read_unlock()`, we cannot yet report a QS until the RQ/PI lock is itself
released since the act of holding the lock itself means preemption is disabled
and that would cause a QS deferral. As a result, the act of priority
de-boosting would also be deferred and prevent a possible scheduler deadlock.

However, RCU consolidation introduces even newer scenarios where the
`rcu_read_unlock()`  has to enter the scheduler, if the "scheduler rules" above
is not honored, as explained below: 

Consider the previous code example. Now also assume that the RCU reader is
blocking an expedited RCU grace period. That is just a fancy term for a grace
period that needs to end fast. These grace periods have to complete much more
quickly than normal grace period. An expedited grace period causes currently
running RCU reader sections to receive IPIs that [set a
hint](https://github.com/joelagnel/linux-kernel/blob/rcu/rcu-check-unsafe-scheduler-use-2/kernel/rcu/tree_exp.h#L641).
Setting of this hint results in the outermost `rcu_read_unlock()` calling
`rcu_read_unlock_special()`, which otherwise would not occur.
When `rcu_read_unlock_special()` gets called in this scenario, it tries to get
more aggressive once it [notices
that](https://github.com/joelagnel/linux-kernel/blob/rcu/rcu-check-unsafe-scheduler-use-2/kernel/rcu/tree_plugin.h#L627)
the reader has blocked an expedited RCU grace period. In particular, it
[notices that preemption is
disabled](https://github.com/joelagnel/linux-kernel/blob/rcu/rcu-check-unsafe-scheduler-use-2/kernel/rcu/tree_plugin.h#L620)
and so the grace period cannot end due to RCU consolidation. Out of
desperation, it raises a softirq (`raise_softirq()`) in the hope that the next
time the softirq runs, the grace period could be ended quickly before the
scheduler tick occurs. But that can cause a scheduler deadlock by way of entry
into the scheduler due to a ksoftirqd-wakeup.

The cure for this problem is the same, holding the RQ/PI locks across the
entire reader section results in no question of a scheduler related deadlock
due to recursively acquiring of these locks; because there would be no question
of expedited-grace-period IPIs, hence no question of setting of any hints, and
hence no question of calling `rcu_read_unlock_special()` from scheduler code.
For a twist of the IPI problem, see [special note](#special-note).

However, the RCU consolidation throws yet another curve ball. Paul McKenney
[explained on
LKML](https://lore.kernel.org/lkml/20190627173831.GW26519@linux.ibm.com/) that
there is yet another situation now due to RCU consolidation that can cause
scheduler deadlocks.

Consider the following code, where `previous_reader()` and `current_reader()`
execute in quick succession in the context of the same task:
```
       previous_reader()
	{
		rcu_read_lock();
		do_something();      // Preemption or IPI happened
		local_irq_disable(); // Cannot be the scheduler
		do_something_else();
		rcu_read_unlock();  // As IRQs are off, defer QS report
                                    //but set deferred_qs bit in 
                                    //rcu_read_unlock_special
		do_some_other_thing();
		local_irq_enable();
	}

        // QS from previous_reader() is still deferred.
	current_reader() 
	{
		local_irq_disable();  // Might be the scheduler.
		do_whatever();
		rcu_read_lock();
		do_whatever_else();
		rcu_read_unlock();    // Must still defer reporting QS
		do_whatever_comes_to_mind();
		local_irq_enable();
	}
```
Here `previous_reader()` had a preemption; even though the `current_reader()`
did not - but the `current_reader()` still needs to call
`rcu_read_unlock_special()` from the scheduler!  This situation would not
happen in the pre-consolidated-RCU world because `previous_reader()`'s
`rcu_read_unlock()` would have taken care of it.

As you can see, just following the scheduler rule of disabling interrupts
across the entire reader section does not help. To detect the above scenario; a
new bitfield  `deferred_qs` has been
[added](https://lore.kernel.org/patchwork/patch/1057344/) to the
`task_struct::rcu_read_unlock_special` union. Now what happens is, at
`rcu_read_unlock()`-time, the `previous reader()` sets this bit, and the
`current_reader()` checks this bit. If set, the call to `raise_softirq()` is
avoided thus eliminating the possibility of a scheduler deadlock.

Hopefully no other scheduler deadlock issue is lurking!

Coming back to the scheduler rule, I have been running overnight rcutorture
tests to detect if this rule is ever violated. Here is the [test
patch](https://github.com/joelagnel/linux-kernel/commits/rcu/rcu-check-unsafe-scheduler-use-2)
checking for the unsafe condition. So far I have not seen this condition occur
which is a good sign.

I may need to check with Paul McKenney about whether proposing this checking
for mainline is worth it. Thankfully, LPC 2019 is right around the corner! ;-)
--------
#### Special Note
[1] The expedited IPI interrupting an RCU reader has a variation. For an
example see below where the IPI was not received, but we still have a problem
because the `->need_qs` bit in the `rcu_read_unlock_special union` got set even
though the expedited grace period started after IRQs were disabled. The start
of the expedited grace period would set the `rnp->expmask` bit for the CPU. In
the unlock path, because the `->need_qs` bit is set, it will call
`rcu_read_unlock_special()` and risk a deadlock by way of a `ksoftirqd` wakeup
because `exp` in that function is true.
```
CPU 0                         CPU 1
preempt_disable();
rcu_read_lock();

// do something real long

// Scheduler-tick sets
// ->need_qs as reader is
// held for too long.

local_irq_disable();
                              // Expedited GP started
// Exp IPI not received
// because IRQs are off.

local_irq_enable();

// Here rcu_read_unlock will
// still call ..._special()
// as ->need_qs got set.
rcu_read_unlock();

preempt_enable();
```
The fix for this issue is the same as described earlier, disabling interrupts
across both `rcu_read_lock()` and `rcu_read_unlock()` in the scheduler path.
