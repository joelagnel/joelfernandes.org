---
layout: post
published: false
title: "RCU and dynticks-idle mode"
comments: true
category: linux
---
The kernel's `dynticks-idle` mode is a mode of a CPU in which the CPU is idle
and the scheduler clock tick has been turned off to save power and let the CPU
go into a lower power state. Also known as NO_HZ.

A CPU in this mode presents some challenges to RCU. This is because an RCU
grace period completion depends on the CPU reporting that it is in a quiescent
state even when its idle. When the CPU is idle but the scheduling clock tick is
not turned off, RCU on that idle-but-ticking-CPU can simply report from the
tick path that the CPU is in a quiescent state. However in dynticks-idle mode
this isn't possible, so something more clever is needed.

For RCU's purposes, the kernel maintains a per-cpu datastructure called
`rcu_dynticks` this dynticks-idle tracking.

Extended Quiescent State (EQS)
------------------------
An extended quiescent state is defined as a processor state in which RCU
considers the CPU as not something that is using RCU. This is also important
for a more aggressive form of dynticks-idle code (CONFIG_NO_HZ_FULL) which not
only turns off the tick in the idle path but also in userspace if there is no
other need for the tick other than RCU (for example if only 1 task is running).

By defining certain contexts as an EQS, RCU will work no matter how aggressive
the dynticks-idle implementation.

AFAICT, there are 2 EQS states: idle-loop and usermode. In both these states,
the tick may be turned off and the CPU is considered to be in a quiescent state
as far as RCU is concerned.

Entry and exit into an EQS due to transition to/from non-idle kernel process context
--------------------------------------------------------------------------
The `rdtp->dynticks_nesting` counter tracks entry and exit into an EQS due to
transition from idle to process context or from usermode to process context.  A
value of 0 indicates that we're in an EQS and a value of > 0 indicates we're
not. i.e. a non-zero value means we transitioned into process context.

An EQS can be exited due to interrupt or NMI entry, but this doesn't really
track that. We'll talk about tracking those later.

Another way to see if we are in an EQS or not
---------------------------------------------
The `rdtp->dynticks` counter is used to track transitions to/from dyntick-idle
mode. But it also can share light on whether we are in an EQS or not. If this
counter is odd, it means we are NOT in an EQS and if its even, then we ARE.
Note: since an EQS entry can happen even because of transition into usermode,
this counter is not only incremented due to entry into dyntick-idle mode, but
also due to transition into usermode. This is observed by seeing that an
increment of this counter can also happen due to
`rcu_user_enter`->`rcu_eqs_enter`->`rcu_dynticks_eqs_enter`.

The following function checks this:
```
/*
 * Is the current CPU in an extended quiescent state?
 *
 * No ordering, as we are sampling CPU-local information.
 */
bool rcu_dynticks_curr_cpu_in_eqs(void)
{
        struct rcu_dynticks *rdtp = this_cpu_ptr(&rcu_dynticks);

        return !(atomic_read(&rdtp->dynticks) & RCU_DYNTICK_CTRL_CTR);
}

```
Any time the rdtp->dynticks counter's lower most bit is not set, we are in an
EQS, and if its set, then we are not.

Entry and exit into an EQS due to interrupts
------------------------------------------
Other than the  entry/exit into usermode or idle, interrupts and NMIs can cause
the CPU to enter/exit a QS. Naturally, RCU needs to be "watching" as RCU
read-side critical sections are permitted in interrupt handlers so an exit from
an EQS for this purpose is a must. This is done by calls to
`rcu_eqs_enter/exit` from `rcu_irq_exit/enter` respectively.

Further, the interrupt nesting is also carefully tracked in
`rdtp->dynticks_nmi_nesting` as of v4.18-rc1, and we'll see later why this is
needed and complications due to nested NMIs (yes NMIs can nest!) that need to
be handled. Both IRQ-nesting and NMI-nesting use the same
`dynticks_nmi_nesting` counter.

With this knowledge in mind, lets discuss how a QS is reported from the tick
path when the tick is infact not turned off.

How are QS reported from the timer tick
---------------------------------
As for 4.18-rc1, the tick call graph which checks for QS is as follows:
```
tick_sched_timer->
    tick_sched_handle->
	update_process_times
		rcu_check_callbacks
```
There are 3 variants of RCU (sched, bh and preempt). All these variants have
different ways of detecting a QS. There's also tasks RCU but we'll skip that.

For the sched RCU variant, we are in a QS if the CPU is either idle, or in
usermode. This awfully sounds like the definition of an EQS. However, we can't
use dynticks eqs detection (`rcu_dynticks_curr_cpu_in_eqs` mentioned earlier in
the article) because `rdtp->dynticks` is just a simple counter. Its even when
we're in an EQS and odd when we're not. It tells us nothing about interrupt
nesting.

Note that the timer tick path is itself triggered through an interrupt, so we
can't rely on the `rcu_dynticks_curr_cpu_in_eqs` detection to tell us if we're
in a QS or not. Instead we rely on other methods. First of all
`rcu_check_callbacks` is passed a user argument booleant, which tells us if the
callback checking (tick) happened during usermode execution. So if that's the
case, its easy, we simply report the CPU to in a QS for rcu-sched. But what are
the other ways we could be in a QS? Just one more: If we were in the idle-loop
at the time of the `rcu_check_callbacks` getting called, AND  we're a 1st level
interrupt that caused a call to rcu_check_callbacks. This first level is infact
the timer tick interrupt. The "first level nesting check" is important, because
only the outer most interrupt that interrupted the idle loop should report the
sched-QS. Any nested interrupts in the idle loop that cause
`rcu_check_callbacks` to be called (I don't know of any) should not report the
QS again.

Old notes:
rcu_is_cpu_rrupt_from_idle

The trick is to determine if we got interrupted (timer tick) while we were
idle. This is determined by a call to `rcu_is_cpu_rrupt_from_idle` in the timer
tick path.

If the CPU is idle while it was interrupted, then both bh and sched RCU
variants are marked to be in a QS on that CPU.

Note: In the `rcu_check_callbacks` function called from the tick path, for the
preempt RCU variant, we don't do this kind of an idle check, instead we simply
check if the read-lock nesting is 0 and if it is, then its a QS. This is
because the read-lock nesting counter is how RCU-preempt enters a read section.
So whether we're in the idle loop or not doesn't matter, all we've to do is
check the nesting counter. For the bh and sched variants though, we disable
preempt or bh when entering the read-side so for those, we need to rely on
other methods (see optimization idea below).

Idea: Can we report a QS for -sched variant faster?  What if a tick is received
during a long running kernel section that didn't disable preemption. Can we
detect from the tick path in rcu_check_callbacks and see how deep was the
preemption disabled nesting and use that info to report a QS for RCU-sched ?

