---
layout: post
published: false
title: "RCU and dynticks-idle mode"
comments: true
category: linux
---
The kernel's `dynticks-idle` mode is a mode of a CPU in which the CPU is idle
and the scheduler clock tick has been turned off to save power and let the CPU
go into a lower power state.

A CPU in this mode presents some challenges to RCU. This is because an RCU
grace period completion depends on the CPU reporting that it is in a quiescent
state even when its idle. When the CPU is idle but the scheduling clock tick is
not turned off, RCU on that idle-but-ticking-CPU can simply report from the
tick path that the CPU is in a quiescent state. However in dynticks-idle mode
this isn't possible, so something more clever is needed.

The kernel maintains a per-cpu datastructure called `rcu_dynticks` which does
most of this dynticks-idle tracking.

For non-dynticks idle-mode: How does RCU know if the CPU is idle from the tick?
---------------------------------
The trick is to determine if we got interrupted (timer tick) while we were
idle. This is determined by a call to `rcu_is_cpu_rrupt_from_idle` in the timer
tick path.
```
tick_sched_timer->
tick_sched_handle
	update_process_times
		rcu_check_callbacks
			rcu_is_cpu_rrupt_from_idle
```

If the CPU is idle while it was interrupted, then both bh and sched RCU
variants are marked to be in a QS on that CPU.

Note: In the `rcu_check_callbacks` function, for the preempt RCU variant, we
don't do this kind of an idle check, we simply check if the read-lock nesting
is 0 and if it is, then its a QS. This is because the read-lock nesting counter
is how RCU-preempt enters a read section.  So whether we're in the idle loop or
not doesn't matter, all we've to do is check the nesting counter. For the bh
and sched variants though, we disable preempt or bh when entering the read-side
so for those, we need to rely on other methods (see optimization idea below).

Idea: Can we report a QS for -sched variant faster?  What if a tick is received
during a long running kernel section that didn't disable preemption. Can we
detect from the tick path how deep was the preemption disabled nesting and use
that info to report a QS for RCU-sched ?

