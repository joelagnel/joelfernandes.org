---
layout: post
published: true
title: "RCU and dynticks-idle mode"
comments: true
category: linuxinternals
---
Note 1: RCU is an extremely complex topic and I make no claims of accuracy,
correctness and don't make any claims that this document is to be used as a
defacto reference for any purpose. You have been warned! For more accurate and
standard references, I will refer you to the kernel RCU documentation.  Please
consider this post as rough notes. That said, your corrections and comments are
welcomed.

Note 2: The article is a WIP and not fully finished (thought it is almost).

The kernel's `dynticks-idle` mode is a mode of a CPU in which the CPU is idle
and the scheduler clock tick has been turned off to save power and let the CPU
go into a lower power state. Also known as NO_HZ.

A CPU in this mode presents some challenges to RCU. This is because an RCU
grace period completion depends on RCU knowing that a CPU has transitioned
through a quiescent state. When the CPU is idle but the scheduling clock tick
is not turned off, RCU on that idle-but-ticking-CPU can simply report from the
tick path that the CPU is in a quiescent state. However in dynticks-idle mode
this isn't possible, so something more clever is needed. The same complications
arise due to the turning off of the tick in user mode (adaptive-ticks support).
This article goes through the design of RCU from this perspective.

For RCU's purposes, the kernel maintains a per-cpu datastructure called
`rcu_dynticks` which does this dynticks-idle state tracking.

Extended Quiescent State (EQS)
------------------------
An extended quiescent state is defined as a processor state in which RCU
considers the CPU as not something that is using RCU. This is also important
for a more aggressive form of dynticks-idle code (CONFIG_NO_HZ_FULL) which not
only turns off the tick in the idle path but also in userspace if there is no
other need for the tick other than RCU (for example if only 1 task is running).

By defining certain contexts as an EQS, RCU will work no matter how aggressive
the dynticks-idle implementation.

AFAICT, there are 2 EQS states: dynticks-idle and usermode. In both these states,
the tick may be turned off and the CPU is considered to be in a quiescent state
and RCU is considered "idle".

Entry and exit into an EQS due to transition to/from non-idle kernel process context
--------------------------------------------------------------------------
The `rdtp->dynticks_nesting` counter tracks entry and exit into an EQS due to
transition from idle to process context or from usermode to process context.  A
value of 0 indicates that the CPU in an EQS and a value of > 0 indicates that
it is not. A non-zero value also means we transitioned into the kernel's
non-idle process context.

An EQS can also be exited due to interrupt or NMI entry, but this doesn't
really track that. We'll talk about tracking those later.

A note about dynticks counters: In general the dynticks counters track the
number of reasons why we're not in an EQS (that is RCU is not "idle"). For
example, a value of zero thus means we ARE in an EQS. The
`rdtp->dynticks_nesting` counter tracks the number of process-level (non-idle
kernel process context)-level reasons why RCU is non-idle.

When I traced `rdtp->dynticks_nesting`, I could only find its value to be
either a 0 or a 1. However looking back at [old kernel
sources](https://elixir.bootlin.com/linux/v3.19.8/source/kernel/rcu/rcu.h#L33),
it appears that these can be nested becaues of so called "half-interrupts". I
believe these are basically interrupts that cause a transition to usermode due
to usermode upcalls (usermode helper subsystem).
So a nesting situation could be something like: 1. Transition from idle to
process context which makes dynticks_nesting == 1. Next, an interrupt comes in
which makes a usermode upcall. This usermode call now makes a system call
causing entry back into process context, which increments the dynticks_nesting
counter to 2. Such a crazy situation is perhaps possible.

Another way some paths see if we are in an EQS or not
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
Any time the rdtp->dynticks counter's second-lowest most bit is not set, we are
in an EQS, and if its set, then we are not (second lowest because lowest is
reserved for something else as of v4.18-rc1). This function is not useful to
check if we're in an EQS from a timer tick though, because its possible the
timer tick interrupt entry caused an EQS exit which updated the counter. IOW,
the 'dynticks' counter is not capable of checking if we had already exited the
EQS before. To check if we were in an EQS or not from the timer tick, we
instead must use `dynticks_nesting` counter. More on that later. The above
function is probably just useful to make sure that interrupt entry/exit is
properly updating the dynticks counter, and also to make sure from
non-interrupt context that RCU is in an EQS (see `rcu_gp_fqs` function).

Entry and exit into an EQS due to interrupts
------------------------------------------
Other than the  entry/exit into usermode or idle, interrupts and NMIs can cause
the CPU to enter/exit a QS. Naturally, RCU needs to be "watching" as RCU
read-side critical sections are permitted in interrupt handlers so an exit from
an EQS for this purpose is a must. This is done by calls to
`rcu_eqs_enter/exit` from `rcu_irq_exit/enter` respectively.

The interrupt nesting level is also carefully tracked in
`rdtp->dynticks_nmi_nesting` as of v4.18-rc1, and we'll see later why this is
needed (reporting of a QS from the timer tick) and complications due to nested
NMIs (yes NMIs can nest!) that need to be handled. Both IRQ-nesting and
NMI-nesting use the same `dynticks_nmi_nesting` counter. More on this in the
"Nested Interrupt Handling" section.

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
different ways of detecting a QS. Lets only talk about the checks for the
reporting of the sched RCU variant which is sufficient for the purposes of this
article.

For the sched RCU variant, we are in a QS if the CPU is either idle, or in
usermode. This awfully sounds like the definition of an EQS. However, we can't
use dynticks eqs detection (`rcu_dynticks_curr_cpu_in_eqs` mentioned earlier in
the article) because `rdtp->dynticks` is just a simple counter. Its has
evenness when we're in an EQS and oddity when we're not. It tells us nothing
about interrupt nesting. More on this in the below note.

Note: The timer tick path is itself triggered through an interrupt, so we
can't rely on the `rcu_dynticks_curr_cpu_in_eqs` detection to tell us if we're
in a QS or not. Instead we rely on other methods. First of all
`rcu_check_callbacks` is passed a user boolean parameter, which tells us if the
callback checking (tick) happened during usermode execution. So if that's the
case, its easy, we simply report the CPU to in a QS for rcu-sched. But what are
the other ways we could be in a QS? Just one more: If we were in the idle-loop
at the time of the `rcu_check_callbacks` getting called, AND  we're a 1st level
interrupt that caused a call to rcu_check_callbacks. This first level is infact
most likely the timer tick interrupt. The "first level nesting check" is
important, because only the outer most interrupt that interrupted the idle loop
should report the sched-QS. Any nested interrupts in the idle loop that cause
`rcu_check_callbacks` to be called (I don't know of any) should not report the
QS again. This interrupt nesting level is determined by `dynticks_nmi_nesting`
mentioned in earlier sections!

Turns out that these above checks (user or interrupt-from-idle) are also
worthwhile causes to report a bh and tasks RCU qs so we report them as such.

Nested Interrupt and NMI Handling
-------------------------
During handling of nested interrupts, the `rcu->dynticks` counter which counts
CPU transitions through dynticks-idle or user mode should correctly maintain
the invariant: If its even, we're in an EQS and if its odd, we're not.

A (naive) algorithm may do something like:
```
void rcu_nmi_enter(void)
{
	if(dynticks_is_even())
		dynticks++;

	dynticks_nmi_nesting++;
}

void rcu_nmi_exit(void)
{
	if (dynticks_nmi_nesting != 1) {
		dynticks_nmi_nesting--;
		return;
	}

	dynticks_nmi_nesting = 0;
	dynticks++;
}
```

The problem with this algorithm is if you have an NMI come in while
rcu_nmi_enter is running, bad things can happen.

Lets take the case where an NMI comes in before dynticks is incremented in the
outer rcu_nmi_enter. In this case nothing bad will happen. But say the NMI
comes in after dynticks is incremented in the outer `rcu_nmi_enter` but before
`dynticks_nmi_nesting` is incremented. The what will happen is:

The steps (lets call this sequence BAD-STEPS):
1. The outer rcu_nmi_enter will update dynticks to be odd.
2. An NMI comes in after dynticks is made odd by dynticks++, but before dynticks_nmi_nesting is updated.
3. The second `rcu_nmi_enter` comes in and it will leave `dynticks` alone but increase `dynticks_nmi_nesting` to 1.
4. Now on the corresponding inner `rcu_nmi_exit`, it will notice
  `dynticks_nmi_nesting` is 1 so it will set it to 0.
5. Next it will wrongly increment `dynticks` messing it up completely.

The problem here is the inner `rcu_nmi_exit` increments the dynticks counter
(thus marking the dynticks-idle mode as exited even though we're still in the
inner nested interrupt!) but there's no way of knowing not to do that because
the outer `rcu_nmi_enter` hasn't incremented `dynticks_nmi_nesting` yet!

The desired behavior is, because the outer `rcu_nmi_enter` exited dynticks-idle
mode (incremented dynticks to odd), only the outer `rcu_nmi_exit` should make
it even (and mark an entry back into dynticks-idle mode).

The fix is an algorithm like the following [proposed by Andy Luto](http://lkml.kernel.org/r/CALCETrXSY9JpW3uE6H8WYk81sg56qasA2aqmjMPsq5dOtzso=g@mail.gmail.com) and [formally written and verified by Paul](https://lkml.kernel.org/r/20141122234157.GB5050@linux.vnet.ibm.com):
```
void rcu_nmi_enter(void)
{
	int incby = 2;

	if(dynticks_is_even()) {
		incby = 1;
		dynticks++;
	}

	dynticks_nmi_nesting += incby;
}

void rcu_nmi_exit(void)
{
	if (dynticks_nmi_nesting != 1) {
		dynticks_nmi_nesting -= 2;
		return;
	}

	dynticks_nmi_nesting = 0;
	dynticks++;
}
```

The GOOD steps would now be:
1. The outer rcu_nmi_enter will update dynticks to be odd and set local
   variable incby to 1.
2. An NMI comes in after dynticks is made odd by dynticks++, but before
   dynticks_nmi_nesting is increased by incby.
3. The second `rcu_nmi_enter` comes in and it will leave `dynticks` alone but
   increase `dynticks_nmi_nesting` to 2 (incby is 2 if dynticks was left
alone).
4. Now on the corresponding inner `rcu_nmi_exit`, it will notice
   `dynticks_nmi_nesting` is not 1, so it will set it decrease nmi_nesting to 0
and return WITHOUT messing up the `dynticks` counter.
5. The outer `rcu_nmi_enter` now finally does increase `dynticks_nmi_nesting`
   by 1.
6. The outer `rcu_nmi_exit` will now set `dynticks_nmi_nesting` to 0 and do the
   `dynticks++` causing an entry back into dynticks-idle mode.

Handling of usermode upcalls from interrupts
----------------------------
RCU's design tries to handle conditions where a usermode upcall was made from
IRQ context, with the IRQ entry never being matched with an IRQ exit! These are
so called "half interrupts". Due to this, the rcu_nmi_nesting counter can go
out sync because an rcu_irq_enter will not be paired properly with an
rcu_irq_exit.

This is the reason for a separate `dynticks_nmi_nesting` counter and a
`dynticks_nesting` counter. Special "fixing up" of the dynticks_nmi_nesting is
done to make sure this counter is sane. See next paragraphs on the fixup info.

When dynticks_nesting is decremented to 0 (the outermost process-context
nesting level exit causes an eqs-entry), the dynticks_nmi_nesting is reset to
0. This makes sense because we're no longer in an NMI at this point.

Similarly, when the dynticks_nesting is set to 1, we have entered a
process-context and dynticks_nmi_nesting is set to a high value. This is also
Ok because the dynticks_nmi_nesting serves no purpose (RCU has already exited
the EQS state).

Conclusion
----------
RCU has to watch over what's happening in the system carefully. This makes the
subsystem complex and requires it to handle various weird usages such as
half-interrupts and nested NMIs. The need to save power via dynticks-idle and
adaptive-ticks modes further complicates RCU. Hopefully this article sheds some
light on the foundation blocks of this dynticks RCU tracking which is the basis
of things happening in other areas such as forcing of quiescent states (fqs).
