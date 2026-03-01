---
layout: post
title: "SRCU-Fast: Dropping smp_mb() With RCU Grace Period Ordering"
date: 2026-03-01
categories: [linux-kernel, rcu]
tags: [linux, kernel, rcu, srcu, memory-ordering, synchronization]
published: false
description: "SRCU-Fast eliminates the smp_mb() calls from the SRCU reader path by replacing them with implicit RCU read-side critical sections. Here is why that works."
---

The SRCU reader path in mainline Linux has, for years, contained two full memory barriers -- `smp_mb()` on lock and `smp_mb()` on unlock. On x86 that is two `mfence` or `lock xadd` instructions per critical section. On anything with a weaker memory model the cost is real.

SRCU-Fast changes this. Reader lock and unlock drop to `barrier()` -- a compiler fence with zero CPU overhead. Same correctness guarantee. No smp_mb(). [[1]](#ref1)

The mechanism that makes this safe is a classic from 2013 that I suspect most people have never had to actually internalize: you can replace one side of a Dekker-style memory barrier pair with an RCU grace period.

## Why SRCU Needs Barriers at All

SRCU (Sleepable RCU) tracks readers using per-CPU counters. Lock increments a "locks" counter; unlock increments an "unlocks" counter. The grace period scans both, waiting for `locks == unlocks` on each CPU. When they match, no reader is active.

The problem is store buffering. Modern CPUs can delay stores. Without barriers, a CPU can write its counter increment into a store buffer, then turn around and read another CPU's counters from cache -- seeing stale values. The grace period check could declare itself complete while a reader's counter increment is still sitting in a buffer, invisible to the scan. That is a broken guarantee.

Standard SRCU fixes this with `smp_mb()` on both sides of the critical section: [[2]](#ref2)

```c
int __srcu_read_lock(struct srcu_struct *ssp)
{
    struct srcu_ctr __percpu *scp = READ_ONCE(ssp->srcu_ctrp);
    this_cpu_inc(scp->srcu_locks.counter);
    smp_mb();  /* B: flush store, order vs. grace period scan */
    return __srcu_ptr_to_ctr(ssp, scp);
}

void __srcu_read_unlock(struct srcu_struct *ssp, int idx)
{
    smp_mb();  /* C: flush critical section stores before unlock */
    this_cpu_inc(__srcu_ctr_to_ptr(ssp, idx)->srcu_unlocks.counter);
}
```

The `B` barrier ensures the lock increment is globally visible before the grace period can see the reader's critical section accesses. The `C` barrier ensures critical section stores drain before the unlock counter increments. Together they form a Dekker pair with the `smp_mb()` in the grace period path.

## The Dekker Trick

Dekker's original mutual exclusion algorithm relies on paired memory barriers to prevent store buffering: both sides write, both sides barrier, both sides read. The barriers guarantee at least one side sees the other's write. Concretely:

```
CPU 0                        CPU 1
-----                        -----
WRITE_ONCE(x, 1)             WRITE_ONCE(y, 1)
smp_mb()                     smp_mb()
r0 = READ_ONCE(y)            r1 = READ_ONCE(x)

Guarantee: !(r0 == 0 && r1 == 0)
```

In 2013, McKenney, Desnoyers, et al. published an LWN article showing that an RCU grace period can replace one side's barrier: [[3]](#ref3)

```
CPU 0 (reader)               CPU 1 (updater)
--------------               ---------------
rcu_read_lock()
  WRITE_ONCE(x, 1)           WRITE_ONCE(y, 1)
  r0 = READ_ONCE(y)          synchronize_rcu()
rcu_read_unlock()            r1 = READ_ONCE(x)

Same guarantee: !(r0 == 0 && r1 == 0)
```

This works because of the fundamental RCU ordering law: if any part of an RCU read-side critical section precedes the start of a grace period, then the entire critical section precedes the end of that grace period -- and the converse holds too.

So either CPU 0's read-side critical section completed before CPU 1's grace period started (CPU 1 sees x=1), or it overlapped and `synchronize_rcu()` waited for it (ordering is enforced). Either way the `r0 == 0 && r1 == 0` outcome is impossible. CPU 0 needs no `smp_mb()` of its own.

## SRCU-Fast Applies This Directly

SRCU-Fast builds on exactly this pattern. The reader path becomes: [[1]](#ref1)

```c
static inline struct srcu_ctr __percpu *
__srcu_read_lock_fast(struct srcu_struct *ssp)
{
    struct srcu_ctr __percpu *scp = READ_ONCE(ssp->srcu_ctrp);
    this_cpu_inc(scp->srcu_locks.counter);  /* Y: implicit RCU reader */
    barrier();  /* compiler fence only */
    return scp;
}

static inline void
__srcu_read_unlock_fast(struct srcu_struct *ssp, struct srcu_ctr __percpu *scp)
{
    barrier();  /* compiler fence only */
    this_cpu_inc(scp->srcu_unlocks.counter);  /* Z: implicit RCU reader */
}
```

The `smp_mb()` is gone. The `barrier()` calls remain to stop the compiler from moving critical section accesses across the counter increments, but they issue no CPU fence instruction.

The CPU ordering comes from a different source: the grace period path now calls `synchronize_rcu()` in addition to its own barriers. And the counter increments themselves -- `this_cpu_inc()` and `atomic_long_inc()` -- are implicit RCU read-side critical sections.

That last point is the non-obvious one. Paul McKenney's explanation: [[4]](#ref4)

> Note that both `this_cpu_inc()` and `atomic_long_inc()` are RCU read-side critical sections either because they disable interrupts, because they are a single instruction, or because they are read-modify-write atomic operations, depending on the whims of the architecture.

Interrupts-disabled regions are RCU readers. Single instructions cannot span a grace period. Atomic read-modify-write operations are uninterruptible. Any of these properties makes the operation an implicit RCU reader -- and which one applies depends on what the architecture does with `this_cpu_inc()`.

The diagram below shows how this maps to the Dekker substitution:

```
SRCU-Fast reader               Grace period path
----------------               -----------------
inc(locks)  [Y] <-- implicit   synchronize_rcu() [X] <-- waits
            RCU reader              for all readers
barrier()
[... critical section ...]
barrier()
inc(unlocks) [Z] <-- implicit
             RCU reader

Y and Z pair with X via RCU grace period ordering.
No smp_mb() needed on the reader side.
```

## The Cost You Pay Elsewhere

This is not free. The grace period path now calls `synchronize_rcu()`, which is heavier than the `smp_mb()`-based scan in standard SRCU. Grace periods take longer -- in the tens of milliseconds range rather than hundreds of microseconds -- because `synchronize_rcu()` has to wait for a full RCU quiescent state cycle across all CPUs.

The tradeoff is explicit: SRCU-Fast is for read-heavy workloads where grace periods are infrequent. If you are doing a lot of updates, standard SRCU wins because its grace periods are cheaper. If your readers vastly outnumber your writers, SRCU-Fast wins because you are paying the slow path rarely and the fast path constantly.

There is also an architecture-specific wrinkle. On some ARM implementations, per-CPU atomic operations have higher latency than on x86, which narrows the reader-side savings. The tradeoff needs to be measured per-platform.

## Key Takeaways

- Standard SRCU puts `smp_mb()` in the reader path to implement Dekker-style ordering against the grace period scan.
- SRCU-Fast replaces those CPU barriers with the RCU grace period ordering guarantee, which is equivalent but needs `synchronize_rcu()` in the grace period path instead.
- The counter increments (`this_cpu_inc()`, `atomic_long_inc()`) serve as implicit RCU read-side critical sections, which is what makes the substitution valid.
- The result is a read path with zero CPU fence instructions -- correct and cheap, at the cost of slower grace periods.
- Use SRCU-Fast when reads dominate. Use standard SRCU when grace period latency matters.

## References

<a name="ref1"></a>[1] SRCU-fast reader implementation: `include/linux/srcutree.h`, Linux kernel source

<a name="ref2"></a>[2] Standard SRCU reader implementation: `kernel/rcu/srcutree.c`, Linux kernel source

<a name="ref3"></a>[3] McKenney, Desnoyers, Jiangshan, Triplett. [The RCU-barrier menagerie](https://lwn.net/Articles/573497/). LWN.net, November 2013.

<a name="ref4"></a>[4] Paul McKenney. [Response on implicit RCU readers in SRCU-fast context](https://lore.kernel.org/all/2f8bb8bb-320e-480f-9a56-8eb5cbd4438a@paulmck-laptop/). LKML, 2025.

<a name="ref5"></a>[5] Mathieu Desnoyers. [Technical review of SRCU-fast memory ordering](https://lore.kernel.org/all/2d9eb910-f880-4966-ba40-9b1e0835279c@efficios.com/). LKML, July 2025.

<a name="ref6"></a>[6] Paul McKenney. [SRCU-fast documentation patch](https://lore.kernel.org/all/20250918102646.2592821-4-paulmck@kernel.org/). LKML, September 2025.
