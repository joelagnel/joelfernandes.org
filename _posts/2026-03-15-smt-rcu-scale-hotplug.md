---
layout: post
title: "Sixty Minutes to Flip a Bit: RCU Bottlenecks in Bulk CPU Hotplug"
date: 2026-03-15
categories: [linux-kernel, rcu, performance]
tags: [linux, kernel, rcu, cpu-hotplug, smt, scalability, percpu-rwsem]
published: false
description: "On a 1536-2048 CPU PowerPC system, toggling SMT mode takes up to an hour. Here is why synchronize_rcu() is the culprit, what three different fixes tried to do about it, and why a testing error nearly sent everyone down the wrong path."
---

Flipping `/sys/devices/system/cpu/smt/control` from `on` to `off` on a 2048-CPU PPC64
machine takes roughly 50 minutes. Flipping it back takes another 49. That is not a
typo. An operation that conceptually changes one bit in the scheduler's view of the
world burns most of a work day on some of the largest production Linux systems in
existence.

This is a recent story from the LKML thread titled
["cpuhp: Expedite synchronize_rcu during SMT switch"][original-patch]
(January/February 2026), and it is a good illustration of what happens when RCU's
grace period machinery collides with O(n) CPU bulk operations at real scale.

---

## Background: Why SMT Switching Involves RCU At All

SMT (Simultaneous Multithreading) mode switching on PPC64 takes each core from exposing
8 hardware threads (SMT8) down to 1. On a 256-core system, switching to SMT1 means
offlining 1,792 CPUs, one at a time, in a loop.

Each CPU hotplug call goes through `_cpu_up()` / `_cpu_down()`, which acquires
`cpus_write_lock()`. That lock is a
[`percpu_rw_semaphore`][percpu-rwsem-source], and a write-side acquisition of a
`percpu_rw_semaphore` calls `synchronize_rcu()` under the hood.

The simplified flow in [`kernel/locking/percpu-rwsem.c`][percpu-rwsem-source]:

```c
void percpu_down_write(struct percpu_rw_semaphore *sem)
{
    /* Phase 1: signal that a writer is waiting */
    rcu_sync_enter(&sem->rss);  /* can call synchronize_rcu() */

    /* Phase 2: wait for existing readers to drain */
    wait_event(sem->writer, /* readers == 0 */);
}
```

`rcu_sync_enter()` transitions the semaphore from fast-path (readers use RCU) to
slow-path (readers use an explicit wait queue). That transition requires a full RCU
grace period to ensure all pre-existing readers finish. With 1,792 CPUs to hotplug and
multiple `synchronize_rcu()` calls per CPU in the path, you hit thousands of grace
periods. On a system where each grace period costs tens of milliseconds, the math works
out to tens of minutes.

The measured numbers from IBM's testing (2048 CPUs, PPC64, kernel 6.19.0-rc5):

```
SMT8 -> SMT1 (offline 1792 CPUs):  ~30 minutes baseline
SMT1 -> SMT8 (online 1792 CPUs):   ~49 minutes baseline
```

---

## Three Approaches

### 1. RCU Expediting

[Vishal Chourasia (IBM)][original-patch] proposed expediting RCU for the duration of
the SMT switch:

```c
int cpuhp_smt_disable(enum cpuhp_smt_control ctrl)
{
    cpu_maps_update_begin();
    rcu_expedite_gp();   /* all following synchronize_rcu() use IPI-based fast path */

    for_each_present_cpu(cpu) {
        /* ... cpu_down() calls in here ... */
    }

    rcu_unexpedite_gp();
    cpu_maps_update_done();
    return ret;
}
```

Expedited RCU uses IPIs to force quiescent states immediately instead of waiting for
the scheduler to observe them naturally. It is more expensive per-grace-period in terms
of IPI traffic, but each grace period completes in microseconds instead of
milliseconds.

Results (2048 CPUs):

```
              | Baseline  | Expedited | Improvement
SMT=off       | 30m53s    | 6m4s      |   -80%
SMT=on        | 49m6s     | 36m50s    |   -25%
```

A 5x win for the offline case. The online case improves less because onlining CPUs
involves more serialized work beyond just the RCU overhead.

The objection raised by both Peter Zijlstra and myself: expediting treats the symptom.
The underlying issue is that we call `synchronize_rcu()` thousands of times for a
single admin operation. Fixing the call pattern is cleaner than throwing IPIs at it.

### 2. The rcu_sync Optimization (Peter Zijlstra)

Peter's observation: `rcu_sync_enter()` transitions the semaphore to slow path once per
lock acquisition. If you could enter slow path *once* for the entire batch, you'd pay
the RCU cost once instead of 1,792 times.

His patch called `rcu_sync_enter(&cpu_hotplug_lock.rss)` directly around the bulk loop,
forcing the percpu-rwsem into slow path for the duration. Each individual `down_write()`
in the loop would find the semaphore already in slow path and skip the RCU sync.

Initial test results looked extraordinary:

```
              | Without Patch | With Patch | Improvement
SMT=off       | 20m32s        | 5m31s      |   -73%
SMT=on        | 62m47s        | 55m46s     |   -11%
```

Seventy-three percent improvement from a 10-line patch. That is the kind of number
that ends a debugging session.

Except it was wrong.

### The b4 am Gotcha

A few days later, Samir M (IBM) [posted a correction][b4-correction]:

> "Apologies for the confusion in the previous report. In the previous report, I used
> the `b4 am` command to apply the patch. As a result, **all patches in the mail thread
> were applied**, rather than only the intended one. Consequently, the results that were
> posted earlier included changes from multiple patches."

`b4 am` is a tool for fetching and applying patches from the mailing list. Given a
message-id, it applies the *entire thread*, not just the patch corresponding to that
message. When Samir ran it against Peter's patch message-id, he got Peter's patch *plus*
the RCU expediting patch *plus* my lock hoisting patch all at once. The impressive 73%
improvement was actually expediting doing its job, not the rcu_sync optimization.

Corrected results for Peter's patch alone (1536 CPUs):

```
              | Without Patch | With Patch | Improvement
SMT=off       | 20m32s        | 20m22s     |   -0.8%
SMT=on        | 62m47s        | 63m01s     |   +0.2%
```

Essentially zero. The rcu_sync optimization, in isolation, does nothing. The semaphore
was not transitioning to slow path on every `down_write()`, or the slow path itself is
equally expensive, or the RCU syncs are happening elsewhere in the call chain. No
definitive answer yet.

The lesson for anyone testing mailing list patches: use `b4 am` with the message-id of
the *specific patch* you want, not a reply in the thread, or extract it manually. The
tool is designed to collect the whole series, which is usually what you want, unless you
are trying to isolate a single change.

### 3. Lock Hoisting (My Approach)

My hypothesis: repeated lock acquisition overhead is the culprit. Hold the lock once
across the entire batch instead of acquiring and releasing it per CPU. I introduced
`cpu_up_locked()`, a variant of `_cpu_up()` that asserts `cpus_held()` instead of
acquiring the lock itself, and updated `cpuhp_smt_enable()` to call `cpus_write_lock()`
once around the loop:

```c
int cpuhp_smt_enable(void)
{
    cpu_maps_update_begin();
    cpus_write_lock();   /* held for entire batch */

    for_each_present_cpu(cpu) {
        ret = cpu_up_locked(cpu, 0, CPUHP_ONLINE);   /* no lock acquire inside */
        if (ret)
            break;
        cpuhp_online_cpu_device(cpu);
    }

    cpus_write_unlock();
    arch_smt_update();
    cpu_maps_update_done();
    return ret;
}
```

The tree is at [`git.kernel.org/pub/scm/linux/kernel/git/jfern/linux.git`][joel-tree],
tag `cpuhp-bulk-optimize-rfc-v1`.

Results (400 CPUs, smaller test system):

```
              | Baseline | Patch  | Improvement
SMT=off       | 1m27s    | 1m26s  |   ~0%
SMT=on        | 1m01s    | 1m03s  |   ~0%
```

Also zero. The lock acquisition pattern is not the bottleneck. I argued for this
approach over expediting because it addresses root cause rather than symptom, but the
data disagreed.

---

## What the Data Says

Three interventions, three non-results (excluding expediting, which helps but does not
solve the problem). The conclusion:

**The bottleneck is RCU grace period *duration*, not grace period *frequency*.**

The structural optimizations (rcu_sync, lock hoisting) tried to reduce the number of
`synchronize_rcu()` calls. They failed. Expediting works because it makes each
individual grace period shorter, not because it reduces how many there are.

That matters for future work. The next thing to try is profiling where exactly the
grace period time goes at this scale. Ftrace on `rcu_gp_init` and `rcu_gp_fqs_loop`
on a 2048-CPU system would show whether the overhead is dominated by quiescent state
collection, IPI latency, or something else.

The SMT=on case is still 36+ minutes even with expediting. The offline case responds
better because quiescent states are easier to observe on CPUs that are going offline.
Online adds CPU initialization overhead that is independent of RCU entirely.

The scaling is also non-linear: 400 CPUs takes about 1 minute; 2048 CPUs takes 30-50
minutes. That suggests O(n^2) or worse somewhere in the path, which expediting alone
will not fix at even larger scales.

---

## Open Questions

1. Where exactly are the remaining `synchronize_rcu()` calls in the SMT=on path? The
   smaller improvement from expediting suggests other bottlenecks dominate there.

2. Should `percpu_rw_semaphore` expose a batch mode? Something like
   `percpu_down_write_batch_begin()` that enters slow path once for a series of
   write-side critical sections. As I noted in the thread, other future
   `percpu_rw_semaphore` users would benefit from this API too.

3. Is expediting the right long-term answer? It is defensible for an admin operation
   that the user explicitly invoked. For anything in a hot path, it would be a problem.

---

## Key Takeaways

- On a 2048-CPU PPC64 system, `synchronize_rcu()` called thousands of times during
  bulk CPU hotplug adds up to 20-60 minutes of latency for a single SMT switch
- `percpu_rw_semaphore` is the amplifier: each `down_write()` can trigger a full grace
  period via `rcu_sync_enter()`
- RCU expediting gives a 5x improvement for SMT=off; structural changes (rcu_sync
  batching, lock hoisting) had no measurable effect
- The real bottleneck is grace period *duration*, not how many there are
- When testing mailing list patches with `b4 am`, use the exact patch message-id, not
  a thread reply. The tool applies the entire series.

---

## References

- [cpuhp: Expedite synchronize_rcu during SMT switch (original patch)][original-patch]
- [Correction: testing error with b4 am][b4-correction]
- [Peter Zijlstra's rcu_sync suggestion][peter-suggestion]
- [Lock hoisting RFC (jfern/linux.git)][joel-tree]
- [`percpu_rw_semaphore` implementation][percpu-rwsem-source]
- [`rcu_sync` implementation][rcu-sync-source]
- [CPU hotplug core (`kernel/cpu.c`)][cpu-c-source]

[original-patch]: https://lore.kernel.org/all/20260119104739.439799-2-vishalc@linux.ibm.com/
[b4-correction]: https://lore.kernel.org/all/367a0168-be38-48ad-b55e-688d8eaaca49@linux.ibm.com/
[peter-suggestion]: https://lore.kernel.org/all/20260119114333.GI1890602@noisy.programming.kicks-ass.net/
[joel-tree]: https://git.kernel.org/pub/scm/linux/kernel/git/jfern/linux.git/tag/?h=cpuhp-bulk-optimize-rfc-v1
[percpu-rwsem-source]: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/kernel/locking/percpu-rwsem.c
[rcu-sync-source]: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/kernel/rcu/sync.c
[cpu-c-source]: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/kernel/cpu.c
