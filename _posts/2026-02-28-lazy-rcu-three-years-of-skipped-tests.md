---
layout: post
title: "Lazy RCU: Three Years of Skipped Tests"
date: 2026-02-28
categories: [linux-kernel, rcu]
tags: [linux, kernel, rcu, rcutorture, lazy-rcu, testing]
published: false
description: "Since late 2022, rcutorture has been using call_rcu_hurry() instead of call_rcu(), accidentally bypassing every lazy RCU code path. Here is how that happened and how we are closing the gap."
---

`CONFIG_RCU_LAZY` has been in the kernel for years. rcutorture, the stress test for RCU, has been running all along. But for most of that time, rcutorture has not been testing lazy RCU's core code paths at all. It has been quietly bypassing them with a one-line workaround.

Here is the story.

## What Lazy RCU Does

The standard `call_rcu()` schedules a callback to run after the current RCU grace period — once all pre-existing readers have finished. Under normal operation this triggers the grace period machinery quickly.

Lazy RCU (`CONFIG_RCU_LAZY`) changes that behavior. When lazy is enabled, `call_rcu()` parks the callback on a per-CPU bypass queue instead of going straight to the main callback list. It stays there, accumulating with other deferred work, until either a timer fires (on the order of seconds) or memory pressure triggers a shrinker flush. On idle or lightly-loaded systems this avoids unnecessary wakeups — the kernel does not spin up grace period machinery for callbacks that can wait. [[1]](#ref1)

For callers that cannot tolerate that deferral, `call_rcu_hurry()` exists. It skips the bypass queue and triggers a grace period right away. When `CONFIG_RCU_LAZY` is not set, `call_rcu_hurry()` is just an alias for `call_rcu()` — same behavior, no overhead. [[2]](#ref2)

## The Stutter Timing Problem

rcutorture stress-tests RCU by cycling through a "stutter" loop: run writer and reader kthreads hard for a few seconds, pause, check for problems, repeat. [[3]](#ref3) The pause matters — it is when rcutorture looks for leaked objects, things that should have completed a grace period but have not.

The stutter pause is typically a few seconds. Lazy RCU can defer callback processing for much longer. Put those together and you get a false positive: the writer calls `call_rcu()`, the callback lands on the bypass queue, the stutter pause fires before the lazy timer does, and the leak check sees the object still in flight. rcutorture fires a `rtort_pipe_count` warning. RCU is not broken — the callback is just waiting in the queue.

This conflict showed up when lazy RCU was being merged in late 2022. The fix was pragmatic: replace `call_rcu()` with `call_rcu_hurry()` throughout rcutorture. No bypass queue, no timing conflict, no false positives. [[4]](#ref4)

Also no lazy RCU testing.

## What Got Bypassed

With `call_rcu_hurry()` in place, these code paths were never touched by a torture run:

```
call_rcu()              call_rcu_hurry()
    |                         |
    v                         v
bypass queue             main cblist
lazy_len tracking        ---skipped---
shrinker hooks           ---skipped---
lazy flush timer         ---skipped---
bypass-to-cblist flush   ---skipped---
```

In March 2023, `CONFIG_RCU_LAZY=y` was added to the TREE04 torture config. [[5]](#ref5) On paper that looked like lazy RCU was getting tested. In practice the workaround was still there: rcutorture called `call_rcu_hurry()` everywhere, so the bypass queue was never populated and none of the lazy code paths ran.

Three-plus years after lazy RCU landed, its bypass queue, `lazy_len` counter, timer logic, and shrinker integration had still never been exercised during a torture run.

## The Fix: Two Parts

### NOCB01 and NOCB02

I sent a series in February 2026 adding two new torture configurations. [[6]](#ref6)

NOCB01 enables `CONFIG_RCU_LAZY` and `CONFIG_RCU_NOCB_CPU` together and boots with `rcu_nocbs=all`. This targets the path where lazy callbacks are managed by NOCB kthreads rather than being handled inline on the CPU's own list.

NOCB02 exercises NOCB poll mode, where the kthreads actively poll for completed grace periods instead of sleeping until woken.

Paul McKenney gave both configs his `Reviewed-by` and `Tested-by`.

### Paul's RFC: Revert to call_rcu()

Two days later Paul sent an RFC that goes straight at the root problem: revert the `call_rcu_hurry()` instances back to `call_rcu()`. [[7]](#ref7)

The stutter timing conflict still exists. His solution: before each stutter pause, queue a workqueue handler that calls `rcu_barrier()`. That function flushes all queued lazy callbacks from the bypass queue to the main callback list, starts their grace periods, and waits for them to complete. By the time the pause fires and the leak check runs, everything is done.

The relevant diff:

```diff
-	call_rcu_hurry(&p->rtort_rcu, rcu_torture_cb);
+	call_rcu(&p->rtort_rcu, rcu_torture_cb);

-	.call = call_rcu_hurry,
+	.call = call_rcu,

-	synchronize_rcu_mult(call_rcu_tasks, call_rcu_hurry);
+	synchronize_rcu_mult(call_rcu_tasks, call_rcu);

+static void rcu_torture_writer_work(struct work_struct *work)
+{
+	if (cur_ops->cb_barrier)
+		cur_ops->cb_barrier();
+}

+	INIT_WORK_ONSTACK(&lazy_work, rcu_torture_writer_work);
+	if (IS_ENABLED(CONFIG_RCU_LAZY))
+		queue_work(system_percpu_wq, &lazy_work);
 	stutter_waited = stutter_wait("rcu_torture_writer");
```

The timing now works:

```
t=0s       t=3s            t=6s
[  Run   ] [  Pause       ][  Run   ]
    ^       ^          ^
 call_    queue_work  leak check
 rcu()    rcu_barrier (all callbacks done)
          flush+wait
```

The lazy queue gets populated, the flush runs before the check, and there are no false positives.

The RFC left a few open questions for reviewers — whether the work should be skipped for non-lazy RCU flavors, whether to check the `rcutree.enable_rcu_lazy` boot parameter, and whether an `ops->call_hurry()` field is needed for mixed testing. These are details to resolve before the patch lands.

## Key Takeaways

- Lazy RCU defers `call_rcu()` callbacks on a per-CPU bypass queue to reduce wakeups on idle systems. `call_rcu_hurry()` bypasses this deferral for latency-sensitive paths.
- A timing conflict between lazy deferral and rcutorture's stutter pause caused false-positive failures at merge time. The workaround was to use `call_rcu_hurry()` everywhere in rcutorture.
- The consequence: the bypass queue, `lazy_len` tracking, the flush timer, and shrinker integration were never exercised by the torture test for over three years.
- Two new configs (NOCB01, NOCB02) add dedicated lazy and NOCB poll mode coverage. Paul's RFC reverts to `call_rcu()` and uses `rcu_barrier()` before stutter pauses to flush the bypass queue safely.

A workaround that prevents a false positive is sometimes also a workaround that prevents a real test from running. Worth keeping track of which is which.

## References

1. <a name="ref1"></a> [Linux kernel source: `kernel/rcu/tree_nocb.h` — NOCB and lazy callback handling](https://github.com/torvalds/linux/blob/master/kernel/rcu/tree_nocb.h)
2. <a name="ref2"></a> [Linux kernel source: `include/linux/rcupdate.h` — `call_rcu_hurry()` declaration under `CONFIG_RCU_LAZY`](https://github.com/torvalds/linux/blob/master/include/linux/rcupdate.h)
3. <a name="ref3"></a> [Linux kernel source: `kernel/rcu/rcutorture.c` — RCU torture test module](https://github.com/torvalds/linux/blob/master/kernel/rcu/rcutorture.c)
4. <a name="ref4"></a> [Patch: `call_rcu_hurry()` introduced in rcutorture (November 2022)](https://lore.kernel.org/all/20221130181325.1012760-10-paulmck@kernel.org/)
5. <a name="ref5"></a> [Patch: TREE04 gains `CONFIG_RCU_LAZY=y` (March 2023)](https://lore.kernel.org/all/20230323043935.1221184-4-boqun.feng@gmail.com/)
6. <a name="ref6"></a> [Joel Fernandes: NOCB01 and NOCB02 torture configs (February 2026)](https://lore.kernel.org/all/20260224230435.3390963-1-joelagnelf@nvidia.com/)
7. <a name="ref7"></a> [Paul E. McKenney: RFC "rcutorture: Fully test lazy RCU" (February 27, 2026)](https://lore.kernel.org/all/07395baa-806d-4e12-84b2-c393aee064cd@paulmck-laptop/)
