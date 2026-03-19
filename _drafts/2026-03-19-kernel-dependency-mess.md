---
layout: post
title: "Kernel Dependency Mess: BPF and SRCU"
date: 2026-03-19
---

# Kernel Dependency Mess: BPF and SRCU

The Linux kernel evolution continues to surprise us with unexpected interactions between subsystems designed years apart. This week, I've been tracking a nasty bug involving BPF local storage, SRCU (Sleepable Read-Copy Update), and PREEMPT_RT that perfectly illustrates how modern kernel features can collide in spectacular ways.

The issue surfaced when Andrea Righi from NVIDIA hit problems with sched_ext (the extensible scheduler framework) calling BPF task storage deletion from atomic context. What started as a simple "invalid wait context" bug morphed into a circular deadlock when the initial fix exposed a fundamental lock ordering problem.

Let me walk through both bugs in detail, because they represent two distinct facets of the same underlying problem: legacy subsystems meeting modern atomic context requirements.

## Bug 1: Invalid Wait Context (7.0.0-rc1)

The first bug appeared when Andrea was testing the cosmos scheduler (a sched_ext implementation). The problem manifests when `bpf_task_storage_delete()` gets called from `sched_ext's ops.exit_task()` callback while holding the runqueue lock:

```
=============================
[ BUG: Invalid wait context ]
7.0.0-rc1-virtme #1 Not tainted
-----------------------------
(udev-worker)/115 is trying to lock:
ffffffffa6970dd0 (rcu_tasks_trace_srcu_struct_srcu_usage.lock){....}-{3:3}, at: spin_lock_irqsave_ssp_contention+0x54/0x90
other info that might help us debug this:
context-{5:5}
3 locks held by (udev-worker)/115:
 #0: ffff8e16c634ce58 (&p->pi_lock){-.-.}-{2:2}, at: _task_rq_lock+0x2c/0x100
 #1: ffff8e16fbdbdae0 (&rq->__lock){-.-.}-{2:2}, at: raw_spin_rq_lock_nested+0x24/0xb0
 #2: ffffffffa6971b60 (rcu_read_lock){....}-{1:3}, at: __bpf_prog_enter+0x64/0x110
...
Sched_ext: cosmos_1.0.7_g780e898fc_dirty_x86_64_unknown_linux_gnu (enabled+all), task: runnable_at=-2ms
Call Trace:
 dump_stack_lvl+0x6f/0xb0
 __lock_acquire+0xf86/0x1de0
 lock_acquire+0xcf/0x310
 _raw_spin_lock_irqsave+0x39/0x60
 spin_lock_irqsave_ssp_contention+0x54/0x90
 srcu_gp_start_if_needed+0x2a7/0x490
 bpf_selem_unlink+0x24b/0x590
 bpf_task_storage_delete+0x3a/0x90
 bpf_prog_3b623b4be76cfb86_scx_pmu_task_fini+0x26/0x2a
 bpf_prog_4b1530d9d9852432_cosmos_exit_task+0x1d/0x1f
 bpf__sched_ext_ops_exit_task+0x4b/0xa7
 __scx_disable_and_exit_task+0x10a/0x200
 scx_disable_and_exit_task+0xe/0x60
```

### The Root Cause

The call path shows the problem clearly:

1. `sched_ext's ops.exit_task()` runs with `&rq->__lock` (runqueue lock) held - this is a **raw spinlock**
2. `bpf_task_storage_delete()` calls `bpf_selem_unlink()` 
3. Eventually reaches `call_rcu_tasks_trace()` which internally uses SRCU
4. SRCU's `spin_lock_irqsave_ssp_contention()` tries to acquire `rcu_tasks_trace_srcu_struct_srcu_usage.lock`
5. **Problem**: This SRCU lock is a regular spinlock, not a raw spinlock

Under PREEMPT_RT, regular spinlocks can sleep (they become mutex-like), but we're in atomic context holding a raw spinlock. Lockdep catches this as an "invalid wait context" - trying to acquire a {3:3} lock (sleeping) from context-{5:5} (atomic with raw locks held).

### The Background: RCU Tasks Trace + SRCU

The issue stems from a recent change where RCU Tasks Trace was converted to use SRCU internally. Previously, `call_rcu_tasks_trace()` was safe from raw spinlock contexts, but the SRCU conversion introduced this sleeping lock requirement.

Kumar Kartikeya Dwivedi explained: *"It was always safe to call_rcu_tasks_trace() under raw spin lock, but became problematic on RT with the recent conversion that uses SRCU underneath."*

## Bug 2: Circular Lock Dependency (7.0.0-rc4)

**Important:** This is a completely separate class of problem from Bug 1. Bug 1 was about sleeping in atomic context. Bug 2 is a **lock ordering violation** that can cause a real deadlock, independent of PREEMPT_RT or atomic context constraints. It would exist on any kernel with lockdep enabled.

Paul McKenney initially attempted to fix Bug 1 by converting SRCU's lock to a raw spinlock. This fixed the "invalid wait context" but immediately exposed a circular lock dependency. Andrea's testing revealed:

```
======================================================
WARNING: possible circular locking dependency detected
7.0.0-rc4-virtme #15 Not tainted
------------------------------------------------------
schbench/532 is trying to acquire lock:
ffffffff9cd70d90 (rcu_tasks_trace_srcu_struct_srcu_usage.lock){....}-{2:2}, at: raw_spin_lock_irqsave_sdp_contention+0x5b/0xe0

but task is already holding lock:
ffff8df7fb9bdae0 (&rq->__lock){-.-.}-{2:2}, at: raw_spin_rq_lock_nested+0x24/0xb0

which lock already depends on the new lock.

the existing dependency chain (in reverse order) is:

-> #3 (&rq->__lock){-.-.}-{2:2}:
       lock_acquire+0xcf/0x310
       _raw_spin_lock_nested+0x2e/0x40
       raw_spin_rq_lock_nested+0x24/0xb0
       ___task_rq_lock+0x42/0x110
       wake_up_new_task+0x198/0x440
       kernel_clone+0x118/0x3c0
       user_mode_thread+0x61/0x90
       rest_init+0x1e/0x160

-> #2 (&p->pi_lock){-.-.}-{2:2}:
       lock_acquire+0xcf/0x310
       _raw_spin_lock_irqsave+0x39/0x60
       try_to_wake_up+0x57/0xbb0
       create_worker+0x17e/0x200
       workqueue_init+0x28d/0x300

-> #1 (&pool->lock){-.-.}-{2:2}:
       lock_acquire+0xcf/0x310
       _raw_spin_lock+0x30/0x40
       __queue_work+0xdb/0x6d0
       queue_delayed_work_on+0xc7/0xe0
       srcu_gp_start_if_needed+0x3cc/0x540
       __synchronize_srcu+0xf6/0x1b0

-> #0 (rcu_tasks_trace_srcu_struct_srcu_usage.lock){....}-{2:2}:
       check_prev_add+0xe1/0xd30
       __lock_acquire+0x1561/0x1de0
       lock_acquire+0xcf/0x310
       _raw_spin_lock_irqsave+0x39/0x60
       raw_spin_lock_irqsave_sdp_contention+0x5b/0xe0
       srcu_gp_start_if_needed+0x92/0x540
       bpf_selem_unlink+0x267/0x5c0
       bpf_task_storage_delete+0x3a/0x90
       bpf_prog_134dba630b11d3b7_scx_pmu_task_fini+0x26/0x2a

Chain exists of:
rcu_tasks_trace_srcu_struct_srcu_usage.lock --> &p->pi_lock --> &rq->__lock

Possible unsafe locking scenario:

        CPU0                    CPU1
        ----                    ----
   lock(&rq->__lock);
                               lock(&p->pi_lock);
                               lock(&rq->__lock);
   lock(rcu_tasks_trace_srcu_struct_srcu_usage.lock);

*** DEADLOCK ***
```

### Dissecting the Circular Dependency

The deadlock chain is four locks deep, but the circular dependency involves three key relationships:

**Chain Visualization:**
```
srcu_usage.lock -> pool->lock -> pi_lock -> rq->__lock
       ^                                       |
       |                                       |
       +----------- DEADLOCK CYCLE -----------+
```

**The Dependency Chain:**

1. **rq->__lock → srcu_usage.lock** (Bug 2 scenario)
   - sched_ext holds `&rq->__lock` 
   - Calls `bpf_task_storage_delete()`
   - Eventually needs `srcu_usage.lock` in `srcu_gp_start_if_needed()`

2. **srcu_usage.lock → pool->lock** (Existing dependency)
   - `srcu_gp_start_if_needed()` holds `srcu_usage.lock`
   - Calls `queue_delayed_work_on()`
   - Needs workqueue `&pool->lock` in `__queue_work()`

3. **pool->lock → pi_lock** (Existing dependency)  
   - Workqueue initialization calls `create_worker()`
   - `create_worker()` calls `try_to_wake_up()`
   - `try_to_wake_up()` needs `&p->pi_lock`

4. **pi_lock → rq->__lock** (Existing dependency)
   - `wake_up_new_task()` holds `&p->pi_lock`
   - Calls `___task_rq_lock()` → `raw_spin_rq_lock_nested()`
   - Needs `&rq->__lock`

**The Deadlock Scenario:**

```
CPU0: Holds rq->__lock, waiting for srcu_usage.lock
CPU1: Holds srcu_usage.lock, queues work needing pool->lock
CPU2: pool->lock → pi_lock → rq->__lock (blocks on CPU0)
CPU0: Still waiting for srcu_usage.lock (blocks on CPU1)
```

The cycle completes because the workqueue subsystem establishes a lock ordering that eventually leads back to the runqueue lock.

## The SRCU Context Problem

Looking at the kernel source in `srcutree.c`, we can see how SRCU's locking works. The problematic function is:

```c
static unsigned long srcu_gp_start_if_needed(struct srcu_struct *ssp,
                                             struct rcu_head *rhp, bool do_norm)
{
    unsigned long flags;
    // ... setup code ...
    
    spin_lock_irqsave_sdp_contention(sdp, &flags);
    
    if (rhp)
        rcu_segcblist_enqueue(&sdp->srcu_cblist, rhp);
    
    // Grace period startup logic that can call queue_delayed_work_on()
    // ... more logic ...
    
    spin_unlock_irqrestore_rcu_node(sdp, flags);
    // ... return ...
}
```

The function `spin_lock_irqsave_sdp_contention()` eventually calls the locking wrappers. Even after Paul's raw spinlock conversion, the problem persists because SRCU's grace period logic calls into the workqueue subsystem while holding locks.

The `srcu_gp_start_if_needed()` function calls `queue_delayed_work_on()` when starting grace periods:

```c
if (likely(srcu_init_done))
    queue_delayed_work(rcu_gp_wq, &sup->work, !!srcu_get_delay(ssp));
```

This is where the `srcu_usage.lock → pool->lock` dependency gets established.

## Why Fixing Bug 1 Created Bug 2

The evolution from Bug 1 to Bug 2 illustrates a classic kernel development challenge, but it's crucial to understand that these are **different categories of bug**:

**Bug 1 (Invalid wait context):** An atomic-context problem. SRCU used regular spinlocks (sleeping under RT), so calling it from raw spinlock context was illegal. This is specific to PREEMPT_RT.

**Paul's initial fix:** Convert SRCU's locks to raw spinlocks.

**Bug 2 (Circular lock dependency):** A lock ordering problem. Now that SRCU uses raw spinlocks, it *can* be called from atomic context, but `srcu_gp_start_if_needed()` calls `queue_delayed_work_on()` while holding `srcu_usage.lock`. This creates a dependency chain through the workqueue system that circles back to the runqueue lock. This is **not** an atomic-context issue. It's a genuine deadlock that lockdep catches regardless of PREEMPT_RT.

The fix for the immediate problem (sleeping in atomic context) unmasked a deeper architectural issue: SRCU's grace period machinery calls into the workqueue subsystem, and the workqueue's lock ordering is incompatible with being called while holding scheduler locks.

## Current Status and Proposed Solutions

### Andrea's BPF-side workaround (rejected)

Andrea Righi proposed adding a `reuse_now` flag to `bpf_selem_unlink()` to free storage immediately via `call_rcu()` instead of `call_rcu_tasks_trace()` when the task is dead. Kumar Kartikeya Dwivedi shot this down:

*"The fix you provided below unfortunately can't work, we cannot free the selem immediately as the program may have formed pointers to the local storage before calling delete."*

### irq_work deferral (short-term fix, but incomplete)

Boqun Feng proposed using `irq_work` to defer `call_srcu()` out of the atomic context, essentially restoring the pre-conversion behavior of `call_rcu_tasks_trace()`. Sebastian Siewior pushed for something more substantial, but Boqun argued this is a reasonable short-term fix for v7.0.

However, **irq_work deferral alone doesn't fix Bug 2**. The circular lock dependency exists because `srcu_gp_start_if_needed()` calls `queue_delayed_work_on()` while holding `srcu_usage.lock`, regardless of what context the original `call_srcu()` came from. Even if BPF defers its `call_srcu()` via irq_work, the dependency chain remains:

```
srcu_usage.lock -> pool->lock -> pi_lock -> rq->__lock
       ^                                       |
       |                                       |
       +----------- DEADLOCK CYCLE -----------+
```

This cycle exists because SRCU's grace period machinery uses the workqueue system, and the workqueue's lock ordering (via `create_worker() -> try_to_wake_up()`) creates a path back to the runqueue lock. Any code path that holds `rq->__lock` and eventually calls `call_srcu()` will trigger this, with or without irq_work deferral.

Note: reverting the RCU Tasks Trace to SRCU conversion (commit c27cea4416a3) isn't straightforward either, since the original body of the RCU Tasks Trace code was deleted. Perhaps we should have kept an easier escape hatch.

### The real fix needs to address `queue_delayed_work()` under SRCU locks

Boqun noted that *"the root cause of the issues is that BPF is actually like a NMI unless the code is noinstr"* and proposed a general defer mechanism for BPF. But even a general BPF defer mechanism doesn't fix the `srcu_usage.lock -> pool->lock` dependency that SRCU itself creates.

Kumar suggested the fix should be in SRCU: *"defer the pi->lock -> rq->lock in call_srcu() when irqs_disabled() is true."* But this really means SRCU needs to avoid calling `queue_delayed_work_on()` while holding its own locks, or use an alternative mechanism (like a timer, which Sebastian noted should be safe under rq/pi locks) to kick off grace period processing.

### Possible longer-term approaches

**1. SRCU avoids workqueue while holding locks:** Restructure `srcu_gp_start_if_needed()` so that the `queue_delayed_work_on()` call happens after releasing `srcu_usage.lock`, breaking the dependency chain.

**2. Timer-based grace period initiation:** Replace the workqueue-based grace period scheduling with timers when called from constrained contexts. Timers don't take `pool->lock` and avoid the circular dependency.

**3. General BPF defer mechanism:** A systematic way for BPF to defer lock-holding operations to a safe context. This addresses the broader problem (BPF calling `kfree_rcu()` and other lock-holding functions) but doesn't fix the SRCU-internal lock ordering issue by itself.

## The Broader Implications

This bug exemplifies several challenges facing modern kernel development:

**Legacy System Integration:** SRCU was designed when sleepable contexts were the norm for its use cases. Modern subsystems like BPF and sched_ext have atomic context requirements that weren't anticipated.

**Lock Ordering Complexity:** As the kernel grows more interconnected, establishing safe lock ordering becomes increasingly difficult. The workqueue system's lock requirements interact with scheduler locks in ways that constrain other subsystems.

**RT Kernel Constraints:** PREEMPT_RT's conversion of regular spinlocks to sleeping locks exposes latent atomicity assumptions throughout the codebase.

**Subsystem Boundaries:** Different kernel subsystems make different assumptions about calling context, lock ordering, and preemption behavior. When these assumptions clash, the interactions can be subtle and difficult to predict.

## Looking Forward

The resolution of these bugs will likely influence how we approach atomic context synchronization throughout the kernel. The interaction between BPF, sched_ext, SRCU, and RT kernels represents the growing complexity of modern kernel subsystems.

As Paul McKenney noted in his acknowledgment of the circular dependency issue, this is "*something to fix*" that requires careful consideration of the broader SRCU architecture.

The kernel development community is grappling with fundamental questions: How do we maintain the flexibility and composability that makes Linux powerful while ensuring that subsystem interactions remain predictable and safe? 

This BPF/SRCU dependency mess may be a specific bug, but it's also a window into the ongoing evolution of kernel synchronization primitives in an increasingly complex and real-time-aware world.

## Reproduction

Andrea provided a clean reproduction case:

```bash
$ cat << EOF > /tmp/config
CONFIG_BPF=y
CONFIG_BPF_SYSCALL=y
CONFIG_SCHED_CLASS_EXT=y
CONFIG_PREEMPT=y
CONFIG_DEBUG_LOCKDEP=y
CONFIG_DEBUG_ATOMIC_SLEEP=y
CONFIG_PROVE_LOCKING=y
EOF

$ vng -vb --config /tmp/config
$ vng -v -- "scx_cosmos & schbench -L -m 4 -t 48 -n 0"
```

This reliably triggers both the original invalid wait context and the circular dependency, depending on which kernel version and patches are applied.

---

*This analysis is based on ongoing LKML discussions as of March 2026. The final resolution may involve different approaches as the kernel community continues to refine the solution.*