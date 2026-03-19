---
layout: post
title: "Kernel Dependency Mess: An Age Old Reality!"
date: 2026-03-19
---

# Kernel Dependency Mess: An Age Old Reality!

The Linux kernel has always been a complex beast with intricate dependencies between subsystems. But sometimes, these dependencies create truly nasty bugs that highlight just how interconnected our kernel infrastructure has become. Recently, I've been diving into one such bug involving SRCU (Sleepable Read-Copy Update), RCU Tasks Trace, and PREEMPT_RT that perfectly illustrates the dependency mess we live with.

## The Bug: When Modern Subsystems Clash

The problem surfaced when newer kernel subsystems like BPF, sched_ext (the extensible scheduler), and various tracing mechanisms began calling `call_srcu()` from contexts where preemption is disabled. This wouldn't be a problem in a normal kernel, but with PREEMPT_RT (the real-time patches), it creates a perfect storm.

Here's what happens:

1. **SRCU traditionally uses sleeping locks** - SRCU (Sleepable RCU) was designed to allow blocking within read-side critical sections, unlike classic RCU. To support this, SRCU uses regular spinlocks internally, which can sleep under PREEMPT_RT.

2. **Modern subsystems call SRCU from atomic contexts** - BPF programs, sched_ext schedulers, and tracing code often run with preemption disabled for performance or correctness reasons. When they call `call_srcu()`, they're doing so from a context where sleeping is not allowed.

3. **PREEMPT_RT converts spinlocks to sleeping locks** - Under PREEMPT_RT, regular spinlocks become mutex-like constructs that can block. This creates a fundamental conflict: atomic context code trying to acquire sleeping locks.

## The Call Stacks: How We Got Here

Let me show you some of the call stacks that trigger this issue. From the kernel source in `kernel/rcu/srcutree.c`, the problem flows like this:

```c
call_srcu(struct srcu_struct *ssp, struct rcu_head *rhp, rcu_callback_t func)
{
    __call_srcu(ssp, rhp, func, true);
}

static void __call_srcu(struct srcu_struct *ssp, struct rcu_head *rhp,
                        rcu_callback_t func, bool do_norm)
{
    // ...
    (void)srcu_gp_start_if_needed(ssp, rhp, do_norm);
}
```

And in `srcu_gp_start_if_needed()`, we see the problematic lock acquisition:

```c
static unsigned long srcu_gp_start_if_needed(struct srcu_struct *ssp,
                                             struct rcu_head *rhp, bool do_norm)
{
    // ...
    raw_spin_lock_irqsave_sdp_contention(sdp, &flags);
    // Critical section that can't sleep
    // ...
    raw_spin_unlock_irqrestore_rcu_node(sdp, flags);
}
```

Wait, you might notice that this code uses `raw_spin_lock_*` functions, which suggests it's already using raw spinlocks. But here's the catch - this is just one part of the story.

## The Root Cause: Legacy Design Meets Modern Reality

The fundamental issue is that SRCU was designed in an era where its use cases were more limited. The original design assumptions were:

1. **SRCU would primarily be used from sleepable contexts** - The "Sleepable" in SRCU wasn't just about read-side critical sections; it reflected the expected calling context.

2. **Atomic context usage would be rare** - When SRCU was originally designed, most atomic context code used classic RCU, which has different (and stricter) guarantees.

3. **PREEMPT_RT wasn't a major consideration** - Real-time kernels were more niche, and the interaction between SRCU and RT wasn't fully explored.

But the kernel has evolved:

- **BPF programs** often run with preemption disabled for performance and to maintain consistency of per-CPU data structures.
- **sched_ext schedulers** need to call SRCU from the scheduler context, which is inherently atomic.
- **Tracing infrastructure** calls SRCU from NMI and other atomic contexts.

From looking at recent discussions, it seems the lock in question might be in the SRCU grace period machinery itself, where different parts of the implementation use different locking strategies.

## The Different Perspectives: A Tale of Many Experts

This bug has attracted attention from the kernel's top RCU and real-time experts, and their perspectives are fascinating:

### Paul McKenney's View
Paul E. McKenney, the father of RCU, has been deeply involved in the discussion. His approach typically focuses on maintaining the fundamental RCU guarantees while finding creative solutions. From the email threads I've been following, Paul seems to be exploring ways to make SRCU's locking more RT-friendly without breaking existing semantics.

### Boqun Feng's Perspective  
Boqun Feng, another RCU maintainer, brings deep knowledge of memory ordering and concurrency to the discussion. He's likely considering the memory model implications of any changes to SRCU's locking.

### Sebastian Andrzej Siewior's RT Focus
Sebastian, one of the key PREEMPT_RT maintainers, represents the real-time perspective. His concern is ensuring that RT kernels can maintain their latency guarantees while supporting modern subsystems.

### Kumar Kartikeya Dwivedi's BPF Angle
Kumar (KKD), known for his work on BPF, represents the consumer side of this issue. BPF programs need predictable, low-latency access to SRCU functionality.

### Gary Guo's Input
Gary brings additional perspective, likely from the Rust for Linux project, where memory safety guarantees interact with these low-level synchronization primitives.

## The Approaches to Fix: Multiple Paths Forward

Based on the discussions and code inspection, several approaches are being considered:

### 1. Raw Spinlocks for SRCU
One straightforward approach is converting more of SRCU's internal locking to raw spinlocks. This would make SRCU usable from atomic contexts even under PREEMPT_RT, but it comes with trade-offs:

```c
// Current problematic code (simplified):
spin_lock_irqsave(&ssp->lock, flags);  // Can sleep under RT
// ... critical section ...
spin_unlock_irqrestore(&ssp->lock, flags);

// Potential fix:
raw_spin_lock_irqsave(&ssp->lock, flags);  // Never sleeps
// ... critical section ...
raw_spin_unlock_irqrestore(&ssp->lock, flags);
```

**Pros:** Simple, maintains existing API semantics
**Cons:** Increases RT latency, might not be appropriate for all SRCU operations

### 2. Deferred Work Approach
Another approach is to defer the problematic SRCU work to a context where sleeping is allowed:

```c
call_srcu() in atomic context
    ↓
Queue work to workqueue
    ↓  
Workqueue runs __call_srcu() in sleepable context
```

**Pros:** Maintains RT properties, doesn't increase atomic section duration
**Cons:** Adds complexity and latency to SRCU operations

### 3. Context-Aware SRCU
A more sophisticated approach would be to make SRCU context-aware:

```c
void call_srcu(struct srcu_struct *ssp, struct rcu_head *rhp, 
               rcu_callback_t func)
{
    if (in_atomic())
        call_srcu_atomic(ssp, rhp, func);  // Use raw locks
    else
        call_srcu_sleepable(ssp, rhp, func);  // Use regular locks
}
```

**Pros:** Optimizes for both use cases
**Cons:** Significant complexity increase, potential for subtle bugs

### 4. Separate SRCU Variants
The most radical approach would be to provide separate SRCU implementations:

- `srcu_*()` - Original sleepable-only version
- `srcu_atomic_*()` - Atomic context-safe version with raw locks

**Pros:** Clear separation of concerns, optimal performance for each case
**Cons:** API fragmentation, migration complexity

## Kernel Code Deep Dive

Let's look at some actual kernel code to understand the problem better. Here's the current `call_srcu()` implementation:

```c
void call_srcu(struct srcu_struct *ssp, struct rcu_head *rhp,
               rcu_callback_t func)
{
    __call_srcu(ssp, rhp, func, true);
}
EXPORT_SYMBOL_GPL(call_srcu);
```

This calls `__call_srcu()`, which eventually leads to `srcu_gp_start_if_needed()`. Looking at that function:

```c
static unsigned long srcu_gp_start_if_needed(struct srcu_struct *ssp,
                                             struct rcu_head *rhp, bool do_norm)
{
    // ... setup code ...
    
    raw_spin_lock_irqsave_sdp_contention(sdp, &flags);
    
    if (rhp)
        rcu_segcblist_enqueue(&sdp->srcu_cblist, rhp);
    
    // ... grace period logic ...
    
    raw_spin_unlock_irqrestore_rcu_node(sdp, flags);
    
    // ... more logic ...
}
```

The interesting thing here is that this part of the code already uses raw spinlocks (`raw_spin_lock_irqsave_sdp_contention`). But the issue might be in other parts of the SRCU machinery, or in the interaction between SRCU and RCU Tasks Trace.

From what I can see in the email threads, the problem specifically involves the interaction with RCU Tasks Trace under PREEMPT_RT. RCU Tasks Trace is used to track tasks that are running BPF programs, and when combined with SRCU in a PREEMPT_RT environment, the locking assumptions break down.

## The Broader Picture: Why Dependencies Are Hard

This bug perfectly illustrates why kernel development is so challenging. We have:

1. **SRCU** - A synchronization primitive from the early 2000s
2. **PREEMPT_RT** - Real-time patches that change fundamental lock semantics
3. **BPF** - A modern subsystem that needs atomic context SRCU
4. **sched_ext** - A brand new scheduler extensibility framework
5. **RCU Tasks Trace** - A specialized RCU variant for tracking task execution

None of these were designed with full knowledge of how they'd interact with the others. Each represents the state-of-the-art thinking of its time, but when combined, they create unexpected emergent behaviors.

## Where Things Stand: The Current Status

As of this writing (March 2026), the discussion is ongoing. From what I can see in the NVIDIA email threads I've been following, the kernel developers are leaning toward a solution that involves:

1. **Strategic use of raw spinlocks** in critical SRCU paths
2. **Possible API additions** to provide atomic context-safe variants
3. **Careful testing** with RT kernels to ensure latency guarantees

The challenge is that any solution needs to:
- Maintain backward compatibility
- Not break RT latency guarantees  
- Support the performance requirements of modern BPF and sched_ext workloads
- Be maintainable long-term

This is exactly the kind of problem that makes kernel development both fascinating and frustrating. Every solution has trade-offs, and the "right" answer depends on which use case you prioritize.

## Lessons for the Future

This dependency mess teaches us several important lessons:

1. **Design for the future, not just today** - When creating kernel infrastructure, consider how it might be used in contexts you haven't imagined.

2. **Test interactions early and often** - Subsystem integration testing needs to include weird combinations, not just the obvious ones.

3. **Document assumptions** - Every piece of kernel code makes assumptions about its calling context. Make those explicit.

4. **Plan for evolution** - Kernel ABIs need to be designed with extensibility in mind, because requirements will change.

5. **Embrace incremental solutions** - Sometimes the "perfect" solution is the enemy of the "working" solution. 

The kernel dependency mess is an age-old reality because the kernel is a living system that must evolve while maintaining compatibility with decades of existing code. This SRCU/RCU Tasks Trace/PREEMPT_RT bug is just the latest chapter in that ongoing story.

---

*This post is based on ongoing kernel development discussions and code analysis. The technical details may evolve as the final solution is implemented. For the most up-to-date information, follow the linux-kernel mailing list and the RCU development tree.*