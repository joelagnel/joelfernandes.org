---
layout: post
title: "PowerPC stack guard false positives in Linux kernel"
comments: true
categories: [kernel, stack]
---
Recently, the RCU mailing list
[received](https://lore.kernel.org/rcu/CAABZP2xVCQhizytn4H9Co7OU3UCSb_qNJaOszOawUFpeo=qpWQ@mail.gmail.com/T/#t)
a report about an SRCU function failing stack guard checks.

Stack guard canaries are a security mechanism used to detect stack buffer
overflows. This mechanism works by placing a random value, called a canary,
between the local variables and the return address on the stack. If a buffer
overflow occurs, the canary value will be overwritten and the stack guard check
will fail, indicating that the program is being attacked. False positives can
occur if the canary value is overwritten by a legitimate write operation, such
as when a large structure is copied onto the stack.

Closer inspection of the function (`srcu_gp_start_if_needed`) did not reveal
any buffers that may be overflowed.

After discussions with a number of kernel developers, it is clear what the
issue is. Firstly, credit to Boqun Feng for looking through disassembly and
pointing things out which led to the whole email chain and discovery of the
issue.

A significant hint came from Christophe who is the kernel author of PPC’s stack
protection, he mentioned:

```
Each task has its own canary, stored in task struct :

kernel/fork.c:1012:
tsk->stack_canary = get_random_canary();

On PPC32 we have register 'r2' that points to task struct at all time, 
so GCC is instructed to find canary at an offset from r2.
But on PPC64 we have no such register.

Instead we have r13 that points to the PACA struct which is a per-cpu
structure, and we have a pointer to 'current' task struct in the PACA struct.
So in order to be able to have the canary as an offset of a fixed register as
expected by GCC, we copy the task canary into the cpu's PACA struct during
_switch():

	addi	r6,r4,-THREAD	/* Convert THREAD to 'current' */
	std	r6,PACACURRENT(r13)	/* Set new 'current' */
  #if defined(CONFIG_STACKPROTECTOR)
	  ld	r6, TASK_CANARY(r6)
	  std	r6, PACA_CANARY(r13)
  #endif
```

---

In 64-bit PPC, the TLS (Thread Local Storage) cannot be pointed to by register `r2` as it is used to store the TOC (Table of Contents) pointer instead, which is used for accessing global and static data. Therefore, the kernel saves the currently running `task_struct` in the per-CPU area pointed to by `r13` instead. This is known as the Processor Access Control Area (PACA). The PACA is a per-CPU memory area that stores information about the CPU's context. One of the users of this data structure is to save the current instruction location prior to interrupt processing.

On PPC64, each task has its own stack canary, stored in the task struct. However, unlike PPC32, there is no fixed register that points to the currently running `task_struct` at all times. Instead, the per-CPU PACA struct contains a pointer to the current `task_struct`. Therefore, in order to be able to have the canary as an offset of a fixed register as expected by GCC, the task canary is copied into the PACA struct during `_switch()`. False positives can occur if GCC keeps an old value of the per-CPU struct pointer, which then gets the canary from the wrong CPU struct, leading to a different task.

This issue with storing canaries of the currently running task is related to the issue of not being able to use `r2` to point to the TLS on 64-bit PPC. The kernel must use the per-CPU area to store the currently running `task_struct`, which leads to the need to copy the task canary into the PACA struct during `_switch()`. The compiler optimization that causes the register `r13` to be cached into `r10` can then lead to false positives if GCC keeps an old value of the per-CPU struct pointer. 

### So what the heck is PACA?

This `r13` register points to a structure in the kernel called PACA which is a per-CPU memory area storing information about the CPU’s context.

Per the PPC64 [paper](https://www.kernel.org/doc/ols/2001/ppc64.pdf):

This structure contains information unique to each processor; therefore an array of PACAs are created, one for each logical processor. One of the users of this data structure to save the current instruction location prior to interrupt processing.

### So what’s up with the Canary?

As explained earlier in the article and as Christpophe answered this for me:

```
PPC64 uses a per-task canary. But unlike PPC32, PPC64 doesn't have a fixed 
register pointing to 'current' at all time so the canary is copied into 
a per-cpu struct (PACA) during _switch().

If GCC keeps an old value of the per-cpu struct pointer, it then gets 
the canary from the wrong CPU struct so from a different task.
```

### Compiler optimizations

What Christophe refers to in the last line is exactly a Compilter optimization. It turns out that from the reporter’s email, the `r10` register was used as a base pointer to the per-CPU PACA area. This means the compiler must have cached `r13` into `r10`, perhaps because it wanted to use `r13` for something else. Boqun Feng provided the following snippet which Zhouyi verified fixes the issue:

```
diff --git a/kernel/rcu/srcutree.c b/kernel/rcu/srcutree.c
        index ab4ee58af84b..f5ae3be3d04d 100644
        --- a/kernel/rcu/srcutree.c
        +++ b/kernel/rcu/srcutree.c
        @@ -747,6 +747,7 @@ void __srcu_read_unlock_nmisafe(struct srcu_struct *ssp, int idx)

                smp_mb__before_atomic(); /* C */  /* Avoid leaking the critical section. */
                atomic_long_inc(&sdp->srcu_unlock_count[idx]);
        +       asm volatile("" : : : "r13", "memory");
         }
         EXPORT_SYMBOL_GPL(__srcu_read_unlock_nmisafe);
```

In this snippet, `r13` is added to an extended inline asm statement, which instructs the compiler that `r13` may be clobbered by the asm statement, which hopefully prevents the compiler from caching its value before exiting the function. In fact this is exactly what prevents the issue.

Boqun also included a `memory` clobber which is equivalent to `barrier()` and is additional step which hopefully ensures memory access compiler optimizations don’t span the inline assembly statement.

### Finally the issue is clear

Later in the email chain, I mentioned the series of events as outlined by Christophe and Boqun which could lead to the issue:

```
The issue requires the following ingredients:
1. Task A is running on CPU 1, and the task's canary is copied into
the CPU1's per-cpu area pointed to by r13.
2. r13 is now cached into r10 in the offending function due to the compiler.
3. Task A running on CPU 1 now gets preempted right in the middle of
the offending SRCU function and gets migrated to CPU 2.
4.  CPU 2's per-cpu canary is updated to that of task A since task A
is the current task now.
5. Task B now runs on CPU 1 and the per-cpu canary on CPU 1 is now that of B.
6. Task A exits the function, but stack checking code reads r10 which
contains CPU 1's canary which is that of task B!
7. Boom.

So the issue is precisely in #2.  The issue is in the compiler that it
does not treat r13 as volatile as Boqun had initially mentioned.
```
## How do we fix it?

My / our current take on it is it appears to be a compiler bug where the register `r13` is not considered volatile (which works for user land but not for the kernel). Seher Boessenkool who has worked on similar PPC64 issues before is on the email chain and can hopefully fix it in the compiler but lets see where it goes.

As a quick hack to fix this (as shown by Boqun above),`r13` can be added to an extended inline asm statement, which instructs the compiler that `r13` may be clobbered by the asm statement, hopefully preventing the compiler from caching its value before exiting the function.

## Can we fix it in the kernel?

According to Michael Ellerman, a possible solution would be to keep current in a register (GPR) on 64-bit, but we'd need to do that in addition to the register reserved for the PACA, so that would consume another GPR which we'd need to think hard about.

There's another reason to have the canary in the PACA, according to him: The PACA is always accessible, even when the MMU is off (because it is in a register), whereas `current` isn't (in some situations).

Even though, we prefer not to use stack protector in code that runs with the MMU off — if the canary wasn't in the PACA to begin with, then we'd have a hard requirement to not use stack protector in code paths where the MMU is off.
