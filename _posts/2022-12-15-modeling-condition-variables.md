---
layout: post
title: "Modeling Condition Variables using Formal Methods"
comments: true
categories: formal-methods, synchronization
---
Condition variables can seldom be used in isolation and depend on proper usage from the users of it. This article is a gentle introduction to condition variable usage using TLA+ / PlusCal to formally model it: A parent and child process waiting on each other. During this, I actually discovered a bug in my understanding, causing a deadlock which I will share in the end.

Note that you may need to refer to PlusCal and TLA+ documentation. There's a ton of it written by people who know more than I. My interest is just as a practical user and this article does not explain PlusCal much. To be honest it is quite readable any way.

First we start with a simple C program which we'll model later, a parent waiting on its child after spawning it.

```c
volatile int done = 0;
void *child(void *arg) {
    done = 1;
    return NULL;
}

int main(int argc, char *argv[]) {
    printf("parent: begin\n");
    pthread_t c;
    pthread_create(&c, NULL, child, NULL); // create child
    while (done == 0); // spin
    printf("parent: end\n");
    return 0;
}
```

This trivial program works but wastes a lot of CPU due to the spin loop, especially, say if the child runs for a long time. But it is useful to write a formal model which we'll use as the basis for more advanced design.

Following is the PlusCal program to verify this:

```
(*--fair algorithm ThreadJoinSpin

variables
    \* Spin variable used by child to signal to parent that it finished running.
    done = 0,

    \* Simulate the parent forking the child.
    childwait = 1,

    \* The below variables are for checking invariants.
    ChildAboutToExit = 0,
    ParentDone = 0;

define
    \* The invariant that has to hold true always.
    ExitChildBeforeParent == (ParentDone = 0) \/ ((ParentDone = 1) /\ (ChildAboutToExit = 1))
end define;

procedure thr_exit() begin
texit: done := 1;
ret2: return;
end procedure;

procedure thr_join() begin
check: await done = 1;
ret3: return;
end procedure;

process childproc \in {1} begin
c0: await childwait = 0;
    \* child does something for a long time.
c1: ChildAboutToExit := 1;
c2: call thr_exit();
end process;

process parent \in {2} begin
c3: childwait := 0;
c4: call thr_join();
c5: ParentDone := 1;
end process;
```

There are 2 things that I make the model verify,
1. That the ExitChildBeforeParent invariant is always satisified. This invariant confirms that under no circumstance will the parent process terminate before the child does. More precisely, the parent's join returns only after the child calls thr_exit().
2. That the program terminates without going into a deadlock.

Both of these are true with the trivial program. This gives us a foundation to replace thr_exit() and thr_join() with condition variables.

Following is a first attempt, I am only adding the "new changes" to the above program model to keep the article short.

Here is the first attempt at C program using CVs, that we will try to model check:

```c
void thr_exit() {
    done = 1;
    Pthread_cond_signal(&cv);
}

void thr_join() {
    if (done == 0)
        Pthread_cond_wait(&cv);
}
```

And the PlusCal model is as follows. We add a new set waitSet to keep track of all the waiters on the CV. Along with 2 new functions cvwait and cvsignal to emulate pthread_cond_wait() and pthread_cond_signal() APIs:

```
variables
    waitSet = {};

procedure cvwait(p) begin
    c1: waitSet := waitSet \cup {p};
    c2: await p \notin waitSet;
    c4: return;
end procedure;

procedure cvsignal() begin
    \* This if cond is needed because otherwise
    \* the with statement waits forever if waitset is empty.
    c5:
    if waitSet = {} then
        c7: return;
    end if;

    \* Non deterministically pick something to wake up.
    c8: with x \in waitSet do
        waitSet := waitSet \ {x};
    end with;
    c9: return;
end procedure;

procedure thr_exit() begin
    c11: call cvsignal();
    c13: return;
end procedure;

procedure thr_join(p) begin
    c20: if done = 0 then
        c21: call cvwait(p);
    end if;
    c23: return;
end procedure;
```

It is notable how we model wait/wake by just adding and removing the process number from a set.
The TLC model checker tells us this program deadlocks, why? Because thr_exit() can be run first and signal the CV. But there's nothing waiting for it. The parent then calls thr_wait() and waits forever. Replacing the if with while also does not help. The problem is we need the child to signal only once there's someone waiting.

Seems like we need some synchronization to make sure the signal/wait don't race with each other, let us add a lock.

```c
void thr_exit() {
    pthread_mutex_lock(&m);
    pthread_cond_signal(&c);
    pthread_mutex_unlock(&m);
}

void thr_join() {
    pthread_mutex_lock(&m);
    pthread_cond_wait(&c, &m);
    pthread_mutex_unlock(&m);
}
```

To implement the lock in PlusCal, we can just do a simple test-and-set lock:

```
procedure lock() begin
    cas: if mutex = 0 then
        mutex := 1; return;
    else
        goto cas;
    end if;
end procedure;

procedure unlock() begin
    unlock_it: mutex := 0;
    ret: return;
end procedure;
```

Anything under a label in PlusCal is atomic and executed as one unit. This helps us to model a Compare-And-Swap operation pretty nicely.

We also need to modify our CV signal/wait functions to be callable under a lock. In particular, we cannot wait on the CV with the lock held as we would then deadlock.

```
procedure cvwait(p) begin
    c0: call unlock();
    c1: waitSet := waitSet \cup {p};
    c2: await p \notin waitSet;
    c3: call lock();
    c4: return;
end procedure;

procedure cvsignal() begin
    \* This if cond is needed because otherwise
    \* the with statement waits forever if waitset is empty.
    c5:
    if waitSet = {} then
        c7: return;
    end if;

    \* Non deterministically pick something to wake up.
    c8: with x \in waitSet do
        waitSet := waitSet \ {x};
    end with;
    c9: return;
end procedure;
```

Now armed with these modeled primitives, let us use them in our join/exit functions:

```
procedure thr_exit() begin
    c10: call lock();
    c11: call cvsignal();
    c12: call unlock();
    c13: return;
end procedure;

procedure thr_join(p) begin
    c20: call lock();
    c21: call cvwait(p);
    c22: call unlock();
    c23: return;
end procedure;
```

The model checker again complaints of a deadlock! Though the signaling and wait cannot be happen at the same time, there is still the original problem of the parent running much later than the child even though exclusively, and then waiting on the CV forever. The parent should not wait if no waiting is needed. Looks like we need both the done state variable and the locking.

Lets try to model this, the C program we will model uses both locking and CVs.

```c
void thr_exit() {
    pthread_mutex_lock(&m);
    done = 1;
    pthread_cond_signal(&c);
    pthread_mutex_unlock(&m);
}

void thr_join() {
    pthread_mutex_lock(&m);
    while (done == 0) {
        pthread_cond_wait(&c, &m);
    }
    pthread_mutex_unlock(&m);
}
```

The PlusCal program now becomes:

```
procedure thr_exit() begin
    c9: call lock();
    c10: done := 1;
    c11: call cvsignal();
    c12: call unlock();
    c13: return;
end procedure;

procedure thr_join(p) begin
    c19: call lock();
    c20: while done = 0 do
        c21: call cvwait(p);
    end while;
    c22: call unlock();
    c23: return;
end procedure;
```

However, turns out the model checker fails even for this! After some staring, I discovered there is a bug in my condition variable implementation itself. The cvwait procedures needs a subtle change.

The change is that, the process waiting (in this case the parent) has to be added to the waitqueue while the lock is being held. A slight reordering of c0 and c1 lines fix the cvwait(). With this the model now passes, by satisfying the invariant while not locking up:

```
procedure cvwait(p) begin
    c1: waitSet := waitSet \cup {p};
    c0: call unlock();
    c2: await p \notin waitSet;
    c3: call lock();
    c4: return;
end procedure;
```

Model checking can be a powerful tool to clarify understanding of simple primitives, that can have subtle bugs. While it has its limitations, quick model checking can identify and rule out bugs before they show up in the wild and clarify your understanding.

As an exercise for the reader, replace the while in c20 with an if statement. You should see it fail as well.

Model checking also shows that in thread_exit(), we only need to hold the lock when setting the done variable, though care must be taken to make sure that the waitSet can be concurrently queued and dequeued into. Otherwise holding the lock may still be needed. Assuming that queue and dequeue are atomic, thread_exit() can rewritten as:

```
procedure thr_exit() begin
    c9: call lock();
    c10: done := 1;
    c12: call unlock();
    c11: call cvsignal();
    c13: return;
end procedure;
```

Note: A few examples of the C code were borrowed from the [Operating Systems: Three Easy Pieces text book](https://pages.cs.wisc.edu/~remzi/OSTEP/). I am grateful to them.
