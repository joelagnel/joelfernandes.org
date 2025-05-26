---
layout: post
date: 
title: "Understanding Hazard Pointers"
comments: true
categories: [rcu, synchronization]
---
# **Introduction**

In concurrent systems, managing shared resources efficiently and safely is of paramount importance. Hazard pointers are a powerful synchronization mechanism that can be used to address this issue. In this post, we will explore how hazard pointers work, provide a simple example to illustrate their usage, and compare them with RCU (Read-Copy-Update). Additionally, we will dive into the implementation details of hazard pointers, including the per-CPU and per-thread data structures used to maintain them.

## **Example with a Singly Linked List**

Consider a singly linked list that is shared among multiple threads. In this example, some threads may be traversing the list while others may be adding or removing nodes. The main challenge here is to ensure that a node being accessed by a reader is not deleted or modified by a writer simultaneously.

Hazard pointers provide a mechanism to temporarily "protect" a node while it is being accessed. When a thread wants to access a node, it stores the address of the node in a hazard pointer, signaling that the node is in use. Before a writer can remove or modify a node, it must check all hazard pointers to ensure that the node is not currently being accessed by any other thread.

Here's a step-by-step illustration of how hazard pointers work:

1. Thread A starts traversing the list and wants to access node B.
2. Thread A stores the address of node B in a hazard pointer, marking it as "in use."
3. Thread B attempts to remove node B from the list but first checks all hazard pointers.
4. Thread B sees that node B is in use by Thread A (due to the hazard pointer) and must wait before removing it.
5. Thread A finishes accessing node B and clears its hazard pointer.
6. Thread B can now safely remove node B from the list.

## A reader traversing a list may need to retry

This can happen if the reader steps on a list element that was freed. Let's go through a simple example step by step to see why this might happen:

Imagine a singly-linked list with three elements: A, B, and C. We have two threads, Thread 0 and Thread 1, which read and modify the list concurrently.

1. Thread 0 wants to read element B. To do this safely, it first records a hazard pointer to element B, indicating that it is about to access B and that B should not be deleted while Thread 0 is using it.
2. Thread 1 wants to delete element B from the list. It removes B from the list by updating A's next pointer to point to C instead of B. Thread 1 checks for hazard pointers referencing B and sees that Thread 0 has a hazard pointer to B. Since Thread 0 is still using B, Thread 1 cannot delete it yet.
3. Thread 1 now wants to delete element C. It removes C from the list by updating B's next pointer to be a special value called `HAZPTR_POISON`, marking the deletion. Since no hazard pointers reference C, Thread 1 can immediately delete C.
4. Thread 0 finishes reading element B and moves on to the next element in the list. It tries to acquire a hazard pointer for B's successor, which is now marked as `HAZPTR_POISON` due to the deletion of element C. When Thread 0 encounters the `HAZPTR_POISON` value, it knows that the list has been modified concurrently and must restart its traversal from the beginning of the list.
5. Thread 0 starts over, traversing the list again from the beginning. This time, it finds the updated list without elements B and C.
6. Once Thread 0 is done using element B and releases its hazard pointer, Thread 1 is free to delete B as there are no longer any hazard pointers referencing it.

## **Comparing Hazard Pointers with RCU**

Both hazard pointers and RCU are synchronization mechanisms that allow for efficient handling of shared resources. However, they differ in several key aspects:

1. Read-side performance: RCU typically offers better read-side performance than hazard pointers, as it does not require retries. Hazard pointers, on the other hand, may involve read-side retries due to concurrent modifications, which can slow down the read-side performance.
2. Write-side performance: While hazard pointers generally provide better write-side performance than naive reference counting, they may still be slower than RCU. This is because hazard pointers require writers to check all hazard pointers before modifying an object, which can be time-consuming when there are many of them. Furthermore, traversal of per-thread data structures to scan for hazard pointers may result in cache contention when there are lots of readers storing hazard pointers.
3. Memory footprint: Hazard pointers have a minimal memory footprint, as any object not currently protected by a hazard pointer can be immediately freed. RCU, on the other hand, may require a larger memory footprint due to its deferred reclamation mechanism.
4. Grace period: RCU requires a grace period even when there are no readers, which can cause latency issues. With hazard pointers, objects can be freed instantly as soon as they are no longer being used by any thread.
5. Reclamation of memory: Hazard pointers allow for the immediate reclamation of any object not currently protected by a hazard pointer. This means that hazard pointers can be freed per-object while not freeing others. However, with RCU, reclamation is independent of specific objects and may require a larger memory footprint due to its deferred reclamation mechanism. SRCU can help some with that problem, as the readers that need to be waited on are per-object-type. However, Hazard pointers are per-object and not really per-object type so they are event better than SRCU for this point. Another big disadvantage of SRCU vs HP for this point is, SRCU has large memory footprint due to per-cpu counters, which hazard pointers does not. Imagine an srcu_struct for an object that’s just 20 bytes long but the system has 100 CPUs!

## **Implementation of Hazard Pointers**

Hazard pointers are maintained in per-CPU or per-thread data structures. This design choice allows for improved cache locality, as each thread or CPU only needs to update its own local data structure when accessing shared resources. Consequently, this reduces contention and cache coherency traffic, leading to good read-side performance. However, the write-side performance of hazard pointers may be slower than RCU, as writers need to check all hazard pointers before modifying an object. This can be time-consuming when there are many hazard pointers to check. Furthermore, traversal of per-thread data structures to scan for hazard pointers may result in cache contention when there are many readers storing hazard pointers, leading to additional overhead.

## Virtual Reference Counting using Hazard Pointers

Virtual reference counting is a technique used to avoid the overhead of maintaining reference counts for each object in a concurrent system. Instead of incrementing and decrementing object reference counts, threads maintain a list of "virtual" reference counts in the form of hazard pointers.

When a thread wants to access an object, it temporarily protects the object using a hazard pointer. This marks the object as "in use" and guarantees that it won't be freed or modified by any other thread. By counting the number of hazard pointers in the system, it is possible to determine the "value of the virtual refcount." The absence of hazard pointers implies that the number of references to an object is zero, which means that the object can be destroyed.

Virtual reference counting eliminates the need to maintain reference counts for each object resulting in the object saving space as a separate refcount need to be stored in the object. This is very useful for small objects.

## Design consideration with Hazard Pointer Implementations

### The retry dance

A dance is performed when recording an HP.

Reader has to:

1. Read the object pointer and issue a full MB.
    
    <aside>
    💡 A full MB may be avoidable, if the reader may be IPI’d by an updater, similar to how the “rude” variant of RCU-tasks avoids full MBs. The IPI itself will issue the full MB. Which means in the absence of an updater, readers can avoid a full MB.
    
    </aside>
    
2. Record it into an HP.
3. Pick up the object pointer again and compare it to the recorded HP in #2.
4. Retry if it has changed (back to #1).

This has to be done because it is possible that between #1 and #2, the object was freed by an updater.

### Recording HPs to objects which are embedded in HP-protected objects

An issue can occur if a hazard pointer is recorded to some offset within a data structure, that is itself hazard pointer protected. In this case, the updater may not be aware of hazard pointer protected data within the outer hazard pointer protected object. The updater may then wrongly conclude that there are no readers.

Various strategies can be used to solve this, such as recording a range of memory to protect instead of just the pointer, and then checking if a range of memory has any inner hazard pointer protected ranges, before freeing such memory. Another approach is to use a common memory address as the recorded pointer for both the inner and the other outer structure, such as an rcu_head in the outer structure, or the beginning/end address of the outer structure.

## In summary

Hazard pointers are an effective synchronization mechanism for read-mostly uses. If the usecase is write-mostly, then readers may have too many retries. But they offer good read-side performance, and may generally better memory footprint than RCU, SRCU and per-CPU refcounts.
