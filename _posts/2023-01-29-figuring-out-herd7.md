---
layout: post
title: "Figuring out herd7 models"
comments: true
categories: herd7
---
[Article is under construction]

The herd7 memory consistency tool is used to verify if certain (mostly
undesirable) outcomes (memory-related) of concurrent programs exist, given a
memory model. A memory-model restricts possible candidate executions. After
such restriction, if certain undesirable still exist, the user is notified
via their assertions in the `exists` clause of a litmus test.

This is an advanced article that goes through the different parts of the
linux-kernel.cat code, and tries to explain with examples about how/why
properties are defined the way they are. The motivation to understand this
deeply is, by understanding how to read a memory model written in CAT, it
should be possible to get a deep understanding of how memory consistency and
ordering works and how a memory model behaves. Even though the herd7 memory
model is abstract in some sense (it does not describe CPU implementation but
just a set of properites and rules on memory ordering and program execution),
it can still be considered to model of how a CPU should behave. Thus it can be
argued that you are formally defining how memory accesses in CPUs should
typically behave, if such CPUs are expected to run concurrent Linux kernel
code.

It is assumed that the reader is already familiar with how to use the tool, as
well is familiar with various notations such as `->po`, `->co` etc. This is
described in much detail in the
`tools/memory-model/Documentation/explanation.txt`. There are also several LWN
articles and conference papers on the herd7 tools which we will not replicate
here.

1. Sequential consistency per-variable (SCPV)

This property is fundamental in modern processors, and it basically means that
reads and writes to a certain variable happen in a total-order. In other words,
for a specific variable, it is not possible to observe a sequence of writes to
that variable in an order different from the order in which its values were
written.

This also applies to the writes happening on the same CPU. In a single CPU, the
writes happenning on the same variable happen in program order execution.

The way the memory model can enforce this is by defining a rule forbidding a
certain property. Let us see if we can define the violations of SCPV as a cycle
in a particular candidate execution, and then tell the model that such execution
candidates are forbidden.

Consider a program doing 2 writes to a variable:

```
P0(int *x) {
  WRITE_ONCE(*x, 2);            // event W1
  WRITE_ONCE(*x, 3);            // event W2
}
```

Without `SCPV`, this program has 2 outcomes:

1. The final value of x is 2. This happens because of the following candidate execution:

```
W2 ->co W1
```

2. The final value of x is 3. This happens because of the following candidate execution:

```
W1 ->co W2
```

We wish to forbid the pattern in #1. How do we do that?

Observe that because of program ordering, there is a relation: `W1 -> W2`.

The `->po-loc` relation links to program-ordered memory accesses happening on
the same CPU.

We can combine program ordering (`->po-loc`) and cache coherent ordering
(`->co`) to build a cycle. We can build a new relation by taking the union of
the 2:
`po-loc | co`
This relation becomes `W1 -> po-loc -> W2 -> co -> W1` for case #1 or in other
words, a cycle. So we can say `acyclic po-loc | co` to forbid the bad candidate
execution.

Another possibility is, there are writes happening on different CPUs:
```
P0(int *x) {
  WRITE_ONCE(*x, 2);            // event W1
  WRITE_ONCE(*x, 3);            // event W2
}

P1(int *x) {
  WRITE_ONCE(*x, 4);            // event W3
}
```

Here there are 6 possible candidate executions:

1.

```
W1->W2->W3
```
Final value is 4.

2.

```
W2->W1->W3
```
Final value is 4.

3.

```
W2->W3->W1
```
Final value is 2.

4.

```
W1->W3->W2
```
Final value is 3.

5.

```
W3->W1->W2
```
Final value is 3.

6.

```
W3->W2->W1
```
Final value is 2.


Here cases #3 and #6 should be forbidden. The only allowed outcomes should be 3 or 4.

#3 has the following relations:
```
W2 ->co W3
W3 ->co W1
W1 ->po-loc W2
```
A cycle can be observed when uniting all of these relations using `po-loc | co`

Thus `acyclic po-loc | co` can again be used to forbid the candidate executions #3, and similarly #6.

So far we have only considered stores, however we must order the reads from
these stores as well, and such reads cannot observe the stores to the same
variable out of order. Let us next look at an example, where the above
acyclic definition is incomplete.

Consider the following Litmus test involving read accesses:
```
C scpv-rf

{}

P0(int *x)
{
        WRITE_ONCE(*x, 2);
        WRITE_ONCE(*x, 3);
}

P1(int *x)
{
        int r1;
        int r2;

        r1 = READ_ONCE(*x);
        r2 = READ_ONCE(*x);
}

exists (1:r1=3 /\ 1:r2=2)
```

Here, we hope that the reads to variable `x` are observed by P1 in the
program-order that were written in P0. So the forbidden exists clause should
never occur.

However, if you were to build a CAT model as follows, using the previously
determined acyclic property, then the forbidden case indeed happens.

Here is the CAT code:
```
include "cos.cat"

acyclic po-loc | co
```

This can be run using herd7 as follows, with the `-show prop` options to generate a DOT graph file of the forbidden case:
```
herd7 -bell linux-kernel.bell -macros linux-kernel.def -cat test.cat scpvrf.litmus -show prop -o OUT/
```

Running this shows:
```
Test scpv-rf Allowed
States 9
1:r1=0; 1:r2=0;
1:r1=0; 1:r2=2;
1:r1=0; 1:r2=3;
1:r1=2; 1:r2=0;
1:r1=2; 1:r2=2;
1:r1=2; 1:r2=3;
1:r1=3; 1:r2=0;
1:r1=3; 1:r2=2;
1:r1=3; 1:r2=3;
Ok
Witnesses
Positive: 1 Negative: 8
Condition exists (1:r1=3 /\ 1:r2=2)
Observation scpv-rf Sometimes 1 8
Time scpv-rf 0.00
Hash=f2f1ffdc787b0e923ae8cf087fcd5b12
```
And the graph for the forbidden case is as follows:
![A graph showing failure of read sequential consistency](/images/herd7/scpv/scpvrf.svg)

It is easy to see here that a cycle exists between either `co, rf and fr`, or
`po-loc, rf and fr`.

This shows that both `->rf` and `->fr` should be included in the acyclic
relation as well. Hence to avoid the problematic candidate execution, the SCPV
property should be `acyclic po-loc | co | rf | fr`. That is indeed the case.