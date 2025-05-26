---
layout: post
date: 
title: "Modeling (lack of) store ordering using PlusCal - and a wishlist"
comments: true
categories: [pluscal, tla+, formalmethods]
---
The Message Passing pattern (MP pattern) is shown in the snippet below
(borrowed from LKMM docs). Here, P0 and P1 are 2 CPUs executing some code. P0
stores a message in `buf` and then signals to consumers like P1 that the
message is available -- by doing a store to `flag`. P1 reads `flag` and if it
is set, knows that some data is available in `buf` and goes ahead and reads it.
However, if `flag` is not set, then P1 does nothing else. Without memory
barriers between P0's stores and P1's loads, the stores can appear out of order
to P1 (on some systems), thus breaking the pattern. The condition `r1 == 0 and
r2 == 1` is a failure in the below code and would violate the condition. Only
after the `flag` variable is updated, should P1 be allowed to read the `buf`
("message").

```
        int buf = 0, flag = 0;

        P0()
        {
                WRITE_ONCE(buf, 1);
                WRITE_ONCE(flag, 1);
        }

        P1()
        {
                int r1;
                int r2 = 0;

                r1 = READ_ONCE(flag);
                if (r1)
                        r2 = READ_ONCE(buf);
        }
```

Below is a simple program in PlusCal to model the "Message passing" access
pattern and check whether the failure scenario `r1 == 0 and r2 == 1` could ever
occur. In PlusCal, we can model the non deterministic out-of-order stores to
`buf` and `flag` using an `either or` block. This makes PlusCal evaluate both
scenarios of stores (store to `buf` first and then `flag`, or viceversa) during
model checking. The technique used for modeling this non-determinism is similar
to how it is done in Promela/Spin using an "if block" (Refer to Paul McKenney's
perfbook for details on that).

```
EXTENDS Integers, TLC
(*--algorithm mp_pattern
variables
    buf = 0,
    flag = 0;

process Writer = 1
variables
    begin
e0:
       either
e1:        buf := 1;
e2:        flag := 1;
        or
e3:        flag := 1;
e4:        buf := 1;
        end either;
end process;

process Reader = 2
variables
    r1 = 0,
    r2 = 0;  
    begin
e5:     r1 := flag;
e6:     if r1 = 1 then
e7:         r2 := buf;
        end if;
e8:     assert r1 = 0 \/ r2 = 1;
end process;

end algorithm;*)
```

Sure enough, the `assert r1 = 0 \/ r2 = 1;`  fires when the PlusCal program is run through the TLC model checker.

I do find the `either or` block clunky, and wish I could just do something like:
```
non_deterministic {
        buf := 1;
        flag := 1;
}
```
And then, PlusCal should evaluate both store orders. In fact, if I wanted more than 2 stores, then it can get crazy pretty quickly without such a construct. I should try to hack the PlusCal sources soon if I get time, to do exactly this. Thankfully it is open source software.

Other notes:

* PlusCal is a powerful language that translates to TLA+. TLA+ is to PlusCal what assembler is to C. I do find PlusCal's syntax to be non-intuitive but that could just be because I am new to it. In particular, I hate having to mark statements with labels if I don't want them to atomically execute with neighboring statements. In PlusCal, a label is used to mark a statement as an "atomic" entity. A group of statements under a label are all atomic. However, if you don't specific labels on every statement like I did above (`eX`), then everything goes under a neighboring label. I wish PlusCal had an option, where a programmer could add implict labels to all statements, and then add explicit `atomic { }` blocks around statements that were indeed atomic. This is similar to how it is done in Promela/Spin.

* I might try to hack up my own compiler to TLA+ if I can find the time to, or better yet modify PlusCal itself to do what I want. Thankfully the code for the PlusCal translator is open source software.
