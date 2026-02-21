---
layout: post
title: "The ARM Performance Cliff: Why Code Locality Matters More on Neoverse"
date: 2026-02-21
categories: [linux-kernel, performance, arm]
tags: [linux, kernel, arm, performance, bolt, autofdo]
published: false
description: "Why large applications on ARM Neoverse processors can hit a 50% performance wall, and how branch prediction, FDIP, and iTLB pressure all connect."
---

If you've been optimizing large workloads on ARM Neoverse servers (like NVIDIA Grace), you might have encountered a ghost. You scale your application up, performance scales linearly, and then suddenly — *boom*. You hit a cliff. Throughput drops by 30–50%. Perf profiles show a frontend stall massacre.

To understand why, we need to go back to basics — branch predictors, instruction fetch, and what happens when you decouple them. This is the foundation that makes the ARM performance cliff make sense.

---

## Section 1: The Fundamental Problem

The fetch unit's only job: produce a continuous stream of instruction bytes for the decoder. For straight-line code, that's trivial — `PC += 4`. For branches, it's a problem.

You don't know where to fetch next until you've *executed* the branch — which is 5–15 cycles after you fetched it. If you wait, your pipeline stalls. So the CPU guesses. That's the entire reason branch predictors exist: **speculate the next PC so fetch never stops.**

---

## Section 2: Classic Branch Predictor Components

```
┌─────────────────────────────────────┐
│          Branch Predictor           │
│                                     │
│  BHT: "is this branch taken?"       │
│  (2-bit saturating counter table,   │
│   indexed by PC)                    │
│                                     │
│  BTB: "if taken, where does it go?" │
│  (PC → target address cache)        │
│                                     │
│  RSB: "where does this ret go?"     │
│  (call/ret stack, ~8-16 entries)    │
└──────────────────┬──────────────────┘
                   │ predicted next PC
                   ▼
             ┌──────────┐
             │  Fetch   │
             └──────────┘
```

**BHT** (Branch History Table): maps PC → taken/not-taken prediction using 2-bit saturating counters.

**BTB** (Branch Target Buffer): a cache of `{branch PC → predicted target address}`. Queried in parallel with the fetch, so the predicted target is ready by end of the same cycle — driving the next fetch address immediately with no wasted cycle:

```
Cycle N:    [Fetch X] + [BTB+BHT lookup] ──► predicted target Y
Cycle N+1:  [Fetch Y]   ← redirect happens immediately
```

**RSB** (Return Stack Buffer): dedicated predictor for `ret`. Pushes on `call`, pops on `ret`. Very high accuracy because call/ret pairs are symmetric. Also the reason Spectre v2 is so painful — attackers can poison it.

---

## Section 3: The Coupled Fetch Problem

In a simple coupled design, the branch predictor and fetch unit **share a single PC register**:

```
         PC register (shared)
              │
              ├──► BTB+BHT lookup → predicted next PC (Y)
              │
              └──► I-cache fetch of current address (X)
                        │
                        └── on completion: PC ← Y
                                            │
                                   only NOW can BP predict again
```

If the I-cache misses, fetch stalls, the PC stays at X, and the branch predictor sits idle — it has nothing new to look up. The BP is throttled to the speed of the memory hierarchy.

**The contradiction:** nothing actually *requires* this. The BP already has the predicted target Y from its BTB lookup. It could immediately feed Y back as its next input, look up BTB[Y] to get Z, look up BTB[Z] to get W — building a chain of predicted addresses without waiting for a single cache line to arrive.

```
What a coupled BP is forced to do:
  Predict X→Y, wait for fetch(X), predict Y→Z, wait for fetch(Y), ...
                ↑ stall              ↑ stall
  (BP runs at the speed of the I-cache)

What the BP is actually capable of:
  Predict X→Y → Y→Z → Z→W → W→...
  (BP runs at BTB lookup speed, ~1 cycle each, no waiting)
```

The bottleneck isn't the branch predictor — it's the shared PC register that artificially couples it to slow fetch. Remove that coupling, and the BP can race arbitrarily far ahead. But you need somewhere to store those ahead-of-time predictions so fetch can consume them later.

---

## Section 4: The Cascading BTB Problem and the FTQ

Once the BP runs ahead, it chains BTB lookups using its own output as input:

```
BTB[X]=Y → BTB[Y]=Z → BTB[Z]=W → BTB[W]=V → ...
```

These predicted addresses need to go somewhere. That buffer is the **Fetch Target Queue (FTQ)**:

```
BP predicts:  X → Y → Z → W → V → U ...  (pushed as fast as BTB can be queried)

              ┌────────────────────────────┐
FTQ:          │ X │ Y │ Z │ W │ V │ U ... │
              └──┬─────────────────────────┘
                 │
                 └──► fetch unit reads from here, one at a time, at its own pace
```

The BP fills the tail; fetch drains the head. They're now fully independent.

**But here's the problem.** The BP is looking up `BTB[Y]`, `BTB[Z]`, `BTB[W]` — for addresses it has **never actually fetched or decoded**. It's doing ghost lookups, trusting that whatever is in the BTB is current and correct.

If `BTB[X]=Y` is wrong — the real branch at X goes to Q, not Y — then every entry after X in the FTQ is garbage:

```
BP predicted:   X  Y  Z  W  V  (all wrong after X, built on bad BTB[X]=Y)
Reality:        X  Q  ...      (completely different path)
```

The BP won't know until fetch catches up to X, decode figures out the actual target, and the correction propagates back. By then, the BP may have chained 20–30 wrong predictions into the FTQ.

**Does the fetch unit check before submitting to decode?**

No. The fetch unit reads an FTQ entry, fetches that cache line, and passes it straight to decode — speculatively, unconditionally. It has no idea whether the address is "correct." It just trusts the FTQ. The pipeline flush is the *only* correction mechanism:

```
FTQ head says "fetch Y" → fetch Y → decode Y → execute Y
                                                    │
                          only here do we know if Y was right
                          if wrong: flush pipeline + flush FTQ + restart BP from Q
```

This is fundamentally the same model as old-school branch prediction — speculate optimistically, flush on misprediction — just with *far more* speculation in flight:

```
Old-school (no FTQ):
  BP predicts Y → fetch Y → decode Y → execute detects mistake → flush
  (small speculative window)

With FTQ:
  BP fills FTQ with X→Y→Z→W→V → all fetched and decoded speculatively
  → execute detects mistake → flush pipeline + flush FTQ
  (much larger speculative window, bigger mess when wrong)
```

The design works because modern BTBs are >95% accurate on hot paths. The occasional flush is a known cost — and it's still a better deal than waiting for decode every cycle.

This larger speculation window is also exactly why Spectre was so severe: the CPU had already executed many speculative memory loads before the flush arrived, leaving traces in the cache.

---

## Section 5: FDIP and When It Actually Works

The FTQ gives us a window of future fetch addresses sitting in a queue before fetch has touched them. The **prefetch engine** exploits this — it looks ahead in the FTQ at entries the fetch unit hasn't consumed yet, and fires off I-cache prefetch requests for those future lines right now:

```
FTQ:  [X] ← fetch unit is here right now
      [Y]
      [Z]
      [W] ← prefetch engine is here, fires: prefetch(W), prefetch(V), prefetch(U)
      [V]
      [U]
      ...
```

By the time fetch drains down to [W], that line is already in L1I. This is **FDIP — Fetch-Directed Instruction Prefetching**. Unlike blindly prefetching the next sequential cache line, FDIP follows the predicted control flow — it prefetches what the program will actually execute, through branches and function calls.

### The math: when does it actually hide latency?

The key variable is **prefetch distance** — how many cycles ahead of fetch the prefetch engine is looking. If the FTQ has 32 entries and fetch consumes ~1 entry per cycle, the prefetch engine looking 20 entries ahead has about **20 cycles of lookahead**.

For latency to be hidden, the prefetch must be issued far enough ahead that the line arrives before fetch needs it:

```
prefetch distance (cycles) > memory latency (cycles) → latency hidden ✓
prefetch distance (cycles) < memory latency (cycles) → stall            ✗
```

Concrete numbers on Neoverse V2:

| Memory level | Latency  | FTQ lookahead (32 entries) | Hidden? |
|---|---|---|---|
| L1I hit      | ~4 cy    | 20–32 cy                   | ✓ yes   |
| L2 hit       | ~12 cy   | 20–32 cy                   | ✓ yes   |
| LLC hit      | ~40 cy   | 20–32 cy                   | ⚠ maybe |
| DRAM         | ~200 cy  | 20–32 cy                   | ✗ no    |

FDIP reliably hides L2 latency. It can hide LLC latency if the FTQ is deep enough. It **cannot** hide DRAM latency — you would need 200 cycles of lookahead, which would mean a 200-entry FTQ. That's impractical in silicon.

So FDIP's job is actually narrower than it sounds: keep the hot instruction working set in L2/LLC so that misses hit there, not DRAM. As long as that holds, the 20–32 cycle lookahead is enough.

### When it breaks: miss spacing

The other variable is **miss spacing** — how many cycles between consecutive I-cache misses. This matters because a new miss arriving before the previous prefetch has completed means stalls start stacking:

```
miss spacing > prefetch distance → each miss is prefetched in time ✓
miss spacing < prefetch distance → misses arrive faster than FDIP can service them ✗
```

**Scenario A — tight loop, small working set (~1KB):**
The hot loop fits in ~16 cache lines. After the first run, all lines are warm in L1I. Miss spacing is effectively infinite — the loop runs for millions of cycles between any eviction. FDIP barely needed; everything just hits.

**Scenario B — hot code in 5 regions, medium scatter:**
Functions are spread across 5 × 2MB regions. I-cache miss every ~50 cycles on average. FTQ lookahead is ~32 cycles. Miss spacing (50) > lookahead (32) — FDIP can issue each prefetch before the next miss arrives. L2 misses get hidden. You're ok.

**Scenario C — hot code scattered across 30+ regions:**
Now you have functions spread across many 2MB chunks. More unique cache lines in the working set, more frequent evictions. Miss every ~10 cycles. FTQ lookahead is still ~32 cycles — but the prefetch for entry [W] was issued for a DRAM-latency miss, which takes 200 cycles to return. Before it's back, 20 more misses have arrived. Prefetch requests pile up. The memory controller is saturated with instruction prefetches that are all taking 200 cycles. Fetch unit stalls waiting for lines. Pipeline starves.

This is the cliff. It's not a single threshold — it's a gradual worsening as scatter increases, until the miss spacing drops far enough below the prefetch distance that FDIP completely loses the race against memory latency.

<!-- SECTION 6: iTLB pressure and the cliff — TO BE ADDED -->
<!-- SECTION 7: The CRT — TO BE ADDED -->
<!-- SECTION 8: BOLT and code layout — TO BE ADDED -->

## References

1. <a name="ref1"></a>NVIDIA. *NVIDIA Grace Performance Tuning Guide — Compilers: Code Locality*. docs.nvidia.com. [https://docs.nvidia.com/grace-perf-tuning-guide/compilers.html](https://docs.nvidia.com/grace-perf-tuning-guide/compilers.html)
2. <a name="ref2"></a>Reinman, G., Calder, B., & Austin, T. (1999). *Fetch Directed Instruction Prefetching*. MICRO-32. [https://cseweb.ucsd.edu/~calder/papers/MICRO-99-FDP.pdf](https://cseweb.ucsd.edu/~calder/papers/MICRO-99-FDP.pdf)
3. <a name="ref3"></a>Arm. *Arm Neoverse V2 Core Software Optimization Guide*. developer.arm.com. [https://developer.arm.com/documentation/109898/0300/](https://developer.arm.com/documentation/109898/0300/)
4. <a name="ref4"></a>Panchenko, M. et al. (2019). *BOLT: A Practical Binary Optimizer for Data Centers and Beyond*. CGO 2019. [https://arxiv.org/abs/1807.06735](https://arxiv.org/abs/1807.06735)
5. <a name="ref5"></a>Xu, R. (2024). *Using Propeller with the Linux kernel*. kernel.org. [https://www.kernel.org/doc/html/latest/dev-tools/propeller.html](https://www.kernel.org/doc/html/latest/dev-tools/propeller.html)
6. <a name="ref6"></a>Corbet, J. (2024). *Kernel optimization with BOLT*. LWN.net. [https://lwn.net/Articles/993828/](https://lwn.net/Articles/993828/)
