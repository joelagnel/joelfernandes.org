---
layout: post
title: "Why attention has an output projection: heads add into the residual stream"
date: 2026-11-21 12:00:00 -0500
categories: [machine-learning]
tags: [attention, multi-head-attention, residual-stream, output-projection, transformers]
author: Joel Fernandes
description: "Multi-head attention's output projection W_O does not mix the heads: each head writes its own contribution into the residual stream, and W_O only chooses where it lands. Notation, proof and a checked numeric example."
published: true
---

In multi-head attention, each head produces its own output, the outputs are concatenated, and the result is multiplied by an **output projection** called W<sub>O</sub>. Why is that matrix needed? It is easy to say it "mixes the heads", but that is not what it does. The short version: **the heads are independent and additive.** Each head adds its own contribution into the residual stream, and W<sub>O</sub> only decides *where* in the stream that contribution lands.

This post works through it with explicit shapes and a small numeric example (4-dimensional model, two heads, two tokens, checked with numpy). The framing follows [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) by Elhage et al.

**[Download this post as a PDF](/resources/attention-output-projection-explained.pdf)**

## The big picture

<style>
.wo { line-height: 1.5; }
.wo-fig { margin: 1.2em 0; }
.wo-scroll { overflow-x: auto; }
.wo-scroll svg { display: block; width: 100%; min-width: 680px; height: auto; }
.wo-fig figcaption { margin-top: .5em; font-size: .92em; line-height: 1.45; }
.wo h2 { margin-top: 1.4em; }
.wo table.sym { border-collapse: collapse; width: 100%; font-size: .92em; }
.wo table.sym td, .wo table.sym th { border: 1px solid #ccc; padding: 3px 6px; text-align: left; vertical-align: top; background: #fff; }
.wo table.sym th { background: #eef2f7; }
.wo code, .wo .v { font-family: "DejaVu Sans Mono", Menlo, monospace; font-size: .9em; background: #f3f3f3; padding: 0 3px; border-radius: 2px; }
.wo .box { background: #f6f9fc; border-left: 3px solid #3b73b9; padding: 6px 10px; margin: 8px 0; overflow-x: auto; }
.wo .key { background: #fff8e6; border-left: 3px solid #d9a100; padding: 6px 10px; margin: 8px 0; overflow-x: auto; }
.wo .eq { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 10px; margin: 10px 0; }
.wo .op { font-size: 1.3em; padding: 0 2px; }
.wo .mb { text-align: center; }
.wo .nm { font-size: .85em; color: #444; margin-top: 2px; max-width: 14em; }
.wo table.mat { border-collapse: collapse; border-left: 2px solid #333; border-right: 2px solid #333; border-radius: 4px; margin: 0 auto; }
.wo table.mat td { padding: 2px 6px; text-align: right; font-family: "DejaVu Sans Mono", Menlo, monospace; font-size: .95em; border: 0; background: none; }
.wo table.mat.a td { background: #dbe9ff; }
.wo table.mat.b td { background: #ffe3cc; }
.wo .small { font-size: .9em; color: #444; }
.wo ul { margin: 4px 0 4px 18px; padding: 0; }
.wo li { margin: 3px 0; }
</style>
<figure class="wo-fig">
<div class="wo-scroll">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 500" role="img" aria-label="Several attention heads each read the residual stream, compute their output, project it through their own slice of W_O and add it back into the residual stream" font-family="Helvetica, Arial, sans-serif">
<defs>
<marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1a1a1a"/></marker>
</defs>
<rect x="0" y="0" width="900" height="500" fill="#ffffff"/>
<text x="30" y="26" font-size="15" font-weight="bold" fill="#1a1a1a">Residual stream (X, n &#215; d): every head reads it, and every head adds its own write back into it</text>
<rect x="30" y="40" width="840" height="40" rx="8" fill="#f0f0f0" stroke="#555" stroke-width="2"/>
<text x="46" y="65" font-size="15" fill="#1a1a1a">X</text>
<text x="406" y="64" font-size="14" text-anchor="middle" fill="#1a1a1a">X<tspan baseline-shift="sub" font-size="10">next</tspan> = X + &#931;<tspan baseline-shift="sub" font-size="10">h</tspan> Z<tspan baseline-shift="sub" font-size="10">h</tspan>&#183;W<tspan baseline-shift="sub" font-size="10">O,h</tspan></text>
<!-- head 1 -->
<rect x="40" y="110" width="240" height="250" rx="10" fill="#f4f8ff" stroke="#3b73b9" stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="78" y="132" font-size="14" font-weight="bold" fill="#2a5a99">head 1</text>
<rect x="70" y="145" width="180" height="56" rx="6" fill="#dbe9ff" stroke="#3b73b9" stroke-width="1.5"/>
<text x="160" y="168" font-size="13" text-anchor="middle" fill="#1a1a1a">own Q, K, V weights</text>
<text x="160" y="187" font-size="13" text-anchor="middle" fill="#1a1a1a">pattern P<tspan baseline-shift="sub" font-size="10">1</tspan> (n &#215; n)</text>
<rect x="70" y="226" width="180" height="46" rx="6" fill="#dbe9ff" stroke="#3b73b9" stroke-width="1.5"/>
<text x="160" y="246" font-size="13" text-anchor="middle" fill="#1a1a1a">Z<tspan baseline-shift="sub" font-size="10">1</tspan> = P<tspan baseline-shift="sub" font-size="10">1</tspan>&#183;X&#183;Wv<tspan baseline-shift="sub" font-size="10">1</tspan></text>
<text x="160" y="263" font-size="12" text-anchor="middle" fill="#444">n &#215; k</text>
<rect x="70" y="297" width="180" height="46" rx="6" fill="#dbe9ff" stroke="#1f4f8f" stroke-width="3"/>
<text x="160" y="318" font-size="13" text-anchor="middle" font-weight="bold" fill="#1a1a1a">slice W<tspan baseline-shift="sub" font-size="10">O,1</tspan></text>
<text x="160" y="335" font-size="12" text-anchor="middle" fill="#444">k &#215; d</text>
<line x1="160" y1="201" x2="160" y2="224" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<line x1="160" y1="272" x2="160" y2="295" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<path d="M55,82 L55,173 L68,173" fill="none" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<path d="M250,320 L266,320 L266,98" fill="none" stroke="#3b73b9" stroke-width="2.4"/>
<circle cx="266" cy="82" r="14" fill="#ffffff" stroke="#3b73b9" stroke-width="2.4"/>
<text x="266" y="89" font-size="20" text-anchor="middle" fill="#3b73b9">+</text>
<line x1="266" y1="98" x2="266" y2="96" stroke="#3b73b9" stroke-width="2.4" marker-end="url(#ar)"/>
<!-- head 2 -->
<rect x="320" y="110" width="240" height="250" rx="10" fill="#fff8f1" stroke="#d9822b" stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="358" y="132" font-size="14" font-weight="bold" fill="#b5651a">head 2</text>
<rect x="350" y="145" width="180" height="56" rx="6" fill="#ffe3cc" stroke="#d9822b" stroke-width="1.5"/>
<text x="440" y="168" font-size="13" text-anchor="middle" fill="#1a1a1a">own Q, K, V weights</text>
<text x="440" y="187" font-size="13" text-anchor="middle" fill="#1a1a1a">pattern P<tspan baseline-shift="sub" font-size="10">2</tspan> (n &#215; n)</text>
<rect x="350" y="226" width="180" height="46" rx="6" fill="#ffe3cc" stroke="#d9822b" stroke-width="1.5"/>
<text x="440" y="246" font-size="13" text-anchor="middle" fill="#1a1a1a">Z<tspan baseline-shift="sub" font-size="10">2</tspan> = P<tspan baseline-shift="sub" font-size="10">2</tspan>&#183;X&#183;Wv<tspan baseline-shift="sub" font-size="10">2</tspan></text>
<text x="440" y="263" font-size="12" text-anchor="middle" fill="#444">n &#215; k</text>
<rect x="350" y="297" width="180" height="46" rx="6" fill="#ffe3cc" stroke="#a85a14" stroke-width="3"/>
<text x="440" y="318" font-size="13" text-anchor="middle" font-weight="bold" fill="#1a1a1a">slice W<tspan baseline-shift="sub" font-size="10">O,2</tspan></text>
<text x="440" y="335" font-size="12" text-anchor="middle" fill="#444">k &#215; d</text>
<line x1="440" y1="201" x2="440" y2="224" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<line x1="440" y1="272" x2="440" y2="295" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<path d="M335,82 L335,173 L348,173" fill="none" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<path d="M530,320 L546,320 L546,98" fill="none" stroke="#d9822b" stroke-width="2.4"/>
<circle cx="546" cy="82" r="14" fill="#ffffff" stroke="#d9822b" stroke-width="2.4"/>
<text x="546" y="89" font-size="20" text-anchor="middle" fill="#d9822b">+</text>
<line x1="546" y1="98" x2="546" y2="96" stroke="#d9822b" stroke-width="2.4" marker-end="url(#ar)"/>
<!-- dots -->
<text x="610" y="240" font-size="34" text-anchor="middle" fill="#555">&#8230;</text>
<!-- head H -->
<rect x="640" y="110" width="220" height="250" rx="10" fill="#f6f6f6" stroke="#888" stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="678" y="132" font-size="14" font-weight="bold" fill="#555">head H</text>
<rect x="662" y="145" width="160" height="56" rx="6" fill="#e6e6e6" stroke="#888" stroke-width="1.5"/>
<text x="742" y="168" font-size="13" text-anchor="middle" fill="#1a1a1a">own Q, K, V weights</text>
<text x="742" y="187" font-size="13" text-anchor="middle" fill="#1a1a1a">pattern P<tspan baseline-shift="sub" font-size="10">H</tspan></text>
<rect x="662" y="226" width="160" height="46" rx="6" fill="#e6e6e6" stroke="#888" stroke-width="1.5"/>
<text x="742" y="246" font-size="13" text-anchor="middle" fill="#1a1a1a">Z<tspan baseline-shift="sub" font-size="10">H</tspan> = P<tspan baseline-shift="sub" font-size="10">H</tspan>&#183;X&#183;Wv<tspan baseline-shift="sub" font-size="10">H</tspan></text>
<text x="742" y="263" font-size="12" text-anchor="middle" fill="#444">n &#215; k</text>
<rect x="662" y="297" width="160" height="46" rx="6" fill="#e6e6e6" stroke="#555" stroke-width="3"/>
<text x="742" y="318" font-size="13" text-anchor="middle" font-weight="bold" fill="#1a1a1a">slice W<tspan baseline-shift="sub" font-size="10">O,H</tspan></text>
<text x="742" y="335" font-size="12" text-anchor="middle" fill="#444">k &#215; d</text>
<line x1="742" y1="201" x2="742" y2="224" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<line x1="742" y1="272" x2="742" y2="295" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<path d="M655,82 L655,173 L660,173" fill="none" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#ar)"/>
<path d="M822,320 L842,320 L842,98" fill="none" stroke="#777" stroke-width="2.4"/>
<circle cx="842" cy="82" r="14" fill="#ffffff" stroke="#777" stroke-width="2.4"/>
<text x="842" y="89" font-size="20" text-anchor="middle" fill="#777">+</text>
<line x1="842" y1="98" x2="842" y2="96" stroke="#777" stroke-width="2.4" marker-end="url(#ar)"/>
<!-- W_O stack -->
<text x="30" y="396" font-size="14" font-weight="bold" fill="#1a1a1a">The single matrix W<tspan baseline-shift="sub" font-size="10">O</tspan> ((H&#183;k) &#215; d) is just these slices stacked on top of each other:</text>
<rect x="30" y="408" width="170" height="26" fill="#dbe9ff" stroke="#1f4f8f" stroke-width="2"/>
<text x="115" y="426" font-size="13" text-anchor="middle" fill="#1a1a1a">W<tspan baseline-shift="sub" font-size="10">O,1</tspan> (rows 1 to k)</text>
<rect x="30" y="434" width="170" height="26" fill="#ffe3cc" stroke="#a85a14" stroke-width="2"/>
<text x="115" y="452" font-size="13" text-anchor="middle" fill="#1a1a1a">W<tspan baseline-shift="sub" font-size="10">O,2</tspan> (next k rows)</text>
<rect x="30" y="460" width="170" height="26" fill="#e6e6e6" stroke="#555" stroke-width="2"/>
<text x="115" y="478" font-size="13" text-anchor="middle" fill="#1a1a1a">W<tspan baseline-shift="sub" font-size="10">O,H</tspan> (last k rows)</text>
<text x="230" y="426" font-size="14" fill="#1a1a1a">concatenate-then-project, [Z<tspan baseline-shift="sub" font-size="10">1</tspan> | Z<tspan baseline-shift="sub" font-size="10">2</tspan> | &#8230; | Z<tspan baseline-shift="sub" font-size="10">H</tspan>]&#183;W<tspan baseline-shift="sub" font-size="10">O</tspan>,</text>
<text x="230" y="448" font-size="14" fill="#1a1a1a">is the same as each head writing Z<tspan baseline-shift="sub" font-size="10">h</tspan>&#183;W<tspan baseline-shift="sub" font-size="10">O,h</tspan> and adding the writes up.</text>
<text x="230" y="474" font-size="14" fill="#444">No head sees another head's output inside the layer: the only link between heads is the stream.</text>
</svg>
</div>
<figcaption>Every head reads the residual stream, computes its own output Z<sub>h</sub>, projects it with its own slice of W<sub>O</sub>, and <b>adds</b> the result back into the stream. Heads never talk to each other inside the layer.</figcaption>
</figure>

## Contents

- [Notation](#notation)
- [The layer, standard form and the equivalent form](#layer-forms)
- [Toy numbers](#toy-numbers)
- [What this means](#what-this-means)
- [Two intuitions, scored](#two-ideas)
- [Caveats](#caveats)
- [A runnable check](#a-runnable-check)

<div class="wo">
<h2 id="notation">Notation (read this first)</h2>
<p>Convention: <b>one row per token</b>. Matrices multiply on the right: a token's vector is a row, and <code>X &middot; W</code> applies W to every row.</p>
<table class="sym"><tr><th>Symbol</th><th>Meaning</th><th>Shape</th></tr>
<tr><td><code>n</code></td><td>number of tokens in the sequence</td><td>&ndash;</td></tr>
<tr><td><code>d</code></td><td>model width (the size of the residual stream)</td><td>&ndash;</td></tr>
<tr><td><code>H</code>, <code>k</code></td><td>number of heads; width of one head (<code>k = d / H</code> in most models, so <code>H&middot;k = d</code>)</td><td>&ndash;</td></tr>
<tr><td><code>X</code></td><td>layer input: one row per token (the residual stream at this layer)</td><td>n &times; d</td></tr>
<tr><td><code>Wq<sub>h</sub>, Wk<sub>h</sub>, Wv<sub>h</sub></code></td><td>head h's query, key and value weights</td><td>d &times; k</td></tr>
<tr><td><code>P<sub>h</sub></code></td><td>head h's attention pattern: softmax(X&middot;Wq<sub>h</sub> &middot; (X&middot;Wk<sub>h</sub>)<sup>T</sup> / &radic;k + causal mask). Row i = how token i spreads its attention over earlier tokens. Depends on head h's Q and K only.</td><td>n &times; n</td></tr>
<tr><td><code>Z<sub>h</sub></code></td><td>head h's output: <code>P<sub>h</sub> &middot; X &middot; Wv<sub>h</sub></code> (a weighted average of value vectors)</td><td>n &times; k</td></tr>
<tr><td><code>[Z<sub>1</sub> | Z<sub>2</sub> | &hellip;]</code></td><td>concatenation: place the Z's side by side (columns)</td><td>n &times; (H&middot;k)</td></tr>
<tr><td><code>W<sub>O</sub></code></td><td>the output projection, one matrix for the whole layer</td><td>(H&middot;k) &times; d</td></tr>
<tr><td><code>W<sub>O,h</sub></code></td><td>rows <code>(h-1)&middot;k+1 &hellip; h&middot;k</code> of W<sub>O</sub>: head h's own slice; stacking W<sub>O,1</sub> on top of W<sub>O,2</sub> &hellip; gives W<sub>O</sub></td><td>k &times; d</td></tr>
<tr><td><code>OV<sub>h</sub></code></td><td><code>Wv<sub>h</sub> &middot; W<sub>O,h</sub></code>: what head h does to a token vector, end to end</td><td>d &times; d, rank &le; k</td></tr>
</table>
<h2 id="layer-forms">The layer, standard form and the equivalent form</h2>
<div class="box"><b>Standard form (how it is implemented, one big matmul):</b><br>
<code>Out = [Z<sub>1</sub> | Z<sub>2</sub> | &hellip; | Z<sub>H</sub>] &middot; W<sub>O</sub></code> &nbsp; (n &times; d), then added to the residual stream: <code>X<sub>next</sub> = X + Out</code>.</div>
<div class="key"><b>Equivalent form (the key identity):</b><br>
<code>Out = Z<sub>1</sub>&middot;W<sub>O,1</sub> + Z<sub>2</sub>&middot;W<sub>O,2</sub> + &hellip; + Z<sub>H</sub>&middot;W<sub>O,H</sub> = &Sigma;<sub>h</sub> P<sub>h</sub> &middot; X &middot; OV<sub>h</sub></code></div>
<p><b>Why it holds (2 heads):</b> splitting a product by blocks. A row <code>[a&nbsp;b]</code> times a stacked matrix <code>[U; V]</code> equals <code>a&middot;U + b&middot;V</code>, because each output entry is a sum over the inputs' positions, and the sum over the first k positions and the sum over the last k positions just add. So <code>[Z<sub>1</sub>|Z<sub>2</sub>]&middot;[W<sub>O,1</sub>; W<sub>O,2</sub>] = Z<sub>1</sub>W<sub>O,1</sub> + Z<sub>2</sub>W<sub>O,2</sub></code>. The second equality uses <code>Z<sub>h</sub> W<sub>O,h</sub> = P<sub>h</sub> X (Wv<sub>h</sub> W<sub>O,h</sub>)</code>.</p>
<h2 id="toy-numbers">Toy numbers (d = 4, H = 2, k = 2, n = 2 tokens)</h2>
<p class="small">Head 1 is blue (columns 1-2 of the concatenation, rows 1-2 of W<sub>O</sub>), head 2 is orange. Patterns are given rather than computed: P<sub>1</sub> lets token 2 average tokens 1 and 2; P<sub>2</sub> lets each token look only at itself.</p>
<div class="eq"><div class="mb"><table class="mat "><tr><td>1</td><td>0</td><td>2</td><td>1</td></tr><tr><td>0</td><td>1</td><td>1</td><td>2</td></tr></table><div class="nm">X</div></div><div class="mb"><table class="mat "><tr><td>1</td><td>0</td></tr><tr><td>0.5</td><td>0.5</td></tr></table><div class="nm">P1</div></div><div class="mb"><table class="mat "><tr><td>1</td><td>0</td></tr><tr><td>0</td><td>1</td></tr></table><div class="nm">P2</div></div></div>
<div class="eq"><div class="mb"><table class="mat a"><tr><td>1</td><td>0</td></tr><tr><td>0</td><td>1</td></tr><tr><td>1</td><td>1</td></tr><tr><td>0</td><td>0</td></tr></table><div class="nm">Wv1 (4x2)</div></div><div class="mb"><table class="mat b"><tr><td>0</td><td>1</td></tr><tr><td>1</td><td>0</td></tr><tr><td>0</td><td>0</td></tr><tr><td>1</td><td>1</td></tr></table><div class="nm">Wv2 (4x2)</div></div></div>
<p><b>Head outputs</b> <code>Z<sub>h</sub> = P<sub>h</sub>&middot;X&middot;Wv<sub>h</sub></code> (each n &times; k):</p>
<div class="eq"><div class="mb"><table class="mat a"><tr><td>3</td><td>2</td></tr><tr><td>2</td><td>2</td></tr></table><div class="nm">Z1</div></div><div class="mb"><table class="mat b"><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>2</td></tr></table><div class="nm">Z2</div></div></div>
<p><b>Way A, concatenate then project:</b> <code>[Z<sub>1</sub>|Z<sub>2</sub>] &middot; W<sub>O</sub></code></p>
<div class="eq"><div class="mb"><table class="mat "><tr><td>3</td><td>2</td><td>1</td><td>2</td></tr><tr><td>2</td><td>2</td><td>3</td><td>2</td></tr></table><div class="nm">[Z1 | Z2]</div></div><span class="op">&middot;</span><div class="mb"><table class="mat "><tr><td>1</td><td>0</td><td>1</td><td>0</td></tr><tr><td>0</td><td>1</td><td>0</td><td>1</td></tr><tr><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>0</td><td>0</td><td>1</td><td>-1</td></tr></table><div class="nm">W_O (4x4): top 2 rows = W_O,1, bottom 2 rows = W_O,2</div></div><span class="op">=</span><div class="mb"><table class="mat "><tr><td>4</td><td>3</td><td>5</td><td>0</td></tr><tr><td>5</td><td>5</td><td>4</td><td>0</td></tr></table><div class="nm">Out</div></div></div>
<p><b>Way B, each head writes its own contribution, then add:</b> <code>Z<sub>1</sub>W<sub>O,1</sub> + Z<sub>2</sub>W<sub>O,2</sub></code></p>
<div class="eq"><div class="mb"><table class="mat a"><tr><td>3</td><td>2</td><td>3</td><td>2</td></tr><tr><td>2</td><td>2</td><td>2</td><td>2</td></tr></table><div class="nm">Z1 &middot; W_O,1</div></div><span class="op">+</span><div class="mb"><table class="mat b"><tr><td>1</td><td>1</td><td>2</td><td>-2</td></tr><tr><td>3</td><td>3</td><td>2</td><td>-2</td></tr></table><div class="nm">Z2 &middot; W_O,2</div></div><span class="op">=</span><div class="mb"><table class="mat "><tr><td>4</td><td>3</td><td>5</td><td>0</td></tr><tr><td>5</td><td>5</td><td>4</td><td>0</td></tr></table><div class="nm">Out (same as A)</div></div></div>
<p><b>Way C, with the single OV matrices:</b> <code>Wv<sub>h</sub>&middot;W<sub>O,h</sub></code> (4 &times; 4, but rank 2 each):</p>
<div class="eq"><div class="mb"><table class="mat a"><tr><td>1</td><td>0</td><td>1</td><td>0</td></tr><tr><td>0</td><td>1</td><td>0</td><td>1</td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td></tr></table><div class="nm">OV1</div></div><div class="mb"><table class="mat b"><tr><td>0</td><td>0</td><td>1</td><td>-1</td></tr><tr><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>1</td><td>1</td><td>1</td><td>-1</td></tr></table><div class="nm">OV2</div></div></div>
<p class="small">and <code>P<sub>1</sub>&middot;X&middot;OV<sub>1</sub> + P<sub>2</sub>&middot;X&middot;OV<sub>2</sub></code> also gives Out exactly (verified).</p>
<h2 id="what-this-means">What this means</h2>
<ul>
<li><b>Nothing is mixed between heads inside a layer.</b> Head h's pattern uses only its own Q and K; its output goes into the sum linearly. Two heads never influence each other in the same layer. They only affect each other <i>across</i> layers, when a later head reads what an earlier head wrote into the residual stream (the paper calls this Q-, K- and V-composition).</li>
<li><b>W<sub>O</sub> decides where each head writes.</b> Without W<sub>O</sub>, the concatenation puts head 1 in coordinates 1-2 and head 2 in coordinates 3-4, always. In the toy: head 1's write would be <code>Z<sub>1</sub></code> in columns 1-2 only. With W<sub>O,1</sub> it is <code>[3 2 3 2; 2 2 2 2]</code>: a different 2-dimensional slice of the 4-dimensional residual stream, free to overlap with other heads' slices and with what the MLPs and embeddings write.</li>
<li><b>Only the product OV<sub>h</sub> = Wv<sub>h</sub>&middot;W<sub>O,h</sub> matters.</b> You can replace Wv<sub>h</sub> by Wv<sub>h</sub>&middot;R and W<sub>O,h</sub> by R<sup>-1</sup>&middot;W<sub>O,h</sub> for any invertible k &times; k matrix R and the model is identical (verified in the toy with R = [[1,1],[0,1]]). So "V" and "O" are one linear map split in two for dimension bookkeeping: d &rarr; k, then k &rarr; d.</li>
<li><b>Same split on the other side:</b> the QK circuit is <code>Wq<sub>h</sub>&middot;Wk<sub>h</sub><sup>T</sup></code> (d &times; d, rank &le; k). It decides <i>where</i> head h looks; the OV circuit decides <i>what</i> it moves.</li>
</ul>
<h2 id="two-ideas">Two intuitions, scored</h2>
<table class="sym"><tr><th>Idea</th><th>Verdict</th></tr>
<tr><td>"With separate queries per head there is less cross-mixing, so W<sub>O</sub> compensates."</td><td>Not quite. W<sub>O</sub> does not mix the heads' attention; the sum is additive and independent. The only cross-head effects come from later layers reading the residual stream.</td></tr>
<tr><td>"It is like the last layer of an MLP: heads are hidden neurons, W<sub>O</sub> is the down-projection."</td><td>Half right. Both are parallel units summed into one shared space through a learned output matrix. But a head writes a rank-k chunk whose content depends on the attention pattern, not a scalar times a fixed vector, and no features are mixed inside the sum.</td></tr>
</table>
<h2 id="caveats">Caveats</h2>
<ul>
<li>I did not find (and am not citing) a clean experiment measuring how much quality is lost if W<sub>O</sub> is dropped. In pure linear algebra a later reader could absorb a change of basis, so W<sub>O</sub> is a <i>freedom to choose the write subspace</i>, not a mathematical necessity. The shared basis matters in practice because MLP activations and normalisation are not basis-independent.</li>
<li>The paper analyses attention-only transformers and mostly ignores MLPs and LayerNorm. It does not cover RoPE, GQA, MLA or KV-cache cost.</li>
<li>In llama.cpp, W<sub>O</sub> is the single <code>attn_output</code> matmul on the concatenated head outputs; the per-head split above is for understanding only.</li>
</ul>
<p class="small">Suggested reading: the paper's sections on the residual stream and on "attention heads are independent and additive", and the QK/OV circuit decomposition. All toy numbers were checked in numpy: Way A, B and C agree exactly, and the R-gauge check passes.</p>
</div>

## A runnable check

The numbers above come from this script. It verifies that concatenate-then-project, the per-head sum and the OV form agree exactly, and that swapping Wv<sub>h</sub> and W<sub>O,h</sub> by an invertible R changes nothing.

```python
import numpy as np
np.set_printoptions(linewidth=200)
X=np.array([[1,0,2,1],[0,1,1,2]])
P1=np.array([[1,0],[.5,.5]]); P2=np.array([[1,0],[0,1]])
Wv1=np.array([[1,0],[0,1],[1,1],[0,0]]); Wv2=np.array([[0,1],[1,0],[0,0],[1,1]])
Wo=np.array([[1,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,-1]])
Wo1,Wo2=Wo[:2],Wo[2:]
Z1=P1@X@Wv1; Z2=P2@X@Wv2
Z=np.hstack([Z1,Z2])
A=Z@Wo
B=Z1@Wo1+Z2@Wo2
OV1=Wv1@Wo1; OV2=Wv2@Wo2
C=P1@X@OV1+P2@X@OV2
for n,v in [('X',X),('Wv1',Wv1),('Wv2',Wv2),('Wo',Wo),('Z1',Z1),('Z2',Z2),('Z',Z),('Out concat',A),('Out sum',B),('OV1',OV1),('OV2',OV2),('Out via OV',C),('rank OV1',np.linalg.matrix_rank(OV1)),('rank OV2',np.linalg.matrix_rank(OV2))]:
    print(n);print(v)
print(np.allclose(A,B),np.allclose(A,C))
R=np.array([[1,1],[0,1]]);Ri=np.array([[1,-1],[0,1]])
print('gauge',np.allclose((Wv1@R)@(Ri@Wo1),OV1))
print('head1 write no Wo:',(P1@X@Wv1)); print('head1 write with Wo:',Z1@Wo1)
```
