# Automaton composition over the graph FST

Implemented follow-up: [scheduled compositional automata](SCHEDULED_AUTOMATA.md)
combines whole-atom and prefix execution, bounded planning, and lazy compiled masks.

An implemented follow-up, [On-demand automaton product](ON_DEMAND_JOIN.md), keeps
relations in the source FST and expands their constrained product during traversal.

The automaton direction deserves an implementation experiment alongside the
[planner proposal](PROPOSED_JOIN.md). Planning changes which search runs;
composition can change the representation of that search and share repeated
subproblems. This document distinguishes existing capabilities, an exploratory
measurement, and a proposed architecture. No production executor was changed.

Implemented prototypes and their comparisons are now available in
[PROTOTYPE_RESULTS.md](PROTOTYPE_RESULTS.md).

## What transfers from string FSTs

A query automaton and a data acceptor can be traversed together through product
states. The Rust `fst` interface exposes acceptance, transitions, `can_match`,
and `will_always_match`, plus intersection, union, and complement. Its alphabet
is bytes; ours is the level-dependent S/P/O slot encoding. The principle
transfers when the query and data consume the same encoded symbols.
[Source: `fst::Automaton`](https://docs.rs/fst/latest/fst/automaton/trait.Automaton.html).

There are three different operations behind the word “merge”:

| Operation | Meaning here |
|---|---|
| Product/intersection | Accept a suffix only when all participating constraints accept it |
| Union | Accept suffixes allowed by either constraint, with proper state transitions for both branches |
| State minimization/sharing | Represent equivalent remaining accepted languages with one state |

Finite-state transducer composition connects one machine's output labels to
another's input labels. That is useful for relational composition, but does not
by itself express every graph-pattern join with shared variables. Our current
triple representation is primarily a finite-language acceptor over encoded
coordinates; its storage deltas are navigation metadata, not semantic output
labels. Do not import weighted-transducer assumptions without an encoding.

OpenFst provides delayed operations and look-ahead composition filters. They
support the engineering direction of constructing only reached product states
and rejecting incompatible continuations early; their particular filters are
not a ready-made join algorithm for our layout.
[Sources: efficiency guidance](https://openfst.org/twiki/bin/view/FST/FstEfficiency),
[composition filters](https://openfst.org/twiki/bin/view/FST/FstAdvancedUsage).

## Where the current implementation stops

`src/level_automaton.rs` already separates a walk from its restrictions and
implements `And` using tuple states. However:

- The current `can_match` implementations always return true. A `Multi` state
  can have no live ranges while still reaching the node-read path.
- There is no terminal acceptance method. The walker emits leaf slots without
  advancing and checking a final query state. That is appropriate for the
  current prefix-restriction masks, but must change for general automata.
- There is no universal-remaining-suffix state exposed to the walker.
- `Multi` carries and clones a vector of optional range states when advancing.
- Composition is interpreted through component objects; there is no flattened,
  canonical query-state table.
- `AutomatonJoin` bypasses those objects entirely. It builds exact/free masks
  per atom entry and performs nested complete walks. Its name does not mean
  that the whole join is already a compiled automaton intersection.

The data builder already shares identical continuation structure within each
level. Query compilation can exploit an analogous idea: the identity of a state
should describe its remaining accepted language, rather than the full history
used to reach it.

## First improvement: compile filter automata

Compose restrictions before entering the node loop. Normalize overlapping
intervals and simplify constants, empty branches, and universal branches.
Compile reachable query states into compact IDs and transition tables; merge
equivalent residual states where the compilation budget permits it.

```text
CompiledFilter:
    start_id
    allowed[level, state_id] -> slot bitmap
    next[level, state_id, slot] -> next state_id
    accepting[terminal_state_id] -> bool
    live[level, state_id] -> conservative possibility of an accepted completion
    universal[level, state_id] -> every valid remaining encoding is accepted

COMPILE_FILTER(expression, schedule, state_budget):
    normalize expression and interval unions
    simplify EMPTY, ALL, and redundant components
    construct only reachable product states
    assign IDs to shared residual states
    precompute slot masks and transitions
    compute terminal acceptance and backward liveness
    recognize universal remaining languages when provable
    if construction reaches state_budget:
        retain an exact interpreter or lazy representation for unfinished parts
        # Unknown liveness stays true; never turn uncertainty into rejection.
    return compiled filter
```

An Exact/Free filter should still lower to the existing per-level mask array.
Do not require table lookup, hashing, or generic state dispatch for that case.
The composed representation is an intermediate form that can specialize into
different kernels, not a requirement to run every query through a general DFA.

The generalized walker must consume the final symbol too:

```text
WALK(data_node, level, query_state, output_prefix):
    if not live[level, query_state]: return
    if universal[level, query_state]:
        enumerate the data suffix without further query-state checks
        return

    candidates = data_node.word AND allowed[level, query_state]
    for slot in set_bits(candidates):
        next_state = next[level, query_state, slot]
        if slot completes the key:
            if accepting[next_state]: emit reconstructed key
        else if live[level + 1, next_state]:
            WALK(data_node.child(slot), level + 1, next_state,
                 extend(output_prefix, slot))
```

Union must send a component to DEAD when that component cannot consume the
chosen slot, even if the other component can. Complement requires terminal
acceptance and a total transition semantics over the chosen finite universe;
it is not obtained by complementing every prefix mask. Equivalent-state merging
must respect level, alphabet interpretation, and terminal acceptance.

## Second improvement: reuse product continuations

Once both operands have stable residual identities, cache reached products:

```text
ProductKey = (data_state_id, query_residual_id, level, interpretation_id)
```

The interpretation identifies schedule, variable mapping, and other context
not already included in the state IDs. A residual ID must include every
remaining constraint that can affect future acceptance. Prefix information can
be omitted only after proving it irrelevant or translating it into residual
constraints. Hash equality must verify full keys, not assume collision freedom.

Separate what can be reused from how answers are delivered:

- **Existence:** memoize whether the product has an accepting suffix.
- **Counting:** reuse suffix counts and sum them for each distinct incoming
  logical prefix. A global visited set would undercount.
- **Enumeration:** reuse accepted suffix structure, but replay it under each
  incoming output prefix. The shared physical node can represent different
  complete answers at different prefixes.

Start with query-local, bounded storage. Eviction means recomputation. Exhausting
a cache budget must never mean skipping an answer. Prefer small IDs, indexed
tables, and caching at repeated or expensive subproblems over a mandatory hash
lookup at every node.

### An exploratory negative result

I tested a deliberately simple negative cache in an isolated source copy. Its
key contained atom identity, data address, level, and the still-unconsumed exact
coordinate bits. Free dimensions were fixed by the atom plan. It cached only
“this atom has no matching leaf,” not a later join continuation's failure.
Each visit started with an empty cache, capped at 65,536 entries.

On the original class-first skew fixture, five warm repetitions produced:

| Variant | Median | Node reads | Answers |
|---|---:|---:|---:|
| Current AutomatonJoin | 111.40 ms | 8,192,603 | 1,053 |
| Atom-level negative cache | 467.40 ms | 8,096,488 | 1,053 |
| Current walk, rare-first order | 5.46 ms | 216,250 | 1,053 |

The cache recorded 96,089 hits, but eliminated only about 1.2% of node reads.
It filled its capacity, with 40-byte keys before hash-table allocation overhead.
Many saved checks were already cheap empty-node tests. Full binding vectors
matched the original executor and the fixture's source-derived oracle.

This experiment does not measure a compiled filter or a composed join product.
It shows that storage sharing alone does not guarantee profitable cache reuse:
different query residuals can distinguish visits to the same data node. It also
shows why unconditionally adding a HashSet probe to the hot walker is a poor
starting implementation for this fixture. Raw data is in
[`lab/product-memo.csv`](lab/product-memo.csv).

To reproduce the prototype, run `python3 examples/join/lab/product_memo_probe.py`.
It prints a temporary crate directory. Build/run that crate's `investigate`
example with the `profiling` profile and:

```text
JOIN_INPUT_ORDER=1
JOIN_CASE=skew-c20-r0.01-o500000
JOIN_REPS=5
```

The extra executor is named `memo`. This is an exploratory, per-atom negative
cache; it is not the proposed general composition implementation.

## Third improvement: represent joins over a common assignment alphabet

For a query with variables V, encode a complete assignment using one chosen
variable/bit order. An atom defines a language over those assignments: it
accepts exactly when the atom's instantiated triple occurs in the graph.
Variables absent from that atom are unconstrained within their valid domains.

For the triangle, the logical expression is:

```text
L_result(x, y, z) = L_R(x, y) AND L_S(y, z) AND L_T(z, x)
```

This is intersection after lifting each relation to the common variable space.
Qdag/lqdag work develops this extension-and-intersection approach for compressed
quadtrees. It is a closer relational analogue than directly combining unrelated
physical child bitmaps. Its published bounds do not automatically cover our
mixed-radix schedule.
[Source: Optimal Joins Using Compressed Quadtrees](https://doi.org/10.1145/3514231).

The difficult part is the adapter. A variable can be S in one atom and O in
another. Those dimensions can consume different coordinate bits at the same
physical level. A common assignment symbol can therefore correspond to a partial
physical slot, delayed constraints, or several physical states in an atom.
Missing variables require no physical consumption at all. This alignment must
be implemented and measured; “rename dimensions and AND the words” is incorrect.

```text
LiftedAtomState:
    physical node or canonical frontier of physical states
    partially constrained slots / delayed variable bits
    constants and repeated-variable equalities still relevant to the suffix
    common assignment position and physical schedule position

ADVANCE_LIFTED(atom_state, assignment_symbol):
    apply the symbol to every occurrence of its variable bit
    propagate newly determined physical slots and constant restrictions
    retain all feasible physical continuations
    canonicalize the residual representation, or return EMPTY
```

For illustration, the lazy product below assumes those adapters already provide
exact transitions over the same assignment alphabet. It does not solve alignment
by hiding it inside an ordinary slot-mask operation.

```text
AND_PRODUCT(left, right, position):
    if either operand is EMPTY: return EMPTY
    if left is ALL for the remaining domain: return right
    if right is ALL for the remaining domain: return left
    if left == right: return left

    key = (AND, left.id, right.id, position, common_interpretation)
    if expression_table contains key: return expression_table[key]

    result = intern lazy product expression for key
    result.accepting = left.accepting AND right.accepting
    result.successors = lazy function:
        for symbol in left.allowed_symbols AND right.allowed_symbols:
            child = AND_PRODUCT(left.advance(symbol), right.advance(symbol),
                                next_position(position))
            if child is not EMPTY: expose (symbol, child)
    return result
```

Interning the same product expression is inexpensive structural reuse; it is
not a proof that all different expressions with equivalent languages have been
minimized. Additional reduction can merge equivalent expanded continuations.
Use budgets because product states, frontiers, and determinization can grow.

Shared graph representations and cached application of Boolean operators also
have a useful precedent in decision diagrams. Their variable-order sensitivity
is especially relevant when selecting a common assignment encoding.
[Source: Bryant, Graph-Based Algorithms for Boolean Function Manipulation](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf).

The current FstJoin already supplies synchronized bit constraints and early
cross-atom pruning. An automaton implementation must demonstrate additional
sharing, simplification, or lower overhead against it. Recasting its recursion
as tuple states alone would retain the same search space.

## Fuzzy, prefix, suffix, and substring search

The graph index contains integer term IDs. Edit distance on their Morton-code
bits does not implement fuzzy matching of RDF terms. Lexical search should
normally run against the term dictionary, then feed matching IDs into a finite
set/range filter on the graph. Such an ID filter can be compiled and composed
with other graph constraints through the machinery above.

Prefix, suffix, and substring languages can all be represented by automata over
an appropriate text encoding. Whether they prune a forward index well depends
on how early they constrain that encoding; a suffix/substring query can still
visit most of it. A reverse dictionary or other text index is a separate access
path decision. Automaton expressibility does not guarantee selective access.

Graph regular-path queries are another alphabet: their symbols are traversed
edges/predicates, not stored triple-coordinate digits. The current
`src/automaton.rs` is an NFA over graph steps. It can share design principles and
compilation machinery, but composing it directly with a one-triple LevelAutomaton
would require an explicit graph-expansion adapter.

## Recommended experiment sequence

1. Compile and simplify Range/Multi filter compositions; add explicit final
   acceptance, useful liveness, and universal-suffix handling to a general walker.
   Preserve the Exact/Free array kernel. Compare compilation plus one visit and
   repeated visits separately against the current implementations.
2. Measure residual-product repetition before choosing a cache representation.
   Compare no cache, indexed product tables, and bounded caching of expensive
   continuations. Include allocation and metadata bytes in the comparison.
3. Prototype a lifted two-atom product, then a triangle, with mismatched S/O
   domain widths and repeated variables. Compare exact full bindings with all
   existing executors and explicit small-data enumeration.
4. Measure unique product states, executed transitions, output enumeration,
   first-result latency, and peak workspace. Include the uniform, skew, and
   empty group-constraint fixtures. A compact query product is useful only if
   constructing and traversing it is competitive with the current kernels.

This extends the planner proposal: choose an evaluation order and access
strategy, but also compile, simplify, and share the residual computation itself.
