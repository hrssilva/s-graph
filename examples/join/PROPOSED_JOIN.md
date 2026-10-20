# Proposed join: cost-aware atom walks with prefix fallback

This proposal follows the [join investigation](INVESTIGATION.md). It adds a
planner around two execution strategies over the same FST:

- **Nested:** resolve one complete atom, then walk the next atom using its
  bindings. Preserve AutomatonJoin's inexpensive inner traversal.
- **Prefix:** advance all atoms through the physical subdivision schedule,
  checking shared variable bits before completing their bindings. Reuse
  FstJoin's algorithm.

The first implementation should choose one strategy for the whole query.
Executing an interleaved cyclic core with nested branches is a later extension.
This document proposes behavior; the planner, estimates, and planning probes
below have not been implemented or benchmarked.

The complementary [automaton composition proposal](AUTOMATON_COMPOSITION.md)
examines compiling and sharing the residual computation itself, including the
alignment required for joins and an exploratory negative-cache measurement.

## Why this combination

The skew fixture shows that ordering can repair AutomatonJoin: rare-edge-first
took 5.48 ms, versus 111.75 ms with class-first and 12.08 ms for FstJoin.
Changing atom order reduced node reads from 8,192,603 to 216,250. The planner
should choose that order directly instead of ranking atoms by constant count.

However, choosing the smallest relation first is also insufficient. In the
200-class fixture, the class relation is smaller than the rare-edge relation,
yet rare-edge-first still wins. A fully bound membership check costs less than
repeatedly listing an unbound object through the S/P/O schedule. Estimate
**work per input binding and the number of those bindings**, not just matches.

The empty group-constraint triangle establishes the other boundary: every atom
order expands large pairwise joins. Prefix traversal rejects it after 25 node
reads. Retaining prefix execution addresses a failure that ordering cannot fix.
RecursiveJoin traverses the same search space as FstJoin, so it is an alternative
implementation of that strategy rather than a third planning choice.

## Contract and notation

Input is a conjunction of triple patterns over a set of stored triples. Output
is a stream of complete satisfying bindings in variable-ID order; output order
is unspecified. Projection, DISTINCT, OPTIONAL, and other SPARQL operators belong
above this executor.

```text
Atom                    = (subject term, predicate term, object term)
Term                    = Constant(value) | Variable(id)
Binding                 = variable values plus an explicit bound-variable set
Order                   = a permutation of the normalized atoms
Plan                    = NestedPlan(order) | PrefixPlan(stages)
Estimate                = predicted work, uncertainty, estimated result distribution
Work                    = weighted reads, candidates, checks, mask updates, and emits
```

Estimated work is an optimizer score, not a correctness condition or a query
execution limit. An estimated zero never permits discarding an atom or answer.
Only exact data checks can establish emptiness. All callbacks propagate caller
cancellation immediately.

Validate variables and constants as the existing constructors do. Remove exact
duplicate atoms for full conjunctive evaluation. An empty conjunction emits one
empty binding; a nonempty query over an empty graph emits nothing. A repeated
variable within one atom, such as `?x :p ?x`, currently requires PrefixPlan;
the proposed nested fast path retains AutomatonJoin's existing restriction.

## 1. Build small planning statistics

Collect statistics while the input triples are already being read during build.
Use counts over distinct triples, consistent with the graph's set semantics.
Keep the FST layout unchanged, and report planning metadata bytes separately.

```text
BUILD_PLANNING_STATS(distinct_triples, physical_schedule, metadata_budget):
    collect exact counts per predicate where the budget permits
    collect bounded heavy hitters for (predicate, object) and (subject, predicate)
    collect approximate distinct subject/object counts for those groups
    optionally collect bounded samples of degrees and prefix occupancy
    retain the physical schedule and domain widths as model inputs
    stop growing approximate summaries when metadata_budget is exhausted
    return summaries, coverage information
```

Heavy-hitter absence means unknown unless the summary explicitly guarantees an
exact zero. Missing statistics lower confidence. Do not scan the whole graph at
query time to obtain exact cardinalities for planning.

Calibrate operation costs on representative workloads. Access signatures include
the bound dimensions, predicate/heavy-hitter group, and physical schedule. A
signature with subject and predicate bound has a different cost from a fully
bound triple check. Do not assume all tree-node reads have identical latency.

The estimator used below has this explicit contract:

```text
ESTIMATE_EXTENSION(prefix_summary, atom, bound_variables, stats, cost_model):
    access = classify constants and variables bound by the prefix
    infer the distribution of probe keys from prefix_summary and available stats
    estimate work_per_probe for access, including mask setup and traversal
    estimate matches_per_probe, including failed probes
    estimate the resulting binding distribution and its uncertainty
    return work_per_probe, matches_per_probe, output_summary, uncertainty
```

For a prefix expected to produce N bindings, appending atom A costs:

```text
added_work  = N * estimated_work_per_probe(A | prefix)
next_rows   = N * estimated_matches_per_probe(A | prefix)
```

Correlations and skew affect both terms. Multiplying marginal selectivities is
only a low-confidence fallback. The model must represent uncertainty rather than
claiming a precise estimate from insufficient statistics.

## 2. Choose an atom order with bounded search

A small beam retains several candidate prefixes, avoiding a greedy decision
based only on the first scan. For two or three atoms, a beam wide enough for all
permutations can compare every order. For larger queries, cap planning work.

```text
CHOOSE_NESTED_ORDER(atoms, stats, cost_model, beam_width):
    beam = [{ order: [], remaining: atoms, bound: {},
              rows: 1, work: 0, summary: one_empty_binding,
              uncertainty: none }]

    repeat number_of_atoms times:
        candidates = []
        for state in beam:
            for atom in state.remaining:
                e = ESTIMATE_EXTENSION(state.summary, atom, state.bound,
                                       stats, cost_model)
                next = extend state with atom
                next.work = state.work + state.rows * e.work_per_probe
                next.rows = state.rows * e.matches_per_probe
                next.summary = e.output_summary
                next.bound = state.bound union variables(atom)
                next.uncertainty = combine(state.uncertainty, e.uncertainty)
                candidates.append(next)

        beam = lowest_risk_adjusted_work(candidates, limit=beam_width)
        # Prefer connectivity when estimated costs are comparable.
        # Disconnected components remain legal and form Cartesian products.

    return cheapest complete candidate and its estimate
```

Risk adjustment penalizes large or uncertain intermediate expansions. The beam
optimizes an approximate model, so it provides no optimal-plan guarantee.
Statistics and calibration policy are part of the proposal to validate, not
already proven consequences of the timing results.

## 3. Choose the executor; use bounded probes when uncertain

Cost-based ordering is the first deliverable. Executor selection should then
use estimates and bounded observations, with prefix execution as the conservative
fallback. A cycle is a reason to examine risk, not an automatic dispatch rule:
the measured uniform triangle favors nested execution.

```text
PLAN_AND_VISIT(query, tree, stats, config, emit):
    atoms = validate_and_normalize(query)
    handle exact empty-query, empty-graph, and invalid-constant cases

    if any atom repeats a variable within that atom:
        return PREFIX_VISIT(compile_prefix(atoms), tree, emit)

    order, nested_estimate = CHOOSE_NESTED_ORDER(atoms, stats, config.cost_model,
                                               config.beam_width)
    nested = COMPILE_NESTED(order, tree.schedule)

    if well_supported_low_risk_nested_plan(nested_estimate, config):
        return NESTED_VISIT(nested, tree, emit)

    prefix = compile_prefix(atoms)
    prefix_probe = PILOT(prefix, config.probe_work, config.probe_rows)
    if prefix_probe.completed:
        return deliver_all(prefix_probe.buffer, emit)

    nested_probe = PILOT(nested, config.probe_work, config.probe_rows)
    if nested_probe.completed:
        return deliver_all(nested_probe.buffer, emit)

    chosen = CHOOSE_FROM_ESTIMATES_AND_PROBES(
                 nested_estimate, prefix_probe, nested_probe, config)
    discard both private probe buffers
    return chosen.visit_from_root(tree, emit)
```

```text
PILOT(plan, work_budget, row_capacity):
    execute plan with an initially empty private output buffer
    charge tree operations and other modeled work against work_budget
    cancel when work_budget is spent or another row would exceed row_capacity
    completed = traversal reached its natural end without cancellation
    return completed, buffer, observed work, per-stage observations

CHOOSE_FROM_ESTIMATES_AND_PROBES(estimate, prefix_probe, nested_probe, config):
    update cost calibration from the observed operation costs
    inspect observed failures and expansion at each atom/level
    if estimates have adequate coverage and predict a clear nested advantage,
       and the pilot has not contradicted that estimate:
        return NestedPlan
    return PrefixPlan
```

The initial version can implement the final choice conservatively: require a
configured confidence/margin and intermediate-work threshold to select nested;
otherwise choose prefix. These policy values require benchmark calibration.
Prefix total-work forecasting can be added later using sampled stage branching
and shared-prefix occupancy. It is not required to repair atom ordering.

An incomplete DFS pilot is a biased sample. Its row rate does **not** establish
the total query cardinality, and a fast first result does not guarantee cheap
completion. The planner must not interpret pilot row rate as a complete cost
estimate. In particular, uncertainty about cyclic expansions should remain
visible in the decision.

Probe work and buffered rows have explicit limits. No probe row reaches the
consumer before selection. A completed probe's buffer is the entire answer;
an incomplete probe is discarded before restarting the selected plan. This
prevents duplicates or missing rows without requiring output deduplication or
resume-state support. The tradeoff is bounded repeated startup work. There is
no switch after externally emitting rows in this first version.

## 4. Compile inexpensive nested masks

Preserve the current specialization to constant, free, and already-bound terms.
Compile constants once and a short recipe for binding-dependent updates. Each
physical level consumes at most six slot bits, so the update list is bounded
per atom per level.

```text
COMPILE_NESTED(order, schedule):
    previously_bound = {}
    for atom in order:
        plan.new_variables = variables(atom) minus previously_bound
        plan.domain_checks = [(variable, dimension_domain)
                              for already-bound variable occurrences]

        for level in schedule:
            plan.base_mask[level] = valid_slot_mask(level)
            plan.bound_checks[level] = []
            for each consumed coordinate bit in each dimension:
                ones = slots_with_this_coordinate_bit_set(level, dimension, bit)
                if term is Constant(c):
                    intersect base_mask with ones or its complement, from c
                else if term's variable is in previously_bound:
                    append (variable, coordinate_bit, ones) to bound_checks

        previously_bound = previously_bound union variables(atom)
    return compiled atoms in exactly the chosen order

MASKS_FOR_ENTRY(atom_plan, binding):
    if any bound occurrence exceeds that dimension's domain:
        return NO_MATCH
    masks = copy atom_plan.base_mask
    for level:
        for (variable, value_bit, ones) in atom_plan.bound_checks[level]:
            masks[level] &= ones if binding[variable] has value_bit else NOT ones
    return masks
```

Complemented masks remain bounded by `base_mask`. Runtime domain checks matter
when a variable moves between dimensions with different domain sizes. Avoid
rebuilding general restriction automata or doing per-node term dispatch.

## 5. Nested execution

```text
NESTED_VISIT(plan, tree, emit):
    binding = empty binding
    return ENTER_ATOM(0, binding)

ENTER_ATOM(i, binding):
    if i == number_of_atoms:
        return emit(binding)

    atom = plan.atoms[i]
    masks = MASKS_FOR_ENTRY(atom, binding)
    if masks == NO_MATCH:
        return CONTINUE

    return WALK_ATOM(tree.root, level=0, coordinate_prefix=0, masks,
        on_leaf = lambda triple:
            checkpoint = binding.checkpoint()
            bind atom.new_variables from triple
            result = ENTER_ATOM(i + 1, binding)
            binding.restore(checkpoint)
            return result)

WALK_ATOM(address, level, coordinate_prefix, masks, on_leaf):
    word = tree.read_word(address)
    candidates = word AND masks[level]
    for slot in set_bits(candidates):
        child_prefix = advance_coordinate(coordinate_prefix, level, slot)
        if level is the leaf level:
            propagate cancellation from on_leaf(child_prefix)
        else:
            rank = popcount(word AND bits_below(slot))
            child_address = address + tree.read_delta(address, rank)
            propagate cancellation from WALK_ATOM(
                child_address, level + 1, child_prefix, masks, on_leaf)
    return CONTINUE
```

The checkpoint represents scoped ownership of newly introduced bindings; an
implementation can use AutomatonJoin's static first-occurrence assignments to
avoid an actual trail allocation. Stale values must never be considered bound.
Each child address is computed from the unchanged parent address, and rank uses
the **original occupancy word**, not the filtered candidate mask.

## 6. Prefix execution

Reuse the existing compiler's bit ownership, constant masks, equality checks,
and writes. Shared variables can occupy different physical dimensions whose
schedules consume their bits at different times. The compiler must compare
coordinate bit identities, not merely equal slot positions.

```text
PREFIX_VISIT(stages, tree, emit):
    initialize shared variable bits and each atom's root address
    return STEP(level=0, atom=0, current_nodes=roots, next_nodes=empty)

STEP(level, atom, current_nodes, next_nodes):
    if atom == number_of_atoms:
        if level is the leaf level:
            return emit(shared_variable_values)
        return STEP(level + 1, 0, next_nodes, fresh_empty_nodes)

    stage = stages[level, atom]
    address = current_nodes[atom]
    word = tree.read_word(address)
    candidates = word AND stage.constant_and_local_equality_mask
    candidates = filter_by_known_shared_bits(candidates, stage, shared_bits)

    for slot in set_bits(candidates):
        checkpoint = shared_bits.checkpoint()
        write only the variable bits owned by this stage from slot
        if level is not the leaf level:
            rank = popcount(word AND bits_below(slot))
            next_nodes[atom] = address + tree.read_delta(address, rank)
        result = STEP(level, atom + 1, current_nodes, next_nodes)
        shared_bits.restore(checkpoint)
        propagate cancellation from result
    return CONTINUE
```

Every atom has its own physical address. ANDing raw occupancy words from different
atoms would compare unrelated physical slot assignments. The product traversal
shares variable bits while preserving those independent addresses. The pseudocode
uses recursion for clarity; FstJoin's explicit frames implement the same search.

## Expected behavior and limits

| Case | Intended behavior | Reason |
|---|---|---|
| Class + rare edge | Nested, rare edge first | One selective scan followed by cheap fully bound class probes |
| Uniform shared-object star | Usually nested if estimates support it | Both orders have similar fanout; measured nested traversal is cheaper |
| Uniform triangle | Compare estimated work and uncertainty | Being cyclic alone did not make prefix execution faster |
| Empty group-constraint triangle | Prefix pilot can finish immediately | Contradictory shared high bits prune before pairwise enumeration |
| Large uncertain cyclic expansion | Prefer prefix | A nested order may enumerate many partial bindings that never become answers |
| Repeated variable inside an atom | Prefix | Existing nested executor does not enforce this constraint |

Ordering and executor selection never alter the conjunction. Nested traversal
enforces constants and earlier bindings, then enumerates extensions. Prefix
traversal enforces the same constraints incrementally through compiled bit
ownership. Planning probes use these same exact executors. Estimates only choose
which exact search runs.

With beam width B and m atoms, order search considers O(B m²) extensions, excluding
the bounded cost of each estimate. Nested execution uses O(m H + V) traversal
state for tree height H and V variables. The prefix executor uses O(m H + V)
state as well. Planning probes add bounded result buffering. Plan masks and
planning statistics have separately accounted storage costs. Graph-size result
or intermediate tables are unnecessary for either executor.

This does not establish worst-case optimality or a universal speedup. A bad
estimate can still select a poor nested plan, and bounded startup probes cannot
detect every late explosion. Future switching must partition the remaining
search space exactly or preserve resumable state; restarting after emitting
rows would be incorrect without additional duplicate handling.

Implement and measure in this order: explicit atom-order support; the order
estimator; executor selection and private bounded probes; then compiled mask
updates. The investigation already demonstrates the value of the first step.
Validate complete bindings and cancellation behavior alongside planning time,
full-visit and first-result latency, node/candidate counts, metadata bytes, and
the skew/cycle sensitivity sweeps before promoting the complete planner.
