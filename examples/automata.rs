//! Build query automata independently, compose them, then search the FST.
use s_graph::automata::{
    Bgp, Query, Search,
    Term::{Const as C, Var as V},
};
use s_graph::{DEFAULT_BLOCK_CAPACITY, Fst};
use std::ops::ControlFlow;

fn main() -> s_graph::Result<()> {
    let graph = Fst::try_build_dynamic(
        vec![(0, 0, 1), (0, 0, 3), (1, 1, 2)],
        DEFAULT_BLOCK_CAPACITY,
    )?;
    let people = Bgp::new(2).triple([C(0), C(0), V(0)]);
    let detail = Bgp::new(2).triple([V(0), C(1), V(1)]);

    // Conjunction can be fused into one register automaton:
    let required = graph.search(people.clone().and(detail.clone()))?;
    let _: ControlFlow<()> = required.visit(&mut Default::default(), |row| {
        println!("required: {:?}", row.to_options());
        ControlFlow::Continue(())
    });

    // OPTIONAL is a binding operator: the second row keeps its missing slot.
    let machine = people.optional(detail);
    let search = graph.search(&machine)?;
    let mut workspace = Default::default();
    let _: ControlFlow<()> = search.visit(&mut workspace, |row| {
        println!("optional: {:?}", row.to_options());
        ControlFlow::Continue(())
    });
    // ASK/LIMIT cancellation leaves the workspace reusable.
    assert_eq!(
        search.visit(&mut workspace, |_| ControlFlow::Break(true)),
        ControlFlow::Break(true)
    );
    Ok(())
}
