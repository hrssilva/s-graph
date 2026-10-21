//! Typed automata with static composition. The bitmap interface is an optional
//! execution specialization, not a restriction on the general event alphabet.
use crate::path_fst::PathFstDeltaPow2Bfs;

/// Supported values of one Boolean projection of a packed transition set.
/// Intersections combine independent constraints on the same variable bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitSupport(u8);
impl BitSupport {
    pub const BOTH: Self = Self(3);
    pub fn from_mask(allowed: u64, ones: u64) -> Self {
        Self(u8::from(allowed & !ones != 0) | (u8::from(allowed & ones != 0) << 1))
    }
    pub fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn singleton(self) -> Option<bool> {
        match self.0 {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        }
    }
}

/// Language semantics over a typed alphabet. Missing transitions reject.
/// Context is read-only; reversible effects belong to the execution layer.
/// can_match may overapproximate, and must include an accepting empty suffix.
pub trait Automaton<Event, Context: ?Sized = ()> {
    type State: Clone;
    fn start(&self) -> Self::State;
    fn step(&self, state: &Self::State, event: &Event, context: &Context) -> Option<Self::State>;
    fn is_accepting(&self, state: &Self::State, context: &Context) -> bool;
    fn can_match(&self, _state: &Self::State, _context: &Context) -> bool {
        true
    }
}
impl<E, C: ?Sized, A: Automaton<E, C> + ?Sized> Automaton<E, C> for &A {
    type State = A::State;
    fn start(&self) -> Self::State {
        (**self).start()
    }
    fn step(&self, s: &Self::State, e: &E, c: &C) -> Option<Self::State> {
        (**self).step(s, e, c)
    }
    fn is_accepting(&self, s: &Self::State, c: &C) -> bool {
        (**self).is_accepting(s, c)
    }
    fn can_match(&self, s: &Self::State, c: &C) -> bool {
        (**self).can_match(s, c)
    }
}
/// An accepting transition can produce a typed contribution. Recognition and
/// output remain separate, so And/Or/Not keep their language semantics.
pub trait Transducer<E, C: ?Sized = ()>: Automaton<E, C> {
    type Output;
    fn output(&self, state: &Self::State, event: &E, context: &C) -> Self::Output;
}
pub struct Project<A, F>(pub A, pub F);
impl<E, C: ?Sized, A: Automaton<E, C>, F> Automaton<E, C> for Project<A, F> {
    type State = A::State;
    fn start(&self) -> Self::State {
        self.0.start()
    }
    fn step(&self, s: &Self::State, e: &E, c: &C) -> Option<Self::State> {
        self.0.step(s, e, c)
    }
    fn is_accepting(&self, s: &Self::State, c: &C) -> bool {
        self.0.is_accepting(s, c)
    }
    fn can_match(&self, s: &Self::State, c: &C) -> bool {
        self.0.can_match(s, c)
    }
}
impl<E, C: ?Sized, A: Automaton<E, C>, F: Fn(&E, &C) -> O, O> Transducer<E, C> for Project<A, F> {
    type Output = O;
    fn output(&self, _: &Self::State, e: &E, c: &C) -> O {
        (self.1)(e, c)
    }
}
/// Reducers own persistent effects; unlike branch-local binding actions these
/// contributions must not be undone when a storage traversal backtracks.
pub trait Reducer<T> {
    fn push(&mut self, value: T);
}
impl<T, F: FnMut(T)> Reducer<T> for F {
    fn push(&mut self, value: T) {
        self(value)
    }
}
#[derive(Clone, Copy)]
pub struct And<A, B>(pub A, pub B);
impl<E, C: ?Sized, A: Automaton<E, C>, B: Automaton<E, C>> Automaton<E, C> for And<A, B> {
    type State = (A::State, B::State);
    fn start(&self) -> Self::State {
        (self.0.start(), self.1.start())
    }
    fn step(&self, s: &Self::State, e: &E, c: &C) -> Option<Self::State> {
        Some((self.0.step(&s.0, e, c)?, self.1.step(&s.1, e, c)?))
    }
    fn is_accepting(&self, s: &Self::State, c: &C) -> bool {
        self.0.is_accepting(&s.0, c) && self.1.is_accepting(&s.1, c)
    }
    fn can_match(&self, s: &Self::State, c: &C) -> bool {
        self.0.can_match(&s.0, c) && self.1.can_match(&s.1, c)
    }
}
#[derive(Clone, Copy)]
pub struct Or<A, B>(pub A, pub B);
impl<E, C: ?Sized, A: Automaton<E, C>, B: Automaton<E, C>> Automaton<E, C> for Or<A, B> {
    type State = (Option<A::State>, Option<B::State>);
    fn start(&self) -> Self::State {
        (Some(self.0.start()), Some(self.1.start()))
    }
    fn step(&self, s: &Self::State, e: &E, c: &C) -> Option<Self::State> {
        let a = s.0.as_ref().and_then(|a| self.0.step(a, e, c));
        let b = s.1.as_ref().and_then(|b| self.1.step(b, e, c));
        if a.is_none() && b.is_none() {
            None
        } else {
            Some((a, b))
        }
    }
    fn is_accepting(&self, s: &Self::State, c: &C) -> bool {
        s.0.as_ref().is_some_and(|a| self.0.is_accepting(a, c))
            || s.1.as_ref().is_some_and(|b| self.1.is_accepting(b, c))
    }
    fn can_match(&self, s: &Self::State, c: &C) -> bool {
        s.0.as_ref().is_some_and(|a| self.0.can_match(a, c))
            || s.1.as_ref().is_some_and(|b| self.1.can_match(b, c))
    }
}
/// Complement completes missing transitions with a live sink. Inverting a
/// prefix mask is NOT a correct implementation of language complement.
#[derive(Clone, Copy)]
pub struct Not<A>(pub A);
impl<E, C: ?Sized, A: Automaton<E, C>> Automaton<E, C> for Not<A> {
    type State = Option<A::State>;
    fn start(&self) -> Self::State {
        Some(self.0.start())
    }
    fn step(&self, s: &Self::State, e: &E, c: &C) -> Option<Self::State> {
        Some(s.as_ref().and_then(|a| self.0.step(a, e, c)))
    }
    fn is_accepting(&self, s: &Self::State, c: &C) -> bool {
        !s.as_ref().is_some_and(|a| self.0.is_accepting(a, c))
    }
}
pub struct MapInput<A, F, Inner> {
    pub inner: A,
    pub map: F,
    pub alphabet: std::marker::PhantomData<fn() -> Inner>,
}
impl<E, I, C: ?Sized, A: Automaton<I, C>, F: Fn(&E) -> I> Automaton<E, C> for MapInput<A, F, I> {
    type State = A::State;
    fn start(&self) -> Self::State {
        self.inner.start()
    }
    fn step(&self, s: &Self::State, e: &E, c: &C) -> Option<Self::State> {
        self.inner.step(s, &(self.map)(e), c)
    }
    fn is_accepting(&self, s: &Self::State, c: &C) -> bool {
        self.inner.is_accepting(s, c)
    }
    fn can_match(&self, s: &Self::State, c: &C) -> bool {
        self.inner.can_match(s, c)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub level: usize,
    pub value: u32,
}
/// Prefix-safe allowed transitions; acceptance still must be checked at the end.
pub trait BitmapAutomaton<C: ?Sized = ()>: Automaton<Slot, C> {
    fn mask(&self, state: &Self::State, level: usize, context: &C) -> u64;
}
impl<C: ?Sized, A: BitmapAutomaton<C>, B: BitmapAutomaton<C>> BitmapAutomaton<C> for And<A, B> {
    #[inline]
    fn mask(&self, s: &Self::State, l: usize, c: &C) -> u64 {
        self.0.mask(&s.0, l, c) & self.1.mask(&s.1, l, c)
    }
}
impl<C: ?Sized, A: BitmapAutomaton<C>, B: BitmapAutomaton<C>> BitmapAutomaton<C> for Or<A, B> {
    fn mask(&self, s: &Self::State, l: usize, c: &C) -> u64 {
        s.0.as_ref().map_or(0, |a| self.0.mask(a, l, c))
            | s.1.as_ref().map_or(0, |b| self.1.mask(b, l, c))
    }
}
#[derive(Clone, Copy)]
pub struct Mask(pub u64);
impl<C: ?Sized> Automaton<Slot, C> for Mask {
    type State = ();
    fn start(&self) {}
    fn step(&self, _: &(), e: &Slot, _: &C) -> Option<()> {
        (e.value < 64 && self.0 & (1u64 << e.value) != 0).then_some(())
    }
    fn is_accepting(&self, _: &(), _: &C) -> bool {
        true
    }
}
impl<C: ?Sized> BitmapAutomaton<C> for Mask {
    #[inline]
    fn mask(&self, _: &(), _: usize, _: &C) -> u64 {
        self.0
    }
}
#[derive(Clone, Copy, Default)]
pub struct BitTest {
    pub variable: usize,
    pub bit: u32,
    pub ones: u64,
}
#[derive(Clone, Copy)]
pub struct BindingGuard {
    pub tests: [BitTest; 6],
    pub count: u8,
}
impl Automaton<Slot, [u32]> for BindingGuard {
    type State = ();
    fn start(&self) {}
    fn step(&self, _: &(), e: &Slot, c: &[u32]) -> Option<()> {
        (e.value < 64 && self.mask(&(), e.level, c) & (1u64 << e.value) != 0).then_some(())
    }
    fn is_accepting(&self, _: &(), _: &[u32]) -> bool {
        true
    }
}
impl BitmapAutomaton<[u32]> for BindingGuard {
    #[inline]
    fn mask(&self, _: &(), _: usize, c: &[u32]) -> u64 {
        let mut mask = u64::MAX;
        for t in &self.tests[..self.count as usize] {
            mask &= if c[t.variable] & t.bit != 0 {
                t.ones
            } else {
                !t.ones
            };
        }
        mask
    }
}
/// Storage automaton interface specialized for packed outgoing transitions.
/// The opened bitmap is passed to follow(), avoiding a second storage read.
/// This fast path is infallible and immutable for a plan's lifetime. A fallible
/// I/O executor must report I/O errors separately from language rejection.
/// Immutable view of a finite bitmap language. `word` returns valid outgoing
/// slots; `follow` is called only for an occupied slot before the last level.
/// Cursors identify physical states, not the logical prefixes that reach them.
/// Keep words, roots and transitions stable for the lifetime of a prepared plan:
/// adaptive planning may certify that a query has no answers.
pub trait BitmapSource {
    type Cursor: Copy + Default;
    fn root(&self) -> Option<Self::Cursor>;
    fn height(&self) -> usize;
    fn word(&self, cursor: Self::Cursor) -> u64;
    fn follow(&self, cursor: Self::Cursor, word: u64, slot: u32) -> Self::Cursor;
}
/// A bitmap source whose slot digits encode S/P/O coordinates. Domains are
/// sizes in 1..=2^32; layouts partition coordinate bits without overlap, with
/// at most six bits per level and at most 64 levels. `layout` returns digit
/// widths, slot shifts and remaining coordinate bit counts respectively.
/// Implementations must uphold this encoding for every reachable state.
pub trait TripleSource: BitmapSource {
    fn domains(&self) -> [u64; 3];
    fn layout(&self, level: usize) -> ([u8; 3], [u8; 3], [u8; 3]);
}
impl<S: BitmapSource + ?Sized> BitmapSource for &S {
    type Cursor = S::Cursor;
    #[inline]
    fn root(&self) -> Option<Self::Cursor> {
        (**self).root()
    }
    #[inline]
    fn height(&self) -> usize {
        (**self).height()
    }
    #[inline]
    fn word(&self, cursor: Self::Cursor) -> u64 {
        (**self).word(cursor)
    }
    #[inline]
    fn follow(&self, cursor: Self::Cursor, word: u64, slot: u32) -> Self::Cursor {
        (**self).follow(cursor, word, slot)
    }
}
impl<S: TripleSource + ?Sized> TripleSource for &S {
    fn domains(&self) -> [u64; 3] {
        (**self).domains()
    }
    fn layout(&self, level: usize) -> ([u8; 3], [u8; 3], [u8; 3]) {
        (**self).layout(level)
    }
}
#[derive(Clone, Copy)]
pub struct FstSource<'a>(pub &'a PathFstDeltaPow2Bfs);
impl BitmapSource for FstSource<'_> {
    type Cursor = u64;
    #[inline]
    fn root(&self) -> Option<u64> {
        (self.0.height() > 0).then(|| self.0.root_addr())
    }
    #[inline]
    fn height(&self) -> usize {
        self.0.height()
    }
    #[inline]
    fn word(&self, c: u64) -> u64 {
        self.0.read_word(c)
    }
    #[inline]
    fn follow(&self, c: u64, w: u64, q: u32) -> u64 {
        c + self
            .0
            .read_delta(c, (w & ((1u64 << q) - 1)).count_ones() as usize)
    }
}
impl TripleSource for FstSource<'_> {
    fn domains(&self) -> [u64; 3] {
        self.0.dim_sizes()
    }
    fn layout(&self, l: usize) -> ([u8; 3], [u8; 3], [u8; 3]) {
        self.0.level_schedule(l)
    }
}
/// Correct generic graph-language adapter. Fast executors fuse its mask/step
/// into one open/follow operation and retain the already-read word.
pub struct GraphAutomaton<S>(pub S);
#[derive(Clone)]
pub struct GraphState<C> {
    pub cursor: Option<C>,
    pub level: usize,
}
impl<S: BitmapSource, C: ?Sized> Automaton<Slot, C> for GraphAutomaton<S> {
    type State = GraphState<S::Cursor>;
    fn start(&self) -> Self::State {
        GraphState {
            cursor: self.0.root(),
            level: 0,
        }
    }
    fn step(&self, s: &Self::State, e: &Slot, _: &C) -> Option<Self::State> {
        if e.level != s.level || s.level >= self.0.height() || e.value >= 64 {
            return None;
        }
        let c = s.cursor?;
        let w = self.0.word(c);
        if w & (1u64 << e.value) == 0 {
            return None;
        }
        Some(GraphState {
            cursor: Some(if s.level + 1 == self.0.height() {
                c
            } else {
                self.0.follow(c, w, e.value)
            }),
            level: s.level + 1,
        })
    }
    fn is_accepting(&self, s: &Self::State, _: &C) -> bool {
        s.cursor.is_some() && s.level == self.0.height()
    }
    fn can_match(&self, s: &Self::State, _: &C) -> bool {
        s.cursor.is_some()
    }
}
impl<S: BitmapSource, C: ?Sized> BitmapAutomaton<C> for GraphAutomaton<S> {
    fn mask(&self, s: &Self::State, l: usize, _: &C) -> u64 {
        if s.level != l || l >= self.0.height() {
            0
        } else {
            s.cursor.map_or(0, |c| self.0.word(c))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Word(&'static [u8]);
    impl Automaton<u8> for Word {
        type State = usize;
        fn start(&self) -> usize {
            0
        }
        fn step(&self, s: &usize, e: &u8, _: &()) -> Option<usize> {
            self.0.get(*s).filter(|&&v| v == *e).map(|_| s + 1)
        }
        fn is_accepting(&self, s: &usize, _: &()) -> bool {
            *s == self.0.len()
        }
    }
    fn accepts<A: Automaton<u8>>(a: &A, word: &[u8]) -> bool {
        let mut s = a.start();
        for e in word {
            let Some(next) = a.step(&s, e, &()) else {
                return false;
            };
            s = next;
        }
        a.is_accepting(&s, &())
    }
    #[test]
    fn language_algebra_and_typed_mapping() {
        let a = Or(Word(b"ab"), Word(b"ac"));
        let not = Not(Word(b"ab"));
        let both = And(a, Not(Word(b"ac")));
        for w in [b"".as_slice(), b"a", b"ab", b"ac", b"ad", b"aba"] {
            assert_eq!(accepts(&not, w), w != b"ab");
            assert_eq!(accepts(&both, w), w == b"ab");
        }
        let mapped = MapInput {
            inner: Word(b"ab"),
            map: |e: &u8| e.to_ascii_lowercase(),
            alphabet: std::marker::PhantomData,
        };
        assert!(accepts(&mapped, b"AB"));
        assert!(!accepts(&mapped, b"ABC"));
    }
    #[test]
    fn reversible_actions_restore_in_reverse_order() {
        struct Add;
        struct Scale;
        impl ReversibleAction<u32, u32> for Add {
            type Undo = u32;
            fn apply(&self, c: &mut u32, e: &u32) -> u32 {
                let old = *c;
                *c += e;
                old
            }
            fn undo(&self, c: &mut u32, old: u32) {
                *c = old;
            }
        }
        impl ReversibleAction<u32, u32> for Scale {
            type Undo = u32;
            fn apply(&self, c: &mut u32, e: &u32) -> u32 {
                let old = *c;
                *c *= e;
                old
            }
            fn undo(&self, c: &mut u32, old: u32) {
                *c = old;
            }
        }
        let a = Actions(Add, Scale);
        let mut value = 3;
        let undo = a.apply(&mut value, &3);
        assert_eq!(value, 18);
        a.undo(&mut value, undo);
        assert_eq!(value, 3);
        let guard = BindingGuard {
            tests: [BitTest {
                variable: 0,
                bit: 1,
                ones: 2,
            }; 6],
            count: 1,
        };
        let graph = And(GraphAutomaton(Binary), guard);
        assert_eq!(graph.mask(&graph.start(), 0, &[1u32][..]), 2);
    }
    #[derive(Clone, Copy)]
    struct Binary;
    impl BitmapSource for Binary {
        type Cursor = usize;
        fn root(&self) -> Option<usize> {
            Some(0)
        }
        fn height(&self) -> usize {
            2
        }
        fn word(&self, c: usize) -> u64 {
            [3, 2, 1][c]
        }
        fn follow(&self, _: usize, _: u64, q: u32) -> usize {
            1 + q as usize
        }
    }
    #[test]
    fn generic_product_checks_final_acceptance_and_complement() {
        let mapped = || MapInput {
            inner: Word(&[1, 0]),
            map: |e: &Slot| e.value as u8,
            alphabet: std::marker::PhantomData,
        };
        let mut rows = Vec::new();
        let _: std::ops::ControlFlow<()> =
            visit_language(&Binary, &Unfiltered(mapped()), &(), |p| {
                rows.push(p.iter().map(|e| e.value).collect::<Vec<_>>());
                std::ops::ControlFlow::Continue(())
            });
        assert_eq!(rows, vec![vec![1, 0]]);
        rows.clear();
        let _: std::ops::ControlFlow<()> =
            visit_language(&Binary, &Unfiltered(Not(mapped())), &(), |p| {
                rows.push(p.iter().map(|e| e.value).collect::<Vec<_>>());
                std::ops::ControlFlow::Continue(())
            });
        assert_eq!(rows, vec![vec![0, 1]]);
        assert_eq!(
            visit_language(&Binary, &Mask(3), &(), |_| std::ops::ControlFlow::Break(8)),
            std::ops::ControlFlow::Break(8)
        );
    }
    #[test]
    fn graph_acceptance_union_and_mask_step_agree() {
        let a = And(GraphAutomaton(Binary), Mask(3));
        let s = <And<GraphAutomaton<Binary>, Mask> as Automaton<Slot>>::start(&a);
        assert!(!a.is_accepting(&s, &()));
        let s = a.step(&s, &Slot { level: 0, value: 0 }, &()).unwrap();
        assert!(!a.is_accepting(&s, &()));
        assert!(a.step(&s, &Slot { level: 1, value: 0 }, &()).is_none());
        let s = a.step(&s, &Slot { level: 1, value: 1 }, &()).unwrap();
        assert!(a.is_accepting(&s, &()));
        assert!(a.step(&s, &Slot { level: 2, value: 0 }, &()).is_none());
        let union = Or(Mask(5), Mask(10));
        let s = <Or<Mask, Mask> as Automaton<Slot>>::start(&union);
        for q in 0..64 {
            assert_eq!(
                union.mask(&s, 0, &()) & (1u64 << q) != 0,
                union.step(&s, &Slot { level: 0, value: q }, &()).is_some()
            );
        }
        let left = union.step(&s, &Slot { level: 0, value: 0 }, &()).unwrap();
        assert!(left.0.is_some() && left.1.is_none());
        assert_eq!(union.mask(&left, 1, &()), 5);
    }
}

/// Effects are distinct from language recognition. A backtracking executor can
/// compose them statically and restore state in reverse order without cloning
/// the full register set. Implementations specify their own undo payload.
pub trait ReversibleAction<Event, Context: ?Sized> {
    type Undo;
    fn apply(&self, context: &mut Context, event: &Event) -> Self::Undo;
    fn undo(&self, context: &mut Context, undo: Self::Undo);
}
pub struct Actions<A, B>(pub A, pub B);
impl<E, C: ?Sized, A: ReversibleAction<E, C>, B: ReversibleAction<E, C>> ReversibleAction<E, C>
    for Actions<A, B>
{
    type Undo = (A::Undo, B::Undo);
    #[inline]
    fn apply(&self, c: &mut C, e: &E) -> Self::Undo {
        (self.0.apply(c, e), self.1.apply(c, e))
    }
    #[inline]
    fn undo(&self, c: &mut C, u: Self::Undo) {
        self.1.undo(c, u.1);
        self.0.undo(c, u.0);
    }
}

/// Correctness-preserving adapter for ordinary automata without a mask oracle.
/// It leaves prefix pruning to step()/can_match(); complement uses this path.
#[derive(Clone, Copy)]
pub struct Unfiltered<A>(pub A);
impl<C: ?Sized, A: Automaton<Slot, C>> Automaton<Slot, C> for Unfiltered<A> {
    type State = A::State;
    fn start(&self) -> Self::State {
        self.0.start()
    }
    fn step(&self, s: &Self::State, e: &Slot, c: &C) -> Option<Self::State> {
        self.0.step(s, e, c)
    }
    fn is_accepting(&self, s: &Self::State, c: &C) -> bool {
        self.0.is_accepting(s, c)
    }
    fn can_match(&self, s: &Self::State, c: &C) -> bool {
        self.0.can_match(s, c)
    }
}
impl<C: ?Sized, A: Automaton<Slot, C>> BitmapAutomaton<C> for Unfiltered<A> {
    fn mask(&self, _: &Self::State, _: usize, _: &C) -> u64 {
        u64::MAX
    }
}
/// Generic finite-source product runner, including the final transition and
/// explicit acceptance. Join execution specializes this contract into compiled
/// register effects and macro transitions, avoiding state cloning/dispatch.
pub fn visit_language<S: BitmapSource, C: ?Sized, A: BitmapAutomaton<C>, T>(
    source: &S,
    machine: &A,
    context: &C,
    emit: impl FnMut(&[Slot]) -> std::ops::ControlFlow<T>,
) -> std::ops::ControlFlow<T> {
    visit_language_with_workspace(source, machine, context, &mut Vec::new(), emit)
}
/// Reusable path scratch for repeated finite-source product traversals.
pub fn visit_language_with_workspace<S: BitmapSource, C: ?Sized, A: BitmapAutomaton<C>, T>(
    source: &S,
    machine: &A,
    context: &C,
    path: &mut Vec<Slot>,
    mut emit: impl FnMut(&[Slot]) -> std::ops::ControlFlow<T>,
) -> std::ops::ControlFlow<T> {
    use std::ops::ControlFlow;
    fn walk<S: BitmapSource, C: ?Sized, A: BitmapAutomaton<C>, T>(
        source: &S,
        machine: &A,
        c: &C,
        cursor: S::Cursor,
        level: usize,
        state: A::State,
        path: &mut Vec<Slot>,
        emit: &mut impl FnMut(&[Slot]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        if level == source.height() {
            if machine.is_accepting(&state, c) {
                emit(path)?;
            }
            return ControlFlow::Continue(());
        }
        if !machine.can_match(&state, c) {
            return ControlFlow::Continue(());
        }
        let allowed = machine.mask(&state, level, c);
        if allowed == 0 {
            return ControlFlow::Continue(());
        }
        let word = source.word(cursor);
        let mut pending = word & allowed;
        while pending != 0 {
            let value = pending.trailing_zeros();
            pending &= pending - 1;
            let e = Slot { level, value };
            let Some(next) = machine.step(&state, &e, c) else {
                continue;
            };
            path.push(e);
            let child = if level + 1 == source.height() {
                cursor
            } else {
                source.follow(cursor, word, value)
            };
            let result = walk(source, machine, c, child, level + 1, next, path, emit);
            path.pop();
            result?;
        }
        ControlFlow::Continue(())
    }
    path.clear();
    let Some(root) = source.root() else {
        return ControlFlow::Continue(());
    };
    walk(
        source,
        machine,
        context,
        root,
        0,
        machine.start(),
        path,
        &mut emit,
    )
}
