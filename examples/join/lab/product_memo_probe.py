from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
lab=Path(subprocess.check_output(['python3',str(root/'examples/join/lab/prepare.py')],text=True).strip())
s=(lab/'src/automaton_join.rs').read_text().replace('AutomatonJoin','MemoJoin')
s=s.replace('    pub candidates: u64,','    pub candidates: u64,\n    pub cache_hits: u64,\n    pub cache_entries: usize,')
start=s.index('    pub fn visit<T>')
end=s.index('/// Near-verbatim copy')
s=s[:start]+'''    pub fn visit<T>(&self, bindings: &mut Vec<u32>, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        self.run::<false, T>(bindings, &mut MemoJoinStats::default(), &mut emit)
    }
    pub fn visit_with_stats<T>(&self, bindings: &mut Vec<u32>, stats: &mut MemoJoinStats, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        self.run::<true, T>(bindings, stats, &mut emit)
    }
    fn run<const TRACK: bool, T>(&self, bindings: &mut Vec<u32>, stats: &mut MemoJoinStats, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        *stats = MemoJoinStats::default();
        if self.empty { return ControlFlow::Continue(()); }
        bindings.resize(self.variables, 0);
        bindings.fill(0);
        if self.atoms.is_empty() { stats.results += 1; return emit(bindings); }
        let mut dead = std::collections::HashSet::new();
        let result = self.atom::<TRACK, T>(0, bindings, &mut dead, stats, &mut emit);
        stats.cache_entries = dead.len();
        result
    }
    fn atom<const TRACK: bool, T>(&self, atom_idx: usize, bindings: &mut Vec<u32>, dead: &mut std::collections::HashSet<(usize,u64,usize,[u32;3])>, stats: &mut MemoJoinStats, emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        if TRACK { stats.atom_entries += 1; }
        let plan=&self.atoms[atom_idx];
        let mut restrictions=[DimRestriction::Free;3];
        let mut exact=[0u32;3];
        for d in 0..3 {
            restrictions[d]=match plan.terms[d] {
                Term::Const(c)=>DimRestriction::Exact(c),
                Term::Var(v) if !plan.first_occurrence[d]=>DimRestriction::Exact(bindings[v]),
                _=>DimRestriction::Free,
            };
            if let DimRestriction::Exact(c)=restrictions[d] {
                if c as u64 >= self.tree.dim_sizes()[d] { return ControlFlow::Continue(()); }
                exact[d]=c;
            }
        }
        let masks=compute_atom_masks(self.tree,&restrictions);
        self.node::<TRACK,T>(atom_idx,self.tree.root_addr(),[0;3],0,&masks,exact,bindings,dead,stats,emit)?;
        ControlFlow::Continue(())
    }
    #[allow(clippy::too_many_arguments)]
    fn node<const TRACK: bool,T>(&self, atom_idx:usize,addr:u64,lo:[u64;3],level:usize,masks:&[u64],exact:[u32;3],bindings:&mut Vec<u32>,dead:&mut std::collections::HashSet<(usize,u64,usize,[u32;3])>,stats:&mut MemoJoinStats,emit:&mut impl FnMut(&[u32])->ControlFlow<T>)->ControlFlow<T,bool> {
        let (bits,shifts,remaining)=self.tree.level_schedule(level);
        let suffix=std::array::from_fn(|d| (exact[d] as u64 & ((1u64 << (bits[d]+remaining[d]))-1)) as u32);
        let key=(atom_idx,addr,level,suffix);
        if dead.contains(&key) { if TRACK { stats.cache_hits+=1; } return ControlFlow::Continue(false); }
        if TRACK { stats.node_visits+=1; }
        let word=self.tree.read_word(addr);
        let mut relevant=word & masks[level];
        if TRACK { stats.candidates+=relevant.count_ones() as u64; }
        let mut any=false;
        while relevant!=0 {
            let qi=relevant.trailing_zeros(); relevant&=relevant-1;
            let child_lo=advance_lo(lo,bits,shifts,qi as usize,remaining);
            if level+1==self.tree.height() {
                any=true; // This atom has a leaf, regardless of the continuation's result.
                let plan=&self.atoms[atom_idx];
                for d in 0..3 { if plan.first_occurrence[d] { if let Term::Var(v)=plan.terms[d] { bindings[v]=child_lo[d] as u32; } } }
                if atom_idx+1==self.atoms.len() { if TRACK {stats.results+=1;} emit(bindings)?; }
                else { self.atom::<TRACK,T>(atom_idx+1,bindings,dead,stats,emit)?; }
            } else {
                let rank=(word & ((1u64<<qi)-1)).count_ones() as usize;
                let child=addr+self.tree.read_delta(addr,rank);
                any |= self.node::<TRACK,T>(atom_idx,child,child_lo,level+1,masks,exact,bindings,dead,stats,emit)?;
            }
        }
        if !any && dead.len()<65_536 { dead.insert(key); }
        ControlFlow::Continue(any)
    }
}

'''+s[end:]
# The old standalone walker is unused by this prototype.
(lab/'src/memo_join.rs').write_text(s)
with (lab/'src/lib.rs').open('a') as f: f.write('\n#[cfg(feature="heap")]\npub mod memo_join;\n')
s=(lab/'investigate.rs').read_text()
marker='    for (mode, query) in ['
pos=s.index(marker)
s=s[:pos]+'''    if engine("memo") {
        let p=s_graph::memo_join::MemoJoin::new(&q,&fst).unwrap();
        let mut ws=Vec::new(); let mut stats=s_graph::memo_join::MemoJoinStats::default();
        let mut actual=Vec::new(); let _:ControlFlow<()>=p.visit_with_stats(&mut ws,&mut stats,|r|{actual.push(r.to_vec());ControlFlow::Continue(())});
        actual.sort_unstable();assert_eq!(actual,expected);
        let t=measure(||{let _:ControlFlow<()>=p.visit(&mut ws,|r|{black_box(r);ControlFlow::Continue(())});},reps);
        print(name,"memo",t,expected.len(),stats.node_visits,stats.candidates,stats.atom_entries);
        eprintln!("memo: cache_hits={} entries={} key_bytes={}",stats.cache_hits,stats.cache_entries,std::mem::size_of::<(usize,u64,usize,[u32;3])>());
    }
'''+s[pos:]
(lab/'investigate.rs').write_text(s)
print(lab)
