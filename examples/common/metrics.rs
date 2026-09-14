//! Memory-footprint and timing measurement helpers shared by every demo and
//! by the `compare` benchmark harness.
//!
//! Three distinct notions of "size" are reported throughout this
//! comparison:
//! - [`MemSize::mem_size_bytes`]: the actual number of bytes each structure
//!   has allocated (`Vec::capacity() * size_of::<T>()`, summed over every
//!   backing array), not just the logical bit/byte count -- this is what
//!   actually shows up in the allocator and in RSS.
//! - [`peak_rss_kb`]: the OS-reported *peak* resident set size for the
//!   whole process (`VmHWM` in `/proc/self/status`, Linux-only). This
//!   captures allocator overhead, fragmentation, and transient
//!   construction-time peaks that `mem_size_bytes` cannot see -- but it's
//!   a permanent high-water mark that never decreases, so it also
//!   captures build-time spikes (raw input, counting-sort scratch
//!   buffers) that no longer reflect memory actually in use once those
//!   buffers are freed.
//! - [`current_rss_kb`]: the resident set size *right now* (`VmRSS`),
//!   which does go back down once large allocations are freed and their
//!   pages returned to the OS. Reading this immediately after a
//!   structure finishes building (with anything it no longer needs, like
//!   the raw input dataset, already dropped) gives the "steady-state"
//!   footprint a long-running process would actually see post-build --
//!   the number `peak_rss_kb` alone cannot show, since it's permanently
//!   pinned to whatever the build-time transient spike was.

use std::time::{Duration, Instant};

pub trait MemSize {
    fn mem_size_bytes(&self) -> usize;
}

pub fn vec_bytes<T>(v: &Vec<T>) -> usize {
    v.capacity() * std::mem::size_of::<T>()
}

fn status_field_kb(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix(field))
        .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse().ok())
}

/// Peak resident set size in KiB, as reported by the kernel (`VmHWM` in
/// `/proc/self/status`). Returns `None` on non-Linux platforms or if the
/// file can't be read/parsed. A permanent high-water mark for the whole
/// process -- see [`current_rss_kb`] for the "right now" counterpart,
/// which is what actually reflects memory freed since any earlier spike.
pub fn peak_rss_kb() -> Option<u64> {
    status_field_kb("VmHWM:")
}

/// Resident set size *right now*, in KiB (`VmRSS` in `/proc/self/status`).
/// Unlike [`peak_rss_kb`], this goes back down once large allocations are
/// freed and the OS reclaims their pages -- read it after a structure
/// finishes building (with anything no longer needed, like the raw input
/// dataset, already dropped) to see that structure's actual steady-state
/// footprint, not a build-time transient peak. Returns `None` on
/// non-Linux platforms or if the file can't be read/parsed.
pub fn current_rss_kb() -> Option<u64> {
    status_field_kb("VmRSS:")
}

/// Human-readable byte count (KiB/MiB/GiB) for report tables.
pub fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Runs `f` `iters` times, returning the median wall-clock duration of a
/// single call. `iters` should be odd for a well-defined median; even
/// counts fall back to the lower-middle element.
pub fn time_it<F: FnMut()>(iters: usize, mut f: F) -> Duration {
    assert!(iters > 0);
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    samples.sort();
    samples[samples.len() / 2]
}
