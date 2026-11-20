use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct CountingAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

fn grew(size: usize) {
    let now = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(now, Ordering::Relaxed);
    LARGEST.fetch_max(size, Ordering::Relaxed);
}

// SAFETY: every call is forwarded unchanged to the system allocator; the counters only observe.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        grew(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Usage {
    pub peak_above_start: usize,
    pub largest_allocation: usize,
}

pub fn measure<T>(work: impl FnOnce() -> T) -> (T, Usage) {
    let start = CURRENT.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    LARGEST.store(0, Ordering::Relaxed);
    let result = work();
    let usage = Usage {
        peak_above_start: PEAK.load(Ordering::Relaxed).saturating_sub(start),
        largest_allocation: LARGEST.load(Ordering::Relaxed),
    };
    (result, usage)
}
