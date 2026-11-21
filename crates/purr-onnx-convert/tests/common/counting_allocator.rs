use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

pub struct CountingAllocator;

#[derive(Debug, Clone, Copy, Default)]
struct Counters {
    current: isize,
    peak: isize,
    largest: usize,
}

// Per thread, so tests running in parallel do not see each other's allocations; const-initialised
// so reaching them never allocates.
thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static COUNTERS: Cell<Counters> = const {
        Cell::new(Counters { current: 0, peak: 0, largest: 0 })
    };
}

fn record(freed: usize, allocated: usize) {
    // a thread's locals are gone while it exits, and nothing is measured then
    let Ok(true) = MEASURING.try_with(Cell::get) else {
        return;
    };
    COUNTERS.with(|cell| {
        let mut counters = cell.get();
        counters.current = counters.current - freed.cast_signed() + allocated.cast_signed();
        counters.peak = counters.peak.max(counters.current);
        counters.largest = counters.largest.max(allocated);
        cell.set(counters);
    });
}

// SAFETY: every call is forwarded unchanged to the system allocator; the counters only observe.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(0, layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(layout.size(), 0);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(layout.size(), new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Usage {
    pub peak_above_start: usize,
    pub largest_allocation: usize,
}

/// Measures the allocations `work` makes on the calling thread.
pub fn measure<T>(work: impl FnOnce() -> T) -> (T, Usage) {
    COUNTERS.with(|cell| cell.set(Counters::default()));
    MEASURING.with(|cell| cell.set(true));
    let result = work();
    MEASURING.with(|cell| cell.set(false));
    let counters = COUNTERS.with(Cell::get);
    let usage = Usage {
        peak_above_start: counters.peak.max(0).cast_unsigned(),
        largest_allocation: counters.largest,
    };
    (result, usage)
}
