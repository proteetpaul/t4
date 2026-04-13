extern crate io_uring;

use core::slice;
use std::{
    cmp::min,
    io,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use io_uring::IoUring;

use crate::memory::{
    arena::Arena,
    segment::Segment,
    tcache::{TCache, TCacheStats},
};

static FIXED_BUFFER_POOL: OnceLock<FixedBufferPool> = OnceLock::new();

pub const FIXED_BUFFER_SIZE_BYTES: usize = 1 << 20;
pub const FIXED_BUFFER_BITS: u32 = FIXED_BUFFER_SIZE_BYTES.trailing_zeros();

#[derive(Debug)]
pub struct FixedBuffer {
    pub ptr: *mut u8,
    pub buf_id: usize,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct FixedBufferAllocation {
    pub ptr: *mut u8,
    pub size: usize,
}

unsafe impl Send for FixedBufferAllocation {}

impl AsRef<[u8]> for FixedBufferAllocation {
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }
}

impl Drop for FixedBufferAllocation {
    fn drop(&mut self) {
        FixedBufferPool::free(self.ptr);
    }
}

pub struct FixedBufferPool {
    local_caches: Vec<Mutex<TCache>>,
    arena: Arc<Mutex<Arena>>,
    start_ptr: *mut u8,
    capacity: usize,
    registered: AtomicBool,
    foreign_free: AtomicU64,
}

unsafe impl Send for FixedBufferPool {}

unsafe impl Sync for FixedBufferPool {}

impl FixedBufferPool {
    fn new(capacity_mb: usize) -> FixedBufferPool {
        log::info!(
            "Initializing fixed buffer pool with capacity: {} MB",
            capacity_mb
        );
        let num_cpus = std::thread::available_parallelism().unwrap();
        let capacity = capacity_mb << 20;
        let arena = Self::allocate_arena(capacity);
        let start_ptr = {
            let guard = arena.try_lock().unwrap();
            guard.start_ptr()
        };
        let mut local_caches = Vec::<Mutex<TCache>>::new();
        for i in 0..num_cpus.get() {
            local_caches.push(Mutex::new(TCache::new(arena.clone(), i)));
        }
        FixedBufferPool {
            local_caches,
            arena,
            start_ptr,
            capacity,
            registered: AtomicBool::new(false),
            foreign_free: AtomicU64::new(0),
        }
    }

    pub fn allocate_arena(capacity: usize) -> Arc<Mutex<Arena>> {
        Arc::new(Mutex::new(Arena::new(capacity)))
    }

    pub fn init(capacity_mb: usize) {
        FIXED_BUFFER_POOL.get_or_init(|| FixedBufferPool::new(capacity_mb));
    }

    #[inline]
    pub fn is_initialized() -> bool {
        FIXED_BUFFER_POOL.get().is_some()
    }

    fn get_thread_local_cache() -> &'static Mutex<TCache> {
        let cpu = unsafe { libc::sched_getcpu() as usize };
        let pool = FIXED_BUFFER_POOL.get().expect("fixed buffer pool not initialized");
        let idx = cpu % pool.local_caches.len();
        &pool.local_caches[idx]
    }

    pub fn malloc(size: usize) -> *mut u8 {
        let cpu = unsafe { libc::sched_getcpu() };
        let local_cache = Self::get_thread_local_cache();
        let ptr = local_cache.lock().unwrap().allocate(size);
        log::debug!("Allocated pointer: {:?}, size: {}, cpu: {}", ptr, size, cpu);
        if ptr.is_null() {
            log::info!("Unsuccessful allocation of {} bytes", size);
        }
        ptr
    }

    pub fn register_buffers_with_ring(ring: &IoUring) -> io::Result<()> {
        let Some(pool) = FIXED_BUFFER_POOL.get() else {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "fixed buffer pool not initialized",
            ));
        };
        let mut arena_guard = pool.arena.lock().unwrap();
        let res = arena_guard.register_buffers_with_ring(ring);
        if res.is_ok() {
            log::info!("Registered buffers with io-uring ring");
            pool.registered.store(true, Ordering::Relaxed);
        } else {
            log::warn!("register_buffers failed: {:?}", res.as_ref().err());
        }
        res
    }

    pub(crate) fn get_stats(cpu: usize) -> TCacheStats {
        let Some(pool) = FIXED_BUFFER_POOL.get() else {
            return TCacheStats::new();
        };
        let idx = cpu % pool.local_caches.len();
        let tcache = pool.local_caches[idx].lock().unwrap();
        tcache.get_stats()
    }

    pub fn get_fixed_buffers(alloc: &FixedBufferAllocation) -> Vec<FixedBuffer> {
        let ptr = alloc.ptr;
        let size = alloc.size;
        let pool = FIXED_BUFFER_POOL.get().expect("fixed buffer pool not initialized");
        debug_assert!(
            ptr >= pool.start_ptr && ptr < pool.start_ptr.wrapping_add(pool.capacity),
            "Pointer doesn't lie within the arena"
        );
        let mut remaining = size;
        let mut vec = Vec::<FixedBuffer>::new();
        let mut current = ptr;
        let mut buffer_id =
            (current.wrapping_sub(pool.start_ptr as usize) as usize) >> FIXED_BUFFER_BITS;
        while remaining > 0 {
            let next_buffer_start = pool
                .start_ptr
                .wrapping_add((buffer_id + 1) << FIXED_BUFFER_BITS);
            let bytes = min(remaining, next_buffer_start as usize - current as usize);
            let fb = FixedBuffer {
                ptr: current,
                buf_id: buffer_id,
                bytes: bytes,
            };
            current = next_buffer_start;
            vec.push(fb);
            remaining -= bytes;
            buffer_id += 1;
        }
        vec
    }

    #[inline]
    pub fn buffers_registered() -> bool {
        FIXED_BUFFER_POOL
            .get()
            .map(|p| p.registered.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    fn free(ptr: *mut u8) {
        let Some(pool) = FIXED_BUFFER_POOL.get() else {
            return;
        };
        let segment_ptr = Segment::get_segment_from_ptr(ptr);
        let page_ptr = unsafe { (*segment_ptr).get_page_from_ptr(ptr) };
        let thread_id = unsafe { (*segment_ptr).thread_id };
        log::debug!(
            "Freed pointer: {:?}, size: {}, owner thread id: {}",
            ptr,
            unsafe { (*page_ptr).block_size },
            thread_id
        );

        let cur_cpu = unsafe { libc::sched_getcpu() as usize };
        if cur_cpu == thread_id {
            unsafe {
                (*page_ptr).free(ptr);
            }
            let should_free_page = unsafe { (*page_ptr).is_unused() };
            if should_free_page {
                let local_cache = Self::get_thread_local_cache();
                let mut guard = local_cache.lock().unwrap();
                guard.retire_page(page_ptr);
            }
        } else {
            unsafe {
                (*page_ptr).foreign_free(ptr);
            }
            pool.foreign_free.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn print_stats() {
        if FIXED_BUFFER_POOL.get().is_none() {
            return;
        }
        let num_cpus = std::thread::available_parallelism().unwrap();
        let mut agg_stats = TCacheStats::new();
        for i in 0..num_cpus.get() {
            let stats = Self::get_stats(i);
            agg_stats.allocations_from_arena += stats.allocations_from_arena;
            agg_stats.allocations_from_pages += stats.allocations_from_pages;
            agg_stats.allocations_from_segment += stats.allocations_from_segment;
            agg_stats.fast_allocations += stats.fast_allocations;
            agg_stats.pages_retired += stats.pages_retired;
            agg_stats.segments_retired += stats.segments_retired;
            agg_stats.total_segments_allocated += stats.total_segments_allocated;
            agg_stats.unsuccessful_allocations += stats.unsuccessful_allocations;
            agg_stats.total_allocations += stats.total_allocations;
        }
        agg_stats.print();
    }
}

impl Drop for FixedBufferPool {
    fn drop(self: &mut Self) {
        let arena = self.arena.lock().unwrap();
        drop(arena);
    }
}
