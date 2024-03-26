use cfg_if::cfg_if;
use core::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
    mem::transmute,
    ptr::{drop_in_place, null_mut},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
};
use crossbeam_utils::CachePadded;
use likely_stable::unlikely;

cfg_if! {
    if #[cfg(not(feature = "no_std"))] {
        use std::thread_local;
        use std::alloc::{alloc, dealloc};
    } else {
        use alloc::{
            alloc::{alloc, dealloc},
            vec::Vec,
        };
    }
}

use self::conc_linked_list::ConcLinkedList;

#[cfg(target_pointer_width = "64")]
type UFourth = u16;
#[cfg(target_pointer_width = "32")]
type UFourth = u8;

static GLOBAL_INFO: GlobalInfo = GlobalInfo {
    pile: ConcLinkedList::new(),
    locals_pile: ConcLinkedList::new(),
    epoch: AtomicUsize::new(1),
    update_epoch: CachePadded::new(AtomicUsize::new(0)),
};

struct GlobalInfo {
    pile: ConcLinkedList<Instance>,
    locals_pile: ConcLinkedList<*const Inner>,
    epoch: AtomicUsize,
    update_epoch: CachePadded<AtomicUsize>,
}

#[thread_local]
static LOCAL_INFO: SyncUnsafeCell<*const Inner> = SyncUnsafeCell::new(null_mut());

thread_local! {
    static LOCAL_GUARD: LocalGuard = const { LocalGuard };
}

struct Inner {
    pile: SyncUnsafeCell<Vec<Instance>>,
    epoch: Cell<usize>,
    active_local: AtomicUsize,
    active_shared: AtomicUsize,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

struct LocalGuard;

impl Drop for LocalGuard {
    fn drop(&mut self) {
        if let Some(local) = try_get_local() {
            let glob = get_epoch_info();
            let glob_epoch = glob.epoch;
            let glob_threads = glob.threads;
            for garbage in unsafe { &*local.pile.get() } {
                if glob_epoch >= (garbage.epoch + glob_threads) {
                    // release garbage as nobody looks at it anymore
                    unsafe {
                        garbage.cleanup();
                    }
                } else {
                    // push garbage onto pile
                    GLOBAL_INFO.pile.push_front(garbage.clone());
                }
            }
            if local.active_shared.load(Ordering::Acquire)
                == local.active_local.load(Ordering::Acquire)
            {
                // there are no more external references, so local can be cleaned up immediately
                cleanup_local(local);
            } else {
                // there are still external references alive, so delay cleanup until they die
                GLOBAL_INFO.locals_pile.push_front(local as *const Inner);
            }
        }
    }
}

#[derive(Clone)]
struct Instance {
    drop_fn: unsafe fn(*mut ()),
    data_ptr: *const (),
    epoch: usize,
}

impl Instance {
    fn new<T>(val: *const T, epoch: usize) -> Self {
        let drop_fn = core::ptr::drop_in_place::<T>;
        Self::new_explicit(val, epoch, drop_fn)
    }

    fn new_explicit<T>(val: *const T, epoch: usize, drop_fn: unsafe fn(*mut T)) -> Self {
        Self {
            drop_fn: unsafe { core::mem::transmute::<_, fn(*mut ())>(drop_fn) },
            data_ptr: val.cast::<()>(),
            epoch,
        }
    }

    #[inline]
    unsafe fn cleanup(&self) {
        let drop_fn = self.drop_fn;
        drop_fn(self.data_ptr.cast_mut());
    }
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

#[inline]
fn local_ptr() -> *const Inner {
    let src = LOCAL_INFO.get();
    unsafe { *src }
}

#[inline]
fn try_get_local<'a>() -> Option<&'a Inner> {
    unsafe { local_ptr().as_ref() }
}

#[inline]
fn get_local<'a>() -> &'a Inner {
    #[cold]
    #[inline(never)]
    fn alloc_local<'a>() -> &'a Inner {
        let src = LOCAL_INFO.get();
        unsafe {
            let alloc = alloc(Layout::new::<Inner>()).cast::<Inner>();
            if alloc.is_null() {
                // allocation failed, just exit
                abort();
            }
            alloc.write(Inner {
                pile: SyncUnsafeCell::new(Vec::new()),
                epoch: Cell::new(get_epoch_info().epoch),
                active_local: AtomicUsize::new(0),
                active_shared: AtomicUsize::new(0),
            });
            *src = alloc;
            increment_threads();
            LOCAL_GUARD.with(|_| {});
            &*alloc
        }
    }

    if let Some(local) = try_get_local() {
        return local;
    }

    alloc_local()
}

/// This may not be called if there is already a pin active
#[inline(never)]
#[no_mangle]
pub(crate) fn pin() -> LocalPinGuard {
    let local_info = get_local();
    // this is relaxed as we know that we are the only thread currently to modify any local data
    let active = local_info.active_local.load(Ordering::Relaxed);
    local_info.active_local.store(active + 1, Ordering::Release);
    if active != 0 {
        return LocalPinGuard(local_info as *const Inner);
    }
    let update_epoch = GLOBAL_INFO.update_epoch.load(Ordering::Acquire);
    let local = local_info.epoch.get();
    if update_epoch > local {
        update_local(local_info);
    }
    LocalPinGuard(local_info as *const Inner)
}

const MAX_THREADS: usize = UFourth::MAX as usize;
const THREADS_MASK: usize = MAX_THREADS;

const EPOCH_MASK: usize = usize::MAX & !THREADS_MASK;

// these two get combined into a single AtomicUsize in order to avoid race conditions
struct EpochInfo {
    epoch: usize,
    threads: usize,
}

fn increment_epoch() -> EpochInfo {
    #[cold]
    #[inline(never)]
    fn handle_guard_hit(epoch: &mut usize) {
        if *epoch >= EPOCH_OVERFLOW_GUARD {
            abort();
        }
        if *epoch >= EPOCH_WRAP_AROUND_GUARD {
            let mut curr = *epoch;
            loop {
                match GLOBAL_INFO.epoch.compare_exchange(
                    curr,
                    curr - EPOCH_WRAP_AROUND_GUARD,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(new) => {
                        if new < EPOCH_WRAP_AROUND_GUARD {
                            break;
                        }
                        curr = new;
                    }
                }
            }
            *epoch -= EPOCH_WRAP_AROUND_GUARD;
        }
    }

    const EPOCH_ONE: usize = MAX_THREADS + 1;
    const EPOCH_WRAP_AROUND_GUARD: usize = (usize::MAX >> 2) & !THREADS_MASK;
    const EPOCH_OVERFLOW_GUARD: usize = (usize::MAX >> 1) & !THREADS_MASK;

    let mut new = GLOBAL_INFO.epoch.fetch_add(EPOCH_ONE, Ordering::AcqRel) + EPOCH_ONE;

    if unlikely(new >= EPOCH_WRAP_AROUND_GUARD) {
        handle_guard_hit(&mut new);
    }

    EpochInfo {
        epoch: new & EPOCH_MASK,
        threads: new & THREADS_MASK,
    }
}

fn get_epoch_info() -> EpochInfo {
    let curr = GLOBAL_INFO.epoch.load(Ordering::Acquire);
    EpochInfo {
        epoch: curr & EPOCH_MASK,
        threads: curr & THREADS_MASK,
    }
}

fn decrement_threads() {
    const THREAD_ONE: usize = 1;

    GLOBAL_INFO.epoch.fetch_sub(THREAD_ONE, Ordering::AcqRel);
}

fn increment_threads() -> usize {
    const THREAD_ONE: usize = 1;

    (GLOBAL_INFO.epoch.fetch_add(THREAD_ONE, Ordering::AcqRel) + THREAD_ONE) & THREADS_MASK
}

#[cold]
#[inline(never)]
fn update_local(local_info: &Inner) {
    // update epoch
    let new = increment_epoch();
    local_info.epoch.set(new.epoch);

    let local_pile = unsafe { &mut *local_info.pile.get() };
    let mut rem_nodes = 0;
    for item in local_pile.iter() {
        // FIXME: the condition below will race with decrements of thread_count, maybe we can inline the  into
        if item.epoch + new.threads < new.epoch {
            rem_nodes += 1;
            unsafe {
                item.cleanup();
            }
        } else {
            break;
        }
    }
    unsafe {
        core::ptr::copy(
            local_pile.as_mut_ptr().add(rem_nodes),
            local_pile.as_mut_ptr(),
            local_pile.len() - rem_nodes,
        );
    }
    unsafe {
        local_pile.set_len(local_pile.len() - rem_nodes);
    }
    // try resizing if necessary
    if local_pile.len() * 3 < local_pile.capacity() {
        local_pile.shrink_to(local_pile.capacity() / 2);
    }
    let threads = new.threads;
    // try cleanup global
    GLOBAL_INFO.pile.try_drain(|val| {
        // remove instance if distance between current epoch and instance's epoch is at least the number of threads
        // FIXME: this condition seems faulty!
        let remove = val.epoch + threads <= GLOBAL_INFO.update_epoch.load(Ordering::Acquire);
        if remove {
            unsafe {
                val.cleanup();
            }
        }
        remove
    });
    GLOBAL_INFO.locals_pile.try_drain(|local| {
        let remove = unsafe { local.as_ref().unwrap_unchecked() }.epoch.get() + threads < new.epoch;
        if remove {
            cleanup_local(*local);
        }
        remove
    });
}

fn cleanup_local(local: *const Inner) {
    unsafe {
        drop_in_place(local as *mut Inner);
    }
    unsafe {
        dealloc(local as *const Inner as *mut u8, Layout::new::<Inner>());
    }
    // FIXME: there is a bug if a thread acknowledges the new epoch, increments the global counter and then terminates,
    // FIXME: leading to the total number of threads to be decremented, an easy fix would be to increment the update_epoch leading to the dist between,
    // but is that correct?
    // FIXME: is incrementing update_epoch here the correct solution? - this probably is a solution but with unnecessary overhead, try improving this!

    // FIXME: this bug still isn't resolved tho as in case of local updates
    GLOBAL_INFO.update_epoch.fetch_add(1, Ordering::AcqRel);
    decrement_threads();
}

#[inline(never)]
pub(crate) unsafe fn retire_explicit<T>(val: *const T, cleanup: fn(*mut T)) {
    let local_ptr = local_ptr();
    unsafe {
        (&mut *(&*local_ptr).pile.get()).push(Instance::new_explicit(
            val,
            (&*local_ptr).epoch.get(),
            cleanup,
        ));
    }
    let new_epoch = increment_epoch().epoch;
    unsafe { &*local_ptr }.epoch.set(new_epoch);

    let mut curr_epoch = GLOBAL_INFO.update_epoch.load(Ordering::Acquire);
    loop {
        match GLOBAL_INFO.update_epoch.compare_exchange(
            curr_epoch,
            new_epoch,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next_epoch) => {
                if next_epoch > new_epoch {
                    // fail as there was a more recent update than us
                    break;
                }
                // retry setting our epoch
                curr_epoch = next_epoch;
            }
        }
    }
}

pub(crate) unsafe fn retire<T>(val: *const T) {
    let drop_fn = drop_in_place::<T> as *const ();
    let drop_fn = unsafe { transmute(drop_fn) };
    retire_explicit(val, drop_fn);
}

pub struct LocalPinGuard(*const Inner);

unsafe impl Sync for LocalPinGuard {}
unsafe impl Send for LocalPinGuard {}

impl LocalPinGuard {
    pub fn load<T>(&self, ptr: &Guarded<T>, order: Ordering) -> *const T {
        ptr.0.load(order)
    }

    pub fn swap<T>(&self, ptr: &Guarded<T>, val: *const T, order: Ordering) -> *const T {
        ptr.0.swap(val.cast_mut(), order)
    }
}

impl Drop for LocalPinGuard {
    #[inline]
    fn drop(&mut self) {
        // reduce local cnt safely even if src thread terminated

        let local_ptr = local_ptr();
        // fast path for thread local releases
        if self.0 == local_ptr {
            let local = unsafe { &*local_ptr };
            local.active_local.store(
                local.active_local.load(Ordering::Relaxed) - 1,
                Ordering::Release,
            );
            return;
        }

        // reduce shared count
        unsafe { &*self.0 }
            .active_shared
            .fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct Guarded<T>(AtomicPtr<T>);

impl<T> Guarded<T> {
    #[inline]
    pub fn new(ptr: *mut T) -> Self {
        Self(AtomicPtr::new(ptr))
    }
}

struct SyncUnsafeCell<T>(UnsafeCell<T>);

impl<T> SyncUnsafeCell<T> {
    #[inline(always)]
    const fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }

    #[inline(always)]
    const fn get(&self) -> *mut T {
        self.0.get()
    }
}

unsafe impl<T> Send for SyncUnsafeCell<T> {}
unsafe impl<T> Sync for SyncUnsafeCell<T> {}

mod conc_linked_list {
    use core::{
        ptr::null_mut,
        sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    };

    #[cfg(feature = "no_std")]
    use alloc::boxed::Box;

    pub(crate) struct ConcLinkedList<T> {
        root: AtomicPtr<ConcLinkedListNode<T>>,
        flags: AtomicUsize,
    }

    const DRAIN_FLAG: usize = 1 << (usize::BITS as usize - 1);

    impl<T> ConcLinkedList<T> {
        pub(crate) const fn new() -> Self {
            Self {
                root: AtomicPtr::new(null_mut()),
                flags: AtomicUsize::new(0),
            }
        }

        pub(crate) fn is_empty(&self) -> bool {
            self.root.load(Ordering::Acquire).is_null()
        }

        pub(crate) fn is_draining(&self) -> bool {
            self.flags.load(Ordering::Acquire) & DRAIN_FLAG != 0
        }

        pub(crate) fn push_front(&self, val: T) {
            let node = Box::into_raw(Box::new(ConcLinkedListNode {
                next: AtomicPtr::new(null_mut()),
                val,
            }));
            let mut curr = self.root.load(Ordering::Acquire);
            loop {
                unsafe { node.as_ref().unwrap() }
                    .next
                    .store(curr, Ordering::Release);
                match self
                    .root
                    .compare_exchange(curr, node, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => break,
                    Err(new) => {
                        curr = new;
                    }
                }
            }
        }

        pub(crate) fn try_drain<F: Fn(&T) -> bool>(&self, decision: F) -> bool {
            match self
                .flags
                .compare_exchange(0, DRAIN_FLAG, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    let mut next = &self.root;
                    loop {
                        let ptr = next.load(Ordering::Acquire);
                        match unsafe { ptr.as_ref() } {
                            Some(node) => {
                                let rem = decision(&node.val);
                                if rem {
                                    let next_ptr = node.next.load(Ordering::Acquire);
                                    match next.compare_exchange(
                                        ptr,
                                        next_ptr,
                                        Ordering::AcqRel,
                                        Ordering::Relaxed,
                                    ) {
                                        Ok(_) => break,
                                        Err(mut node) => loop {
                                            let new_next = unsafe { node.as_ref().unwrap() }
                                                .next
                                                .load(Ordering::Acquire);
                                            if new_next == ptr {
                                                unsafe { node.as_ref().unwrap() }
                                                    .next
                                                    .store(next_ptr, Ordering::Release);
                                                break;
                                            }
                                            node = new_next;
                                        },
                                    }
                                }
                                next = &node.next;
                            }
                            None => break,
                        }
                    }
                    self.flags.store(0, Ordering::Release);
                    true
                }
                // there is another drain operation in progress
                Err(_) => false,
            }
        }
    }

    #[repr(align(4))]
    struct ConcLinkedListNode<T> {
        next: AtomicPtr<ConcLinkedListNode<T>>,
        val: T,
    }
}

/*
mod untyped_vec {
    use std::{alloc::{alloc, realloc, Layout}, mem::{align_of, size_of}, process::abort, ptr::null_mut};

    pub(crate) struct UntypedVec {
        ptr: *mut usize,
        cap: usize,
        len: usize,
        curr_idx: usize,
    }

    impl UntypedVec {

        pub const fn new() -> Self {
            Self {
                ptr: null_mut(),
                cap: 0,
                len: 0,
                curr_idx: 0,
            }
        }

        #[inline]
        pub const fn capacity(&self) -> usize {
            self.cap
        }

        #[inline]
        pub const fn len(&self) -> usize {
            self.len
        }

        #[inline]
        pub const fn is_empty(&self) -> bool {
            self.len == 0
        }

        const INITIAL_CAP_MUL: usize = 8;
        const CAP_MUL: usize = 2;

        pub fn push<T>(&mut self, val: T) {
            let remaining = self.cap - self.curr_idx;
            let req = size_of::<T>().next_multiple_of(size_of::<usize>() + size_of::<usize>() + align_of::<T>() + size_of::<fn()>()) / size_of::<usize>();
            if remaining < req {
                if self.ptr.is_null() {
                    let alloc = unsafe { alloc(Layout::from_size_align(req * Self::INITIAL_CAP_MUL, size_of::<usize>()).unwrap()) };
                    if alloc.is_null() {
                        abort();
                    }
                    self.ptr = alloc.cast::<usize>();
                    self.cap = req * Self::INITIAL_CAP_MUL;
                } else {
                    // FIXME: should we try using realloc?
                    let new_alloc = unsafe { alloc(Layout::array::<usize>(req).unwrap()) }.cast::<usize>();
                    if new_alloc.is_null() {
                        abort();
                    }
                    unsafe { core::ptr::copy_nonoverlapping(self.ptr, new_alloc, req); }
                }
            }
            self.len += 1;
            self.curr_idx +=
        }

    }
}*/

#[cfg(test)]
mod test {
    use core::{mem::transmute, sync::atomic::Ordering};
    use std::sync::Arc;
    use std::thread;

    use crate::epoch::{self, retire_explicit, Guarded};

    #[test]
    fn load_multi() {
        unsafe fn cleanup_box<T>(ptr: *mut T) {
            let _ = Box::from_raw(ptr);
        }

        let cleanup_fn = cleanup_box::<String> as *const ();
        let cleanup_fn = unsafe { transmute(cleanup_fn) };

        let initial = Box::new("test".to_string());
        let guard = Arc::new(Guarded::new(Box::into_raw(initial)));
        let pin = epoch::pin();
        let pinned = pin.load(&guard, Ordering::Acquire);
        println!("pinned: {}", unsafe { &*pinned });
        let move_guard = guard.clone();
        let other = thread::spawn(move || {
            let guard = move_guard;
            let pin = epoch::pin();
            for i in 0..100 {
                let i = 100 + i;
                let curr = Box::new(format!("test{i}"));
                let prev = pin.swap(&guard, Box::into_raw(curr), Ordering::AcqRel);
                unsafe {
                    retire_explicit(prev, cleanup_fn);
                }
            }
            println!("finished2");
        });
        for i in 0..100 {
            let curr = Box::new(format!("test{i}"));
            let prev = pin.swap(&guard, Box::into_raw(curr), Ordering::AcqRel);
            unsafe {
                retire_explicit(prev, cleanup_fn);
            }
        }
        println!("finished1");

        other.join().unwrap();
        unsafe {
            retire_explicit(pin.load(&guard, Ordering::Acquire), cleanup_fn);
        }
    }
}

#[cfg(not(feature = "no_std"))]
#[inline]
fn abort() -> ! {
    use std::process::abort;

    abort()
}

#[cfg(feature = "no_std")]
#[inline]
fn abort() -> ! {
    use core::intrinsics::abort;

    abort()
}
