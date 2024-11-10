use cfg_if::cfg_if;
#[cfg(feature = "asm_thrid")]
use core::num::NonZeroUsize;
use core::{
    cell::{Cell, UnsafeCell},
    hint::spin_loop,
    mem::transmute,
    ptr::{drop_in_place, null_mut, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    time::Duration,
};
use crossbeam_utils::CachePadded;
use likely_stable::unlikely;
use std::thread;

use crate::{
    epoch::conc_linked_list::ConcLinkedListNode,
    reclamation::{ReclamationMode, ReclamationStrategy},
};

use self::{conc_linked_list::ConcLinkedList, ring_buf::RingBuffer};

static GLOBAL_INFO: GlobalInfo = GlobalInfo {
    cold: CachePadded::new(ColdGlobals {
        pile: ConcLinkedList::new(),
        locals: ConcLinkedList::new(),
    }),
    update_epoch: AtomicUsize::new(0),
    min_safe_epoch: AtomicUsize::new(0),
};

struct GlobalInfo {
    cold: CachePadded<ColdGlobals>,
    update_epoch: AtomicUsize,
    min_safe_epoch: AtomicUsize,
}

struct ColdGlobals {
    pile: ConcLinkedList<Instance>, // FIXME: have a fixed-sized ring-buffer to not require heap allocations for everything
    // a list of all threads that have performed a pin call before
    locals: ConcLinkedList<Inner>, // FIXME: cache reusable locals
}

cfg_if! {
    if #[cfg(feature = "no_std")] {
        use libc::ESRCH;
        use alloc::boxed::Box;

        #[thread_local]
        static LOCAL_INFO: SyncUnsafeCell<*const ConcLinkedListNode<Inner>> =
            SyncUnsafeCell::new(null_mut());

        #[inline]
        fn local_raw_ptr() -> *mut *const ConcLinkedListNode<Inner> {
            let src = LOCAL_INFO.get();
            src
        }
    } else {
        use std::thread_local;

        // FIXME: are 2 separate TLS variables actually better than 1 with an associated destructor?
        thread_local! {
            static LOCAL_INFO: SyncUnsafeCell<*const ConcLinkedListNode<Inner>> = const { SyncUnsafeCell::new(null_mut()) };
            static LOCAL_GUARD: LocalGuard = const { LocalGuard };
        }

        #[inline]
        fn local_raw_ptr() -> *mut *const ConcLinkedListNode<Inner> {
            LOCAL_INFO.with(|ptr| ptr.get())
        }
    }
}

pub(crate) const LOCAL_PILE_SIZE: usize = 64;

#[cfg(feature = "asm_thrid")]
#[inline(always)]
fn get_tid() -> NonZeroUsize {
    // FIXME: use the following once the crate gets recognized: thrid::ThrId::get().value()
    crate::tid::tid_impl()
}

// FIXME: instead of checking if thread is still alive periodically, instead register a TLS destructor
// using pthread_key_create

struct Inner {
    pile: SyncUnsafeCell<RingBuffer<Instance, LOCAL_PILE_SIZE>>,
    global_node_ptr: *mut ConcLinkedListNode<Self>,
    epoch: AtomicUsize,
    active_local: AtomicUsize, // the MSB, if set indicates that this Inner is safe to remove
    active_shared: AtomicUsize,
    // this is used for the threshold and lazy reclamation modes
    action_cnt: Cell<usize>,
    #[cfg(feature = "asm_thrid")]
    thrid: AtomicUsize,
    #[cfg(feature = "no_std")]
    tid: libc::pthread_t,
}

impl Inner {
    fn new() -> Self {
        Self {
            pile: SyncUnsafeCell::new(RingBuffer::new()),
            global_node_ptr: null_mut(),
            epoch: AtomicUsize::new(GLOBAL_INFO.update_epoch.load(Ordering::Acquire)),
            active_local: AtomicUsize::new(0),
            active_shared: AtomicUsize::new(0),
            action_cnt: Cell::new(0),
            #[cfg(feature = "asm_thrid")]
            thrid: AtomicUsize::new(get_tid().get()),
            #[cfg(feature = "no_std")]
            tid: libc::pthread_self(),
        }
    }
}

struct LocalGuard;

impl Drop for LocalGuard {
    fn drop(&mut self) {
        if let Some(local) = try_get_local() {
            destroy_local(local);
        }
    }
}

fn destroy_local(local: &Inner) {
    // store a sentinel in order to ensure that if our thread id gets reassigned to a new thread
    // a comparison in the guard's drop function will always fail (as it should)
    #[cfg(feature = "asm_thrid")]
    local.thrid.store(0, Ordering::Release);
    let min_safe = GLOBAL_INFO.min_safe_epoch.load(Ordering::Acquire);
    for garbage in unsafe { &*local.pile.get() }.iter() {
        if garbage.epoch < min_safe {
            // release garbage as nobody looks at it anymore
            unsafe {
                garbage.cleanup();
            }
        } else {
            // push garbage onto pile
            GLOBAL_INFO.cold.pile.push_front(garbage.clone());
        }
    }
    // if there are no more active references to the local
    if local.active_shared.load(Ordering::Acquire) == local.active_local.load(Ordering::Acquire) {
        #[inline]
        fn finish(local: *const Inner) {
            // there are no more external references, so local can be cleaned up immediately
            // FIXME: don't iterate the whole list but only check a single entry (and store said entry in the local itself)
            // FIXME: understanding: why can the local even not be present in the locals list?
            if GLOBAL_INFO
                .cold
                .locals
                .try_drain(|curr| curr as *const _ == local)
            {
                // we don't have to cleanup the local as its in the same allocation as the node
            } else {
                let local = unsafe { &*local };
                // set the indicator to ensure others can destroy this local
                local.epoch.store(
                    local.epoch.load(Ordering::Acquire) | LOCAL_DESTROYED,
                    Ordering::Release,
                );
            }
        }
        finish(local as *const Inner);
    } else {
        // NOTE: at this point we know that there must be external references
        // to data retrieves through this guard remaining
    }
}

#[derive(Clone)]
struct Instance {
    drop_fn: unsafe fn(NonNull<()>),
    data_ptr: NonNull<()>,
    epoch: usize,
}

impl Instance {
    #[inline]
    fn new_explicit<T>(val: NonNull<T>, epoch: usize, drop_fn: unsafe fn(NonNull<T>)) -> Self {
        Self {
            drop_fn: unsafe { transmute::<_, fn(NonNull<()>)>(drop_fn) },
            data_ptr: val.cast::<()>(),
            epoch,
        }
    }

    #[inline]
    unsafe fn cleanup(&self) {
        let drop_fn = self.drop_fn;
        drop_fn(self.data_ptr);
    }
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

#[inline]
fn local_ptr() -> *const ConcLinkedListNode<Inner> {
    let src = local_raw_ptr();
    unsafe { *src }
}

#[inline]
fn try_get_local<'a>() -> Option<&'a Inner> {
    unsafe { local_ptr().as_ref().map(|node| node.val()) }
}

#[inline]
fn get_local<'a>() -> &'a Inner {
    #[cold]
    #[inline(never)]
    fn alloc_local<'a>() -> &'a Inner {
        let src = local_raw_ptr();
        unsafe {
            let alloc = Box::into_raw(Box::new(ConcLinkedListNode::new(Inner::new())));
            *src = alloc;
            (&mut *alloc).val_mut().global_node_ptr = alloc;
            GLOBAL_INFO.cold.locals.push_front_optimistic(alloc);
            #[cfg(not(feature = "no_std"))]
            LOCAL_GUARD.with(|_| {});
            (*alloc).val()
        }
    }

    if let Some(local) = try_get_local() {
        return local;
    }

    alloc_local()
}

/// This may not be called if there is already a pin active
#[inline]
pub(crate) fn pin<R: ReclamationStrategy>() -> LocalPinGuard {
    let local_info = get_local();
    // this is relaxed as we know that we are the only thread currently to modify any local data
    let active = local_info.active_local.load(Ordering::Relaxed);
    local_info.active_local.store(active + 1, Ordering::Release);
    if active != 0 {
        return LocalPinGuard(local_info as *const Inner);
    }
    let update_epoch = GLOBAL_INFO.update_epoch.load(Ordering::Acquire);
    let local = local_info.epoch.load(Ordering::Relaxed);
    if update_epoch > local {
        update_local::<R>(local_info);
    }
    LocalPinGuard(local_info as *const Inner)
}

#[inline]
fn increment_update_epoch() -> usize {
    #[cold]
    #[inline(never)]
    fn handle_guard_hit(epoch: &mut usize) {
        // if we hit the looser limit, we can't recover anymore
        if *epoch >= EPOCH_OVERFLOW_GUARD {
            abort();
        }
        // this condition should always be checked by the caller before calling this function
        debug_assert!(*epoch >= EPOCH_WRAP_AROUND_GUARD);
        // if we hit the looser limit, try recovering
        if *epoch >= EPOCH_WRAP_AROUND_GUARD {
            let mut curr = *epoch;
            loop {
                match GLOBAL_INFO.update_epoch.compare_exchange(
                    curr,
                    curr - EPOCH_WRAP_AROUND_GUARD,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        *epoch -= EPOCH_WRAP_AROUND_GUARD;
                        return;
                    }
                    Err(new) => {
                        if new < EPOCH_WRAP_AROUND_GUARD {
                            let new = GLOBAL_INFO
                                .update_epoch
                                .fetch_add(EPOCH_ONE, Ordering::AcqRel)
                                + EPOCH_ONE;
                            debug_assert!(new < EPOCH_WRAP_AROUND_GUARD);
                            *epoch = new;
                            return;
                        }
                        curr = new;
                    }
                }
            }
        }
    }

    const EPOCH_ONE: usize = 1;
    const EPOCH_WRAP_AROUND_GUARD: usize = usize::MAX >> 2;
    const EPOCH_OVERFLOW_GUARD: usize = usize::MAX >> 1;

    let mut new = GLOBAL_INFO
        .update_epoch
        .fetch_add(EPOCH_ONE, Ordering::AcqRel)
        + EPOCH_ONE;

    if unlikely(new >= EPOCH_WRAP_AROUND_GUARD) {
        handle_guard_hit(&mut new);
    }

    new
}

#[cold]
#[inline(never)]
fn update_local<R: ReclamationStrategy>(local_info: &Inner) {
    // update epoch
    let new = GLOBAL_INFO.update_epoch.load(Ordering::Acquire);
    local_info.epoch.store(new, Ordering::Release);

    match R::mode().threshold() {
        None => {
            try_cleanup(local_info, new);
        }
        Some(threshold) => {
            let curr_actions = local_info.action_cnt.get() + 1;
            if curr_actions >= threshold {
                try_cleanup(local_info, new);
                local_info.action_cnt.set(0);
            } else {
                local_info.action_cnt.set(curr_actions);
            }
        }
    }
}

fn try_cleanup(local_info: &Inner, update_epoch: usize) {
    let min_safe = GLOBAL_INFO.min_safe_epoch.load(Ordering::Acquire);

    let local_pile = unsafe { &mut *local_info.pile.get() };
    local_pile.evict_while(|item| {
        let rem = unsafe { &*item }.epoch < min_safe;
        if rem {
            unsafe { (*item).cleanup() };
        }
        rem
    });
    // try cleanup global
    GLOBAL_INFO.cold.pile.try_drain(|val| {
        let remove = val.epoch < min_safe;
        if remove {
            unsafe {
                val.cleanup();
            }
        }
        remove
    });
    #[cfg(not(feature = "no_std"))]
    if min_safe == update_epoch {
        // don't try to update min_safe if it is already up-to-date
        return;
    }
    let mut min_used = usize::MAX;
    GLOBAL_INFO.cold.locals.try_drain(|local| {
        let local_epoch = local.epoch.load(Ordering::Acquire);
        #[cfg(feature = "no_std")]
        if unsafe { libc::pthread_kill(local.origin_thread, 0) } == ESRCH {
            // the thread has exited, so cleanup the remaining data
            return true;
        }
        if local_epoch & LOCAL_DESTROYED != 0 {
            // we don't have to cleanup local as its inlined inside the node's allocation
            return true;
        }
        // at this point we know that the local is either still in use locally or should be cleaned up by its remaining remote users
        let outdated = local_epoch < update_epoch;
        if outdated {
            // the order in which these two loads happen in the comparison is very important!
            // we first load active_shared as it may only ever grow and never shrink and then active_local which may both grow and shrink (but never below active_shared)
            // this means that after loading active_shared even if active_local gets decremented as much as it can it could only ever lead to either matching
            // the active_shared we observed (=> not in use) or being larger than active the active shared we observed (=> in use) which would only delay
            // us cleaning up the garbage which is okay and not prematurely which would result in USE-AFTER-FREE
            // TODO: is this okay performance-wise (for the cache) or should we instead add a flag into the epoch variable and check that?
            let in_use = local.active_shared.load(Ordering::Acquire)
                != local.active_local.load(Ordering::Acquire);
            if in_use && local_epoch < min_used {
                min_used = local_epoch;
            }
        }
        false
    });
    #[cfg(feature = "no_std")]
    if min_safe == update_epoch {
        // don't try to update min_safe if it is already up-to-date
        return;
    }
    if min_used == usize::MAX {
        min_used = update_epoch;
    }
    if min_used > min_safe {
        let mut curr_safe = min_safe;
        loop {
            match GLOBAL_INFO.min_safe_epoch.compare_exchange(
                curr_safe,
                min_used,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_safe) => {
                    // check if the new safe value is more recent than our calculated value
                    if new_safe >= min_used {
                        break;
                    }
                    curr_safe = new_safe;
                }
            }
        }
    }
}

const LOCAL_DESTROYED: usize = 1 << (usize::BITS - 1);

// #[inline(never)]
#[inline]
pub(crate) unsafe fn retire_explicit<T, R: ReclamationStrategy>(
    val: NonNull<T>,
    cleanup: fn(NonNull<T>),
) {
    let local = get_local();
    let old = unsafe {
        (*local.pile.get()).try_push_back(Instance::new_explicit(
            val,
            // FIXME: explain ordering
            local.epoch.load(Ordering::Relaxed),
            cleanup,
        ))
    };
    if let Some(old) = old {
        // there is no more space in the local pile, try letting global handle our old garbage instead
        GLOBAL_INFO.cold.pile.push_front(old);
    }
    let new_epoch = increment_update_epoch();
    local.epoch.store(new_epoch, Ordering::Release);
    match R::mode() {
        ReclamationMode::Eager => {
            try_cleanup(local, new_epoch);
        }
        ReclamationMode::Threshold { threshold } => {
            let curr_actions = local.action_cnt.get() + 1;
            if curr_actions >= threshold {
                try_cleanup(local, new_epoch);
                local.action_cnt.set(0);
            } else {
                local.action_cnt.set(curr_actions);
            }
        }
        ReclamationMode::Lazy => {
            // noop
        }
    }
}

#[allow(dead_code)]
pub(crate) unsafe fn retire<T, R: ReclamationStrategy>(val: NonNull<T>) {
    let drop_fn = drop_in_place::<T> as *const ();
    let drop_fn = unsafe { transmute(drop_fn) };
    retire_explicit::<T, R>(val, drop_fn);
}

pub struct LocalPinGuard(*const Inner);

unsafe impl Sync for LocalPinGuard {}
unsafe impl Send for LocalPinGuard {}

impl LocalPinGuard {
    #[inline]
    pub fn load<T>(&self, ptr: &Guarded<T>, order: Ordering) -> *const T {
        ptr.0.load(order)
    }

    #[inline]
    pub fn swap<T>(&self, ptr: &Guarded<T>, val: *const T, order: Ordering) -> *const T {
        ptr.0.swap(val.cast_mut(), order)
    }

    #[cfg(feature = "ptr_ops")]
    #[inline]
    pub fn compare_exchange<T>(
        &self,
        ptr: &Guarded<T>,
        current: *const T,
        new: *const T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<*mut T, *mut T> {
        ptr.0
            .compare_exchange(current.cast_mut(), new.cast_mut(), success, failure)
    }
}

#[cfg(feature = "asm_thrid")]
impl Drop for LocalPinGuard {
    #[inline]
    fn drop(&mut self) {
        // reduce local cnt safely even if src thread terminated
        let inner = unsafe { &*self.0 };

        // we use a thread id that's generated through some (hopefully fast means)
        let curr_id = get_tid().get();
        // fast path for thread local releases
        if inner.thrid.load(Ordering::Relaxed) == curr_id {
            inner.active_local.store(
                inner.active_local.load(Ordering::Relaxed) - 1,
                Ordering::Release,
            );
            return;
        }

        // reduce shared count
        inner.active_shared.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(not(feature = "asm_thrid"))]
impl Drop for LocalPinGuard {
    #[inline]
    fn drop(&mut self) {
        let inner = unsafe { &*self.0 };
        // reduce local cnt safely even if src thread terminated

        // fast path for thread local releases
        if self.0
            == unsafe {
                local_ptr()
                    .byte_add(ConcLinkedListNode::<Inner>::addr_offset())
                    .cast::<Inner>()
            }
        {
            inner.active_local.store(
                inner.active_local.load(Ordering::Relaxed) - 1,
                Ordering::Release,
            );
            return;
        }

        // reduce shared count
        inner.active_shared.fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct Guarded<T>(AtomicPtr<T>);

impl<T> Guarded<T> {
    #[inline]
    pub const fn new(ptr: *mut T) -> Self {
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

pub mod buffered_queue {
    use core::{
        mem::{offset_of, MaybeUninit},
        ptr::NonNull,
        sync::atomic::{AtomicU8, AtomicUsize, Ordering},
        usize,
    };

    use aligned::{Aligned, A2};

    use super::{first_set_bit, wait_while, ConcLinkedList, ConcLinkedListNode, SyncUnsafeCell};

    // FIXME: test this data structure!
    pub struct BufferedQueue<const BUFFER: usize, T> {
        // this field contains the index of the element that'S currently indexed by an iterator
        // if no element is currently indexed by an iterator, this field is usize::MAX
        // FIXME: should we instead of having a single variable for all cells, instead have one field per cell
        // FIXME: in order to reduce false sharing or cache coherency
        iter_idx: AtomicUsize,
        alloc_mask: AtomicUsize,
        cold_list: ConcLinkedList<Aligned<A2, T>>,
        buffer: [Cell<T>; BUFFER],
    }

    /// this flag is for the cell's meta flags
    const PRESENT_FLAG: u8 = 1 << 1;
    const ALLOC_FLAG: u8 = 1 << 0;
    /// no iteration flag indicates that there is no iteration happening on any element currently
    const NO_ITER_FLAG: usize = usize::MAX as usize;

    struct Cell<T> {
        // this may contain flags such as the present flag
        meta_flags: AtomicU8,
        val: SyncUnsafeCell<MaybeUninit<Aligned<A2, T>>>,
    }

    impl<const BUFFER: usize, T> BufferedQueue<BUFFER, T> {
        pub const fn new() -> Self {
            const {
                if BUFFER > usize::BITS as usize / 2 {
                    // FIXME: format (usize::BITS) into this string once that's supported in const contexts
                    panic!("The buffer can be at most of the architecture's word size (usize)");
                }
            };
            Self {
                iter_idx: AtomicUsize::new(usize::MAX),
                alloc_mask: AtomicUsize::new(0),
                buffer: [const {
                    Cell {
                        val: SyncUnsafeCell::new(MaybeUninit::uninit()),
                        /*iter: AtomicBool::new(false), */ meta_flags: AtomicU8::new(0),
                    }
                }; BUFFER],
                cold_list: ConcLinkedList::new(),
            }
        }

        pub(crate) fn push_front_optimistic(&self, val: T) -> QueueNode<T> {

        }

        pub fn push(&self, val: T) -> QueueNode<T> {
            let mut curr_mask = self.alloc_mask.load(Ordering::Acquire);
            while curr_mask != usize::MAX {
                let slot = first_set_bit(curr_mask);
                let prev = self.alloc_mask.fetch_or(slot, Ordering::AcqRel);
                if prev & slot != 0 {
                    // we raced with another thread trying to allocate slot, retry
                    curr_mask = prev;
                    continue;
                }
                let slot_idx = slot.trailing_zeros() as usize;
                unsafe { &mut *self.buffer[slot_idx].val.get() }.write(Aligned(val));
                self.buffer[slot_idx].meta_flags.store(ALLOC_FLAG | PRESENT_FLAG, Ordering::Release);
                return QueueNode(unsafe { NonNull::new_unchecked((&mut *self.buffer[slot_idx].val.get()).as_mut_ptr()) });
            }
            QueueNode(unsafe { self.cold_list.push_front_optimistic(Aligned(val)).cast::<Aligned<A2, T>>().byte_add(ConcLinkedListNode::<Aligned<A2, T>>::VAL_OFF) })
        }

        pub fn try_pop(&self, val: NonNull<T>) -> bool {
            if val.addr().get() > (&self.buffer as *const _) as usize
                && val.addr().get()
                    < (&self.buffer as *const _) as usize
                        + BUFFER * size_of::<SyncUnsafeCell<MaybeUninit<T>>>()
            {
                let idx = (val.addr().get() - (&self.buffer as *const _) as usize) / BUFFER;
                // clear the PRESENT flag from flags
                self.buffer[idx].meta_flags.store(ALLOC_FLAG, Ordering::Release);
                // wait until we are sure there is nobody looking at the cell anymore ;) (and they have to see that it's invalidated when they dare to look again)
                wait_while(|| self.iter_idx.load(Ordering::Acquire) == idx);
                unsafe {
                    (&mut *self.buffer[idx].val.0.get()).assume_init_drop();
                }
                // free the space
                self.buffer[idx].meta_flags.store(0, Ordering::Release);
                self.alloc_mask.fetch_and(!(1 << idx), Ordering::AcqRel);
                return true;
            }
            self.cold_list
                .try_drain(|other| other as *const _ as *mut _ == val.as_ptr())
        }

        pub fn try_drain<F: FnMut(&T) -> bool>(&self, mut decision: F) -> bool {
            if self
                .iter_idx
                .compare_exchange(usize::MAX, 0, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                return false;
            }
            for i in 0..BUFFER {
                let flags = self.buffer[i].meta_flags.load(Ordering::Acquire);
                if flags == ALLOC_FLAG {
                    // fail quickly and give up when we encounter any resistance or contention
                    return false;
                }
                if flags == 0 {
                    // this slot isn't currently allocated
                    continue;
                }
                if decision(unsafe { (&*self.buffer[i].val.0.get()).assume_init_ref() }) {
                    // remove val
                    
                    // clear flags
                    self.buffer[i].meta_flags.store(0, Ordering::Release);
                    unsafe {
                        (&mut *self.buffer[i].val.0.get()).assume_init_drop();
                    }
                    self.alloc_mask.fetch_and(!(1 << i), Ordering::AcqRel);
                    // mark iterator as finished
                    self.iter_idx.store(usize::MAX as usize, Ordering::Release);
                    return true;
                }
                // this may point to an element outside of buffer which is okay as it indicates that an iterator is present but it doesn't point at any element in buffer
                self.iter_idx.store(i + 1, Ordering::Release);
            }

            self.cold_list.drain_unchecked(decision);

            self.iter_idx.store(usize::MAX as usize, Ordering::Release);
            true
        }
    }

    pub struct QueueNode<T>(NonNull<Aligned<A2, T>>);

    impl<T> QueueNode<T> {
        #[inline]
        pub fn get(&self) -> &T {
            unsafe { &*((self.0.addr().get() & !1) as *mut Aligned<A2, T>) }
        }
    }
}

#[inline]
const fn first_set_bit(val: usize) -> usize {
    1 << val.trailing_zeros()
}

fn wait_while<F: Fn() -> bool>(f: F) {
    let mut i = 0;
    while f() {
        spin_loop();
        if i >= (u16::MAX as usize) {
            thread::sleep(Duration::from_millis(1));
        }
        i += 1;
    }
}

mod conc_linked_list {
    use core::{
        mem::offset_of,
        ptr::{null_mut, NonNull},
        sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    };

    #[cfg(feature = "no_std")]
    use alloc::boxed::Box;
    use likely_stable::unlikely;

    pub(crate) struct ConcLinkedList<T> {
        root: AtomicPtr<ConcLinkedListNode<T>>,
        flags: AtomicUsize,
    }

    const DRAIN_FLAG: usize = 1 << (usize::BITS as usize - 1);

    impl<T> ConcLinkedList<T> {
        #[inline]
        pub(crate) const fn new() -> Self {
            Self {
                root: AtomicPtr::new(null_mut()),
                flags: AtomicUsize::new(0),
            }
        }

        #[inline]
        pub(crate) fn is_empty(&self) -> bool {
            self.root.load(Ordering::Acquire).is_null()
        }

        pub(crate) fn push_front(&self, val: T) {
            let node = Box::into_raw(Box::new(ConcLinkedListNode {
                next: AtomicPtr::new(null_mut()),
                val,
            }));
            // this may be relaxed as we aren't accessing any data from curr and only need to confirm
            // that it is the current value of root (even after storing it into our next value)
            let mut curr = self.root.load(Ordering::Relaxed);
            loop {
                *unsafe { &mut *node }.next.get_mut() = curr;
                // FIXME: explain the release ordering here
                match self
                    .root
                    .compare_exchange(curr, node, Ordering::Release, Ordering::Acquire)
                {
                    Ok(_) => break,
                    Err(new) => {
                        curr = new;
                    }
                }
            }
        }

        pub(crate) fn push_front_optimistic(&self, val: T) -> NonNull<ConcLinkedListNode<T>> {
            let mut node = Box::new(ConcLinkedListNode::new(val));
            // this may be relaxed as we aren't accessing any data from curr and only need to confirm
            // that it is the current value of root (even after storing it into our next value)
            let curr = self.root.load(Ordering::Relaxed);
            *node.next.get_mut() = curr;
            let node = Box::into_raw(node);
            // FIXME: explain the release ordering here
            if unlikely(
                self.root
                    .compare_exchange(curr, node, Ordering::Release, Ordering::Relaxed)
                    .is_err(),
            ) {
                #[inline(never)]
                #[cold]
                fn push<T>(
                    root: &AtomicPtr<ConcLinkedListNode<T>>,
                    node: *mut ConcLinkedListNode<T>,
                ) {
                    // this may be relaxed as we aren't accessing any data from curr and only need to confirm
                    // that it is the current value of root (even after storing it into our next value)
                    let mut curr = root.load(Ordering::Relaxed);
                    loop {
                        *unsafe { &mut *node }.next.get_mut() = curr;
                        // FIXME: explain the release ordering here
                        match root.compare_exchange(
                            curr,
                            node,
                            Ordering::Release,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => break,
                            Err(new) => {
                                curr = new;
                            }
                        }
                    }
                }
                push(&self.root, node);
            }
            // FIXME: explain safety here
            unsafe { NonNull::new_unchecked(node) }
        }

        pub(crate) fn try_drain<F: FnMut(&T) -> bool>(&self, decision: F) -> bool {
            if self.is_empty() {
                return true;
            }
            match self
                .flags
                .compare_exchange(0, DRAIN_FLAG, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.drain_unchecked(decision);
                    self.flags.store(0, Ordering::Release);
                    true
                }
                // there is another drain operation in progress
                Err(_) => false,
            }
        }

        pub(crate) fn drain_unchecked<F: FnMut(&T) -> bool>(&self, mut decision: F) {
            let mut node_src = &self.root;
            let mut node = node_src.load(Ordering::Acquire);
            while !node.is_null() {
                let rem = decision(&unsafe { &*node }.val);
                if rem {
                    let next = unsafe { &*node }.next.load(Ordering::Acquire);
                    match node_src.compare_exchange(node, next, Ordering::AcqRel, Ordering::Acquire)
                    {
                        Ok(_) => {
                            // delete node
                            let _ = unsafe { Box::from_raw(node) };
                            node = node_src.load(Ordering::Acquire);
                        }
                        Err(mut new) => {
                            // the update failed so our src has to have been head and a push must have happened

                            // try finding the parent of the to be removed node, starting from the beginning
                            loop {
                                node_src = &unsafe { &*new }.next;
                                new = node_src.load(Ordering::Acquire);
                                if new == node {
                                    break;
                                }
                            }
                            let next = unsafe { &*new }.next.load(Ordering::Acquire);
                            node_src.store(next, Ordering::Release);
                            // delete node
                            let _ = unsafe { Box::from_raw(node) };
                            node = next;
                        }
                    }
                } else {
                    node_src = &unsafe { &*node }.next;
                    node = node_src.load(Ordering::Acquire);
                }
            }
        }
    }

    // FIXME: why are we aligning here?
    #[repr(align(4))]
    pub(crate) struct ConcLinkedListNode<T> {
        next: AtomicPtr<ConcLinkedListNode<T>>,
        val: T,
    }

    impl<T> ConcLinkedListNode<T> {

        pub const VAL_OFF: usize = offset_of!(ConcLinkedListNode<T>, val);

        #[inline]
        pub fn new(val: T) -> Self {
            Self {
                next: AtomicPtr::new(null_mut()),
                val,
            }
        }

        #[inline]
        pub fn val(&self) -> &T {
            &self.val
        }

        #[inline]
        pub fn val_mut(&mut self) -> &mut T {
            &mut self.val
        }

        #[inline]
        pub const fn addr_offset() -> usize {
            offset_of!(Self, val)
        }
    }
}

mod ring_buf {
    use core::mem::MaybeUninit;

    pub(crate) struct RingBuffer<T, const LEN: usize> {
        // FIXME: should we force the head and len data at the start of the struct (as it's probably accessed far more frequently)
        head: usize,
        len: usize,
        buf: [MaybeUninit<T>; LEN],
    }

    impl<T, const LEN: usize> RingBuffer<T, LEN> {
        pub const fn new() -> Self {
            Self {
                // TODO: use uninit_array instead, once stabilized
                buf: [const { MaybeUninit::uninit() }; LEN],
                head: 0,
                len: 0,
            }
        }

        pub fn try_push_back(&mut self, val: T) -> Option<T> {
            if self.len != LEN {
                self.buf[(self.head + self.len) % LEN].write(val);
                self.len += 1;
                return None;
            }

            let old = unsafe { self.buf[(self.head + self.len) % LEN].assume_init_read() };
            self.buf[(self.head + self.len) % LEN].write(val);
            self.head += 1;
            Some(old)
        }

        /// evicts elements starting from head until cond is false
        pub fn evict_while<F: FnMut(*const T) -> bool>(&mut self, mut cond: F) {
            for i in 0..self.len {
                let idx = (self.head + i) % LEN;
                let elem = self.buf[idx].as_ptr();
                if !cond(elem) {
                    self.head += i;
                    self.len -= i;
                    return;
                }
            }
            self.head += self.len;
            self.len = 0;
        }

        #[inline]
        pub fn iter(&self) -> Iter<T, LEN> {
            Iter { src: self, idx: 0 }
        }

        #[inline(always)]
        pub const fn len(&self) -> usize {
            self.len
        }
    }

    pub(crate) struct Iter<'a, T, const LEN: usize> {
        src: &'a RingBuffer<T, LEN>,
        idx: usize,
    }

    impl<'a, T, const LEN: usize> Iterator for Iter<'a, T, LEN> {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            debug_assert!(self.idx <= self.src.len());
            if self.idx == self.src.len() {
                return None;
            }
            let idx = self.src.head + self.idx;
            self.idx += 1;
            Some(unsafe { self.src.buf[idx % LEN].assume_init_ref() })
        }
    }

    impl<T, const LEN: usize> ExactSizeIterator for Iter<'_, T, LEN> {
        #[inline]
        fn len(&self) -> usize {
            self.src.len() - self.idx
        }
    }
}

#[cfg(all(not(feature = "no_std"), test))]
mod test {
    use core::{mem::transmute, ptr::NonNull, sync::atomic::Ordering};
    use std::sync::Arc;
    use std::thread;

    use crate::{
        epoch::{self, retire_explicit, Guarded},
        reclamation::Balanced,
    };

    #[test]
    fn load_multi() {
        unsafe fn cleanup_box<T>(ptr: *mut T) {
            let _ = Box::from_raw(ptr);
        }

        let cleanup_fn = cleanup_box::<String> as *const ();
        let cleanup_fn = unsafe { transmute(cleanup_fn) };

        let initial = Box::new("test".to_string());
        let guard = Arc::new(Guarded::new(Box::into_raw(initial)));
        let pin = epoch::pin::<Balanced>();
        let pinned = pin.load(&guard, Ordering::Acquire);
        println!("pinned: {}", unsafe { &*pinned });
        let move_guard = guard.clone();
        let other = thread::spawn(move || {
            let guard = move_guard;
            let pin = epoch::pin::<Balanced>();
            for i in 0..100 {
                let i = 100 + i;
                let curr = Box::new(format!("test{i}"));
                let prev = pin.swap(&guard, Box::into_raw(curr), Ordering::AcqRel);
                unsafe {
                    retire_explicit::<String, Balanced>(
                        NonNull::new(prev.cast_mut()).unwrap(),
                        cleanup_fn,
                    );
                }
            }
            println!("finished2");
        });
        for i in 0..100 {
            let curr = Box::new(format!("test{i}"));
            let prev = pin.swap(&guard, Box::into_raw(curr), Ordering::AcqRel);
            unsafe {
                retire_explicit::<String, Balanced>(
                    NonNull::new(prev.cast_mut()).unwrap(),
                    cleanup_fn,
                );
            }
        }
        println!("finished1");

        other.join().unwrap();
        unsafe {
            retire_explicit::<String, Balanced>(
                NonNull::new(pin.load(&guard, Ordering::Acquire).cast_mut()).unwrap(),
                cleanup_fn,
            );
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
