use cfg_if::cfg_if;
use core::{
    cell::{Cell, UnsafeCell},
    mem::transmute,
    num::NonZeroUsize,
    ptr::{drop_in_place, null_mut, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
};
use crossbeam_utils::CachePadded;
use likely_stable::unlikely;

use crate::{
    epoch::conc_linked_list::ConcLinkedListNode,
    reclamation::{ReclamationMode, ReclamationStrategy, LAZY_THRESHOLD},
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
    pile: ConcLinkedList<Instance>,
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

        // FIXME: are 2 seperate TLS variables actually better than 1 with an associated destructor?
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

#[inline(always)]
fn get_tid() -> NonZeroUsize {
    // FIXME: use the following once the crate gets recognized: thrid::ThrId::get().value()
    crate::tid::tid_impl()
}

// FIXME: instead of checking if thread is still alive periodically, instead register a TLS destructor
// using pthread_key_create

struct Inner {
    tid: AtomicUsize,
    pile: SyncUnsafeCell<RingBuffer<Instance, LOCAL_PILE_SIZE>>,
    epoch: AtomicUsize,
    active_local: AtomicUsize, // the MSB, if set indicates that this Inner is safe to remove
    active_shared: AtomicUsize,
    // this is used for the threshold and lazy reclamation modes
    action_cnt: Cell<usize>,
}

impl Inner {
    fn new() -> Self {
        Self {
            pile: SyncUnsafeCell::new(RingBuffer::new()),
            epoch: AtomicUsize::new(GLOBAL_INFO.update_epoch.load(Ordering::Acquire)),
            active_local: AtomicUsize::new(0),
            active_shared: AtomicUsize::new(0),
            action_cnt: Cell::new(0),
            tid: AtomicUsize::new(get_tid().get()),
        }
    }
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

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
    local.tid.store(0, Ordering::Release);
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
    if local.active_shared.load(Ordering::Acquire) == local.active_local.load(Ordering::Acquire) {
        #[inline]
        fn finish(local: *const Inner) {
            // there are no more external references, so local can be cleaned up immediately
            // FIXME: don't iterate the whole list but only check a single entry (and store said entry in the local itself)
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
        if *epoch >= EPOCH_OVERFLOW_GUARD {
            abort();
        }
        if *epoch >= EPOCH_WRAP_AROUND_GUARD {
            let mut curr = *epoch;
            loop {
                match GLOBAL_INFO.update_epoch.compare_exchange(
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

    match R::mode() {
        ReclamationMode::Eager => {
            try_cleanup(local_info, new);
        }
        ReclamationMode::Threshold { threshold } => {
            let curr_actions = local_info.action_cnt.get() + 1;
            if curr_actions >= threshold {
                try_cleanup(local_info, new);
                local_info.action_cnt.set(0);
            } else {
                local_info.action_cnt.set(curr_actions);
            }
        }
        ReclamationMode::Lazy => {
            let curr_actions = local_info.action_cnt.get() + 1;
            if curr_actions >= LAZY_THRESHOLD {
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
            // this means that after loading active_shared even is active_local gets decremented as much as it can it could only ever lead to either matching
            // the active_shared we observed (=> not in use) or being larger than active the active shared we observed (=> in use) which would only delay
            // us cleaning up the garbage which is okay
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
    let min_used = if min_used == usize::MAX {
        update_epoch
    } else {
        min_used
    };
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

    // we need to allow dead code here as if the `ptr_ops` feature is disabled, this function won't be used
    #[allow(dead_code)]
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

impl Drop for LocalPinGuard {
    #[inline]
    fn drop(&mut self) {
        // reduce local cnt safely even if src thread terminated
        let inner = unsafe { &*self.0 };

        let curr_id = get_tid().get();
        // fast path for thread local releases
        if unsafe { &*self.0 }.tid.load(Ordering::Relaxed) == curr_id {
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

mod conc_linked_list {
    use core::{
        mem::offset_of,
        ptr::null_mut,
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

        pub(crate) fn push_front_optimistic(&self, node: *mut ConcLinkedListNode<T>) {
            // this may be relaxed as we aren't accessing any data from curr and only need to confirm
            // that it is the current value of root (even after storing it into our next value)
            let curr = self.root.load(Ordering::Relaxed);
            *unsafe { &mut *node }.next.get_mut() = curr;
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
        }

        pub(crate) fn try_drain<F: FnMut(&T) -> bool>(&self, mut decision: F) -> bool {
            if self.is_empty() {
                return true;
            }
            match self
                .flags
                .compare_exchange(0, DRAIN_FLAG, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    let mut node_src = &self.root;
                    let mut node = node_src.load(Ordering::Acquire);
                    while !node.is_null() {
                        let rem = decision(&unsafe { &*node }.val);
                        if rem {
                            let next = unsafe { &*node }.next.load(Ordering::Acquire);
                            match node_src.compare_exchange(
                                node,
                                next,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
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
                    self.flags.store(0, Ordering::Release);
                    true
                }
                // there is another drain operation in progress
                Err(_) => false,
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
        pub const fn addr_offset() -> usize {
            offset_of!(Self, val)
        }
    }
}

mod ring_buf {
    use core::mem::MaybeUninit;

    pub(crate) struct RingBuffer<T, const LEN: usize> {
        buf: [MaybeUninit<T>; LEN],
        head: usize,
        len: usize,
    }

    impl<T, const LEN: usize> RingBuffer<T, LEN> {
        pub const fn new() -> Self {
            Self {
                // TODO: use uninit_array instead, once stabilized
                buf: unsafe { MaybeUninit::<[MaybeUninit<T>; LEN]>::uninit().assume_init() },
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

    impl<'a, T, const LEN: usize> ExactSizeIterator for Iter<'a, T, LEN> {
        #[inline]
        fn len(&self) -> usize {
            self.src.len() - self.idx
        }
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
