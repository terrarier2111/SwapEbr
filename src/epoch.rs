use std::{
    alloc::{alloc, dealloc, Layout},
    cell::{Cell, SyncUnsafeCell},
    process::abort,
    ptr::{self, null_mut},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
};

use self::conc_linked_list::ConcLinkedList;

static GLOBAL_INFO: GlobalInfo = GlobalInfo {
    pile: ConcLinkedList::new(),
    locals_pile: ConcLinkedList::new(),
    epoch: AtomicUsize::new(1),
    update_epoch: AtomicUsize::new(0),
    threads: AtomicUsize::new(0),
};

struct GlobalInfo {
    pile: ConcLinkedList<Instance>,
    locals_pile: ConcLinkedList<*const Inner>,
    epoch: AtomicUsize,
    update_epoch: AtomicUsize,
    threads: AtomicUsize,
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
    tid: usize, // we use the ptr to this inner storage as our tid as it will only ever exist once and will only ever have the same (or no) owner as long as this Inner is allocated
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

struct LocalGuard;

impl Drop for LocalGuard {
    fn drop(&mut self) {
        if let Some(local) = try_get_local() {
            let glob_epoch = GLOBAL_INFO.epoch.load(Ordering::Acquire);
            let glob_threads = GLOBAL_INFO.threads.load(Ordering::Acquire);
            for garbage in unsafe { local.pile.get().as_ref().unwrap() } {
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
                unsafe {
                    ptr::drop_in_place(local as *const Inner as *mut Inner);
                }
                unsafe {
                    dealloc(local as *const Inner as *mut u8, Layout::new::<Inner>());
                }
            } else {
                // there are still external references alive, so delay cleanup until they die
                GLOBAL_INFO.locals_pile.push_front(local as *const Inner);
            }
            GLOBAL_INFO.threads.fetch_sub(1, Ordering::Release);
        }
    }
}

#[repr(transparent)]
struct DropWrapper<T>(T);

impl<T> Drop for DropWrapper<T> {
    fn drop(&mut self) {}
}

#[derive(Clone)]
struct Instance {
    drop_fn: Option<unsafe fn(*mut ())>,
    data_ptr: *const (),
    epoch: usize,
}

impl Instance {
    fn new<T>(val: *const T, epoch: usize) -> Self {
        Self {
            drop_fn: if core::mem::needs_drop::<T>() {
                let drop_fn = core::ptr::drop_in_place::<T> as *const ();
                Some(unsafe { core::mem::transmute::<_, fn(*mut ())>(drop_fn) })
            } else {
                None
            },
            data_ptr: val.cast::<()>(),
            epoch,
        }
    }

    #[inline]
    unsafe fn cleanup(&self) {
        if let Some(drop_fn) = self.drop_fn {
            drop_fn(self.data_ptr.cast_mut());
        }
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
                pile: SyncUnsafeCell::new(vec![]),
                epoch: Cell::new(GLOBAL_INFO.epoch.load(Ordering::Acquire)),
                active_local: AtomicUsize::new(0),
                active_shared: AtomicUsize::new(0),
                tid: src as usize,
            });
            *src = alloc;
            GLOBAL_INFO.threads.fetch_add(1, Ordering::Release);
            LOCAL_GUARD.with(|_| {});
            alloc.as_ref().unwrap_unchecked()
        }
    }

    if let Some(local) = try_get_local() {
        return local;
    }

    alloc_local()
}

/// This may not be called if there is already a pin active
pub(crate) fn pin() -> LocalPinGuard {
    let local_info = get_local();
    // this is relaxed as we know that we are the only thread currently to modify any local data
    let active = local_info.active_local.load(Ordering::Relaxed);
    if active != 0 {
        local_info.active_local.store(active + 1, Ordering::Release);
        return LocalPinGuard(local_info as *const Inner);
    }
    let update_epoch = GLOBAL_INFO.update_epoch.load(Ordering::Acquire);
    let local = local_info.epoch.get();
    if update_epoch > local {
        // update epoch
        let new = GLOBAL_INFO.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        local_info.epoch.set(new);

        let local_pile = unsafe { &mut *local_info.pile.get() };
        let mut rem_nodes = 0;
        for item in local_pile.iter() {
            if item.epoch == local {
                rem_nodes += 1;
            } else {
                break;
            }
        }
        for _ in 0..rem_nodes {
            local_pile.remove(0);
        }
        // try cleanup global
        GLOBAL_INFO.pile.try_drain(|val| val.epoch == local);
    }
    LocalPinGuard(local_info as *const Inner)
}

pub(crate) fn retire<T>(val: *const T) {
    let local_ptr = local_ptr();
    unsafe {
        (&mut *(&*local_ptr).pile.get()).push(Instance::new(val, (&*local_ptr).epoch.get()));
    }
    let new_epoch = GLOBAL_INFO.epoch.fetch_add(1, Ordering::AcqRel) + 1;
    unsafe { &*local_ptr }.epoch.set(new_epoch);

    let mut curr_epoch = GLOBAL_INFO.update_epoch.load(Ordering::Acquire);
    loop {
        match GLOBAL_INFO.update_epoch.compare_exchange(curr_epoch, new_epoch, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => break,
            Err(next_epoch) => {
                if next_epoch > new_epoch {
                    // fail as there was a more recent update than us
                    break;
                }
                // retry setting our epoch
                curr_epoch = next_epoch;
            },
        }
    }
}

pub struct LocalPinGuard(*const Inner);

unsafe impl Sync for LocalPinGuard {}
unsafe impl Send for LocalPinGuard {}

impl LocalPinGuard {
    pub fn defer<T>(&self, ptr: &Guarded<T>, order: Ordering) -> Option<&T> {
        unsafe { ptr.0.load(order).as_ref() }
    }

    // FIXME: provide a way to collect removed garbage
}

impl Drop for LocalPinGuard {
    fn drop(&mut self) {
        // reduce local cnt safely even if src thread terminated

        let local_ptr = local_ptr();
        // fast path for thread local releases
        if self.0 == local_ptr {
            let local = unsafe { &*local_ptr };
            local.active_local.store(
                local.active_local.load(Ordering::Relaxed),
                Ordering::Release,
            );
            return;
        }

        // reduce shared count
        unsafe { &*self.0 }.active_shared.fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct Guarded<T>(AtomicPtr<T>);

mod conc_linked_list {
    use std::{
        ptr::null_mut,
        sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    };

    pub(crate) struct ConcLinkedList<T> {
        root: AtomicPtr<ConcLinkedListNode<T>>,
        flags: AtomicUsize,
        // len: AtomicUsize,
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
                    Ok(_) => {}
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
