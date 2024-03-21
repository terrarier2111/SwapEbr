use std::{
    alloc::{alloc, Layout}, cell::{Cell, SyncUnsafeCell}, process::abort, ptr::null_mut, sync::atomic::{AtomicPtr, AtomicUsize, Ordering}
};

static GLOBAL_INFO: GlobalInfo = GlobalInfo {
    pile: ConcLinkedList::new(),
    epoch: AtomicUsize::new(1),
    update_epoch: AtomicUsize::new(0),
    threads: AtomicUsize::new(0),
};

struct GlobalInfo {
    pile: ConcLinkedList<Instance>,
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
            for garbage in unsafe { local.pile.get().as_ref().unwrap() } {} // FIXME: check on acquiring ptr instead!
        }
    }
}

struct Instance {
    drop_fn: fn(*mut ()),
    data_ptr: *const (),
    epoch: usize,
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
        let new = GLOBAL_INFO.epoch.fetch_add(1, Ordering::AcqRel);
        local_info.epoch.set(new + 1);

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
        GLOBAL_INFO.pile.try_drain(|val| {
            val.epoch == local
        });
    }
    LocalPinGuard(local_info as *const Inner)
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
            local.active_local.store(local.active_local.load(Ordering::Relaxed), Ordering::Release);
            return;
        }

        // FIXME: handle non-local releases
    }
}

pub struct Guarded<T>(AtomicPtr<T>);

struct ConcLinkedList<T> {
    root: AtomicPtr<ConcLinkedListNode<T>>,
    flags: AtomicUsize,
    // len: AtomicUsize,
}

const DRAIN_FLAG: usize = 1 << (usize::BITS as usize - 1);

impl<T> ConcLinkedList<T> {
    const fn new() -> Self {
        Self {
            root: AtomicPtr::new(null_mut()),
            flags: AtomicUsize::new(0),
        }
    }

    fn is_empty(&self) -> bool {
        self.root.load(std::sync::atomic::Ordering::Acquire).is_null()
    }

    fn is_draining(&self) -> bool {
        self.flags.load(std::sync::atomic::Ordering::Acquire) & DRAIN_FLAG != 0
    }

    fn push_front(&self, val: T) {
        let node = Box::into_raw(Box::new(ConcLinkedListNode {
            next: AtomicPtr::new(null_mut()),
            val,
        }));
        let mut curr = self.root.load(std::sync::atomic::Ordering::Acquire);
        loop {
            unsafe { node.as_ref().unwrap() }
                .next
                .store(curr, std::sync::atomic::Ordering::Release);
            match self.root.compare_exchange(
                curr,
                node,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            ) {
                Ok(_) => {}
                Err(new) => {
                    curr = new;
                }
            }
        }
    }

    fn try_drain<F: Fn(&T) -> bool>(&self, decision: F) -> bool {
        match self.flags.compare_exchange(0, DRAIN_FLAG, std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Relaxed) {
            Ok(_) => {
                let mut next = &self.root;
                loop {
                    let ptr = next.load(std::sync::atomic::Ordering::Acquire);
                    match unsafe { ptr.as_ref() } {
                        Some(node) => {
                            let rem = decision(&node.val);
                            if rem {
                                let next_ptr = node.next.load(std::sync::atomic::Ordering::Acquire);
                                match next.compare_exchange(ptr, next_ptr, std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Relaxed) {
                                    Ok(_) => break,
                                    Err(mut node) => {
                                        loop {
                                            let new_next = unsafe { node.as_ref().unwrap() }.next.load(std::sync::atomic::Ordering::Acquire);
                                            if new_next == ptr {
                                                &unsafe { node.as_ref().unwrap() }.next.store(next_ptr, std::sync::atomic::Ordering::Release);
                                                break;
                                            }
                                            node = new_next;
                                        }
                                    },
                                }
                            }
                            next = &node.next;
                        },
                        None => break,
                    }
                }
                self.flags.store(0, std::sync::atomic::Ordering::Release);
                true
            },
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

