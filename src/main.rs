#![feature(sync_unsafe_cell)]
#![feature(thread_local)]

use std::{
    cell::{Cell, SyncUnsafeCell},
    ptr::{null, null_mut},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
};

fn main() {
    println!("Hello, world!");
}

static GLOBAL_INFO: GlobalInfo = GlobalInfo {
    pile: ConcLinkedList::new(),
    epoch: AtomicUsize::new(0),
};

struct GlobalInfo {
    pile: ConcLinkedList<Instance>,
    epoch: AtomicUsize,
}

#[thread_local]
static LOCAL_INFO: LocalInfo = LocalInfo {
    pile: SyncUnsafeCell::new(vec![]),
    epoch: Cell::new(0),
};

#[thread_local]
static LOCAL_GUARD: LocalGuard = LocalGuard;

struct LocalInfo {
    pile: SyncUnsafeCell<Vec<Instance>>,
    epoch: Cell<usize>,
}

unsafe impl Send for LocalInfo {}
unsafe impl Sync for LocalInfo {}

struct LocalGuard;

impl Drop for LocalGuard {
    fn drop(&mut self) {
        // LOCAL_INFO
        for garbage in unsafe { LOCAL_INFO.pile.get().as_ref().unwrap() } {} // FIXME: check on acquiring ptr instead!
    }
}

struct Instance {
    drop_fn: fn(*mut ()),
    data_ptr: *const (),
    size: usize,
    epoch: usize,
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

pub fn pin() -> LocalPinGuard {
    let glob = GLOBAL_INFO.epoch.load(std::sync::atomic::Ordering::Acquire);
    let local_info = &LOCAL_INFO;
    let local = local_info.epoch.get();
    if glob == local {
        if GLOBAL_INFO.epoch.compare_exchange(glob, glob + 1, std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Relaxed).is_ok() {
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
        }
        // try cleanup global
        GLOBAL_INFO.pile.try_drain(|val| {
            val.epoch == local
        });
    }
    local_info.epoch.set(local + 1);
    LocalPinGuard
}

pub struct LocalPinGuard;

impl LocalPinGuard {

    pub fn defer<T>(&self, ptr: &Guarded<T>, order: Ordering) -> Option<&T> {
        // FIXME: should we increment local epoch here?
        unsafe { ptr.0.load(order).as_ref() }
    }

    // FIXME: provide a way to collect removed garbage

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

