#![feature(sync_unsafe_cell)]
#![feature(thread_local)]

use std::{
    cell::{Cell, SyncUnsafeCell},
    ptr::{null, null_mut},
    sync::atomic::{AtomicPtr, AtomicUsize},
};

fn main() {
    println!("Hello, world!");
}

static GLOBAL_GARBAGE_PILE: GlobalInfo = GlobalInfo {
    pile: ConcLinkedList,
    epoch: todo!(),
};

struct GlobalInfo {
    pile: ConcLinkedList<Instance>,
    epoch: AtomicUsize,
}

#[thread_local]
static LOCAL_INFO: LocalInfo = LocalInfo {
    pile: SyncUnsafeCell::new(vec![]),
    epoch: SyncUnsafeCell::new(0),
};

#[thread_local]
static LOCAL_GUARD: LocalGuard = LocalGuard;

struct LocalInfo {
    pile: SyncUnsafeCell<Vec<Instance>>,
    epoch: SyncUnsafeCell<usize>,
}

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

struct ConcLinkedList<T> {
    root: AtomicPtr<ConcLinkedListNode<T>>,
    iter_cnt: AtomicUsize,
    // len: AtomicUsize,
}

const DRAIN_FLAG: usize = 1 << (usize::BITS as usize - 1);

impl<T> ConcLinkedList<T> {
    fn new() -> Self {
        Self {
            root: AtomicPtr::new(null_mut()),
            iter_cnt: AtomicUsize::new(0),
        }
    }

    fn is_empty(&self) -> bool {
        self.root.load(std::sync::atomic::Ordering::Acquire).is_null()
    }

    fn is_draining(&self) -> bool {
        self.iter_cnt.load(std::sync::atomic::Ordering::Acquire) & DRAIN_FLAG != 0
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
        match self.iter_cnt.compare_exchange(0, DRAIN_FLAG, std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Relaxed) {
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
                self.iter_cnt.fetch_sub(DRAIN_FLAG, std::sync::atomic::Ordering::Release); // FIXME: can we make this a store by removing the iterator support?
                true
            },
            Err(_) => false,
        }

    }

    // FIXME: add drain!

    #[inline]
    fn iter(&self) -> ConcListIter<T> {
        self.iter_cnt.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        // FIXME: check if removal is happening and wait if so
        ConcListIter {
            next_lookup: &self.root,
            list: self,
        }
    }
}

#[repr(align(4))]
struct ConcLinkedListNode<T> {
    next: AtomicPtr<ConcLinkedListNode<T>>,
    val: T,
}

struct ConcListIter<'a, T> {
    next_lookup: &'a AtomicPtr<ConcLinkedListNode<T>>,
    list: &'a ConcLinkedList<T>,
}

impl<'a, T> Iterator for ConcListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next_lookup.load(std::sync::atomic::Ordering::Acquire);
        unsafe { next.as_ref() }.map(|node| {
            self.next_lookup = &node.next;
            &node.val
        })
    }
}

impl<'a, T> Drop for ConcListIter<'a, T> {
    fn drop(&mut self) {
        self.list.iter_cnt.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

