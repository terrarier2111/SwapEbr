#![feature(sync_unsafe_cell)]
#![feature(thread_local)]
#![feature(cell_update)]
#![feature(vec_into_raw_parts)]
#![cfg_attr(
    feature = "no_std",
    no_std,
    allow(internal_features),
    feature(core_intrinsics)
)]

#[cfg(feature = "no_std")]
extern crate alloc;

use core::{ops::Deref, sync::atomic::Ordering};

#[cfg(feature = "no_std")]
use alloc::boxed::Box;
use epoch::{pin, Guarded, LocalPinGuard};

mod epoch;

pub struct SwapIt<T> {
    it: Guarded<T>,
}

impl<T> SwapIt<T> {
    pub fn new(val: T) -> Self {
        Self {
            it: Guarded::new(Box::into_raw(Box::new(val))),
        }
    }

    pub fn load(&self) -> Guard<T> {
        let pin = pin();
        Guard {
            it: pin.load(&self.it, Ordering::Acquire),
            _guard: pin,
        }
    }

    pub fn store(&self, val: T) {
        let ptr = Box::into_raw(Box::new(val));
        let pin = pin();
        let old = pin.swap(&self.it, ptr, Ordering::AcqRel);
        let _ = unsafe { Box::from_raw(old.cast_mut()) };
    }
}

impl<T> Drop for SwapIt<T> {
    fn drop(&mut self) {
        let pin = pin();
        let curr = pin.load(&self.it, Ordering::Acquire);
        let _ = unsafe { Box::from_raw(curr.cast_mut()) };
    }
}

pub struct Guard<T> {
    _guard: LocalPinGuard,
    it: *const T,
}

impl<T> Deref for Guard<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.it }
    }
}

unsafe impl<T> Send for Guard<T> {}
unsafe impl<T> Sync for Guard<T> {}

#[cfg(all(test, miri))]
mod test {
    use alloc::sync::Arc;

    use crate::SwapIt;

    #[test]
    fn test_load_multi_miri() {
        use core::hint::black_box;
        use std::thread;
        let tmp = Arc::new(SwapIt::new(Arc::new(3)));
        let mut threads = vec![];
        for _ in 0..4 {
            let tmp = tmp.clone();
            threads.push(thread::spawn(move || {
                for _ in 0..200 {
                    let l1 = tmp.load();
                    black_box(l1);
                }
            }));
        }
        for _ in 0..4 {
            let tmp = tmp.clone();
            threads.push(thread::spawn(move || {
                for _ in 0..200 {
                    tmp.store(Arc::new(rand::random()));
                }
            }));
        }
        threads
            .into_iter()
            .for_each(|thread| thread.join().unwrap());
    }
}
