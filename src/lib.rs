#![feature(thread_local)]
#![feature(const_type_name)]
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
use epoch::{pin, retire_explicit, Guarded, LocalPinGuard};

mod epoch;

pub struct SwapIt<T> {
    it: Guarded<T>,
}

impl<T> SwapIt<T> {
    /// We use this count evaluation to store the Some() option count
    const OPTION_LAYERS: usize = {
        const OPTION_NAME: &str = "core::option::Option<";

        let ty_name = core::any::type_name::<T>();
        let mut curr_idx = 0;
        let mut layers = 0;
        'outer: loop {
            let end = curr_idx + OPTION_NAME.len();
            let mut curr = curr_idx;
            while curr < end {
                if ty_name.as_bytes()[curr] != OPTION_NAME.as_bytes()[curr - curr_idx] {
                    break 'outer;
                }
                curr += 1;
            }
            curr_idx = end;
            layers += 1;
        }
        layers
    };

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
        unsafe {
            retire_explicit(old, cleanup_box::<T>);
        }
    }
}

impl<T> Drop for SwapIt<T> {
    fn drop(&mut self) {
        let pin = pin();
        let curr = pin.load(&self.it, Ordering::Acquire);
        unsafe {
            retire_explicit(curr, cleanup_box::<T>);
        }
    }
}

fn cleanup_box<T>(ptr: *mut T) {
    let _ = unsafe { Box::from_raw(ptr) };
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
    #[cfg(feature = "no_std")]
    use alloc::sync::Arc;
    #[cfg(not(feature = "no_std"))]
    use std::sync::Arc;

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
