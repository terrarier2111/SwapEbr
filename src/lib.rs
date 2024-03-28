#![feature(thread_local)]
#![feature(const_type_name)]
#![cfg_attr(
    feature = "no_std",
    no_std,
    allow(internal_features),
    feature(core_intrinsics)
)]

cfg_if! {
    if #[cfg(not(feature = "no_std"))] {
        use std::sync::Arc;
    } else {
        extern crate alloc;
        use alloc::boxed::Box;
        use alloc::sync::Arc;
    }
}

use cfg_if::cfg_if;
use core::{mem::ManuallyDrop, ops::Deref, sync::atomic::Ordering};
use epoch::{pin, retire_explicit, Guarded, LocalPinGuard};

mod epoch;

pub(crate) mod reclamation {
    use crate::epoch::LOCAL_PILE_SIZE;

    // TODO: expose this once adt_const_params got stabilized
    pub(crate) enum ReclamationMode {
        // reclamation happens whenever possible (after storing a new value, after updating the local epoch, etc.)
        // every single action triggers a reclamation attempt, there is no threshold to be crossed.
        // => this mode is the least cpu-efficient mode as it reclaims garbage as often as possible
        Eager,
        // reclamation happens if a certain amount of possibly reclaiming actions happened such as
        // (storing a new value, after updating the local epoch, etc.)
        // the memory and cpu efficiency of this mode will strongly depend on the choice of the threshold parameter
        Threshold { threshold: usize },
        // reclamation only happens rarely after updating the local epoch and never after storing a new value
        // in this mode only performing store operations from a single thread and no load operations
        // will lead to its local garbage pile to fill up completely and only loading threads will slowly
        // cleanup generated garbage through global
        // => this mode is the most cpu-efficient mode as it reclaims garbage only rarely
        Lazy,
    }

    // FIXME: lazy isn't currently as lazy as it could be, make it attempt to cleanup the writer's pile once its completely filled up and have a really
    // large threshold for readers

    impl ReclamationMode {
        /// this mode is the same as Threshold with some predefined sane threshold value
        #[allow(non_upper_case_globals)]
        pub(crate) const Balanced: ReclamationMode = ReclamationMode::Threshold {
            threshold: LOCAL_PILE_SIZE / 8,
        };
    }

    pub(crate) const RECLAMATION_MODE: ReclamationMode = ReclamationMode::Balanced;
    pub(crate) const LAZY_THRESHOLD: usize = LOCAL_PILE_SIZE;
}

pub struct SwapBox<T> {
    it: Guarded<T>,
}

impl<T> SwapBox<T> {
    const OPTION_LAYERS: usize = option_layers::<T>();

    pub fn new(val: Box<T>) -> Self {
        Self {
            it: Guarded::new(Box::into_raw(val)),
        }
    }

    pub fn load(&self) -> BoxGuard<T> {
        let pin = pin();
        BoxGuard {
            it: pin.load(&self.it, Ordering::Acquire),
            _guard: pin,
        }
    }

    pub fn store(&self, val: Box<T>) {
        let ptr = Box::into_raw(val);
        let pin = pin();
        let old = pin.swap(&self.it, ptr, Ordering::AcqRel);
        unsafe {
            retire_explicit(old, cleanup_box::<T>);
        }
    }
}

#[cfg(feature = "ptr_ops")]
impl<T> SwapBox<T> {
    pub fn compare_exchange(
        &self,
        current: *const T,
        new: Box<T>,
    ) -> Result<(), (Box<T>, *const T)> {
        let new_ptr = Box::into_raw(new);
        let pin = pin();
        match pin.compare_exchange(
            &self.it,
            current,
            new_ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(val) => {
                unsafe {
                    retire_explicit(val, cleanup_box::<T>);
                }
                Ok(())
            }
            Err(val) => Err((unsafe { Box::from_raw(new_ptr) }, val)),
        }
    }
}

impl<T> Drop for SwapBox<T> {
    fn drop(&mut self) {
        let pin = pin();
        let curr = pin.load(&self.it, Ordering::Acquire);
        unsafe {
            retire_explicit(curr, cleanup_box::<T>);
        }
    }
}

pub struct BoxGuard<T> {
    _guard: LocalPinGuard,
    it: *const T,
}

#[cfg(feature = "ptr_ops")]
impl<T> BoxGuard<T> {
    #[inline(always)]
    pub fn get_ptr(&self) -> *const T {
        self.it
    }
}

impl<T> Deref for BoxGuard<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.it }
    }
}

unsafe impl<T> Send for BoxGuard<T> {}
unsafe impl<T> Sync for BoxGuard<T> {}

pub struct SwapArc<T> {
    it: Guarded<T>,
}

impl<T> SwapArc<T> {
    const OPTION_LAYERS: usize = option_layers::<T>();

    pub fn new(val: Arc<T>) -> Self {
        Self {
            it: Guarded::new(Arc::into_raw(val).cast_mut()),
        }
    }

    pub fn load(&self) -> ArcGuard<T> {
        let pin = pin();
        ArcGuard {
            it: ManuallyDrop::new(unsafe {
                Arc::from_raw(pin.load(&self.it, Ordering::Acquire).cast_mut())
            }),
            _guard: pin,
        }
    }

    pub fn store(&self, val: Arc<T>) {
        let ptr = Arc::into_raw(val);
        let pin = pin();
        let old = pin.swap(&self.it, ptr, Ordering::AcqRel);
        unsafe {
            retire_explicit(old, cleanup_arc::<T>);
        }
    }
}

#[cfg(feature = "ptr_ops")]
impl<T> SwapArc<T> {
    pub fn compare_exchange(
        &self,
        current: *const T,
        new: Arc<T>,
    ) -> Result<(), (Arc<T>, *const T)> {
        let new_ptr = Arc::into_raw(new);
        let pin = pin();
        match pin.compare_exchange(
            &self.it,
            current,
            new_ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(val) => {
                unsafe {
                    retire_explicit(val, cleanup_arc::<T>);
                }
                Ok(())
            }
            Err(val) => Err((unsafe { Arc::from_raw(new_ptr) }, val)),
        }
    }
}

impl<T> Drop for SwapArc<T> {
    fn drop(&mut self) {
        let pin = pin();
        let curr = pin.load(&self.it, Ordering::Acquire);
        unsafe {
            retire_explicit(curr, cleanup_arc::<T>);
        }
    }
}

pub struct ArcGuard<T> {
    _guard: LocalPinGuard,
    it: ManuallyDrop<Arc<T>>,
}

impl<T> Deref for ArcGuard<T> {
    type Target = Arc<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.it.deref()
    }
}

unsafe impl<T> Send for ArcGuard<T> {}
unsafe impl<T> Sync for ArcGuard<T> {}

pub struct SwapIt<T> {
    bx: SwapBox<T>,
}

impl<T> SwapIt<T> {
    const OPTION_LAYERS: usize = option_layers::<T>();

    pub fn new(val: T) -> Self {
        Self {
            bx: SwapBox::new(Box::new(val)),
        }
    }

    pub fn load(&self) -> BoxGuard<T> {
        self.bx.load()
    }

    pub fn store(&self, val: T) {
        self.bx.store(Box::new(val));
    }
}

#[cfg(feature = "ptr_ops")]
impl<T> SwapIt<T> {
    pub fn compare_exchange(&self, current: *const T, new: T) -> Result<(), (T, *const T)> {
        let new_ptr = Box::new(new);
        match self.bx.compare_exchange(current, new_ptr) {
            Ok(_) => Ok(()),
            Err(err) => Err((err.0.into_inner(), err.1)),
        }
    }
}

fn cleanup_box<T>(ptr: *mut T) {
    let _ = unsafe { Box::from_raw(ptr) };
}

fn cleanup_arc<T>(ptr: *mut T) {
    let _ = unsafe { Arc::from_raw(ptr) };
}

/// We use this count evaluation to store the Some() option count
const fn option_layers<T>() -> usize {
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
}

#[cfg(all(test, miri))]
mod test {
    #[cfg(feature = "no_std")]
    use alloc::sync::Arc;
    #[cfg(not(feature = "no_std"))]
    use std::sync::Arc;

    use core::hint::black_box;

    use crate::SwapIt;

    #[test]
    fn test_new_miri() {
        black_box(SwapIt::new(Arc::new(3)));
    }

    #[test]
    fn test_load_miri() {
        let swap_it = black_box(SwapIt::new(Arc::new(3)));
        black_box(swap_it.load());
    }

    #[test]
    fn test_store_miri() {
        let swap_it = black_box(SwapIt::new(Arc::new(3)));
        swap_it.store(Arc::new(6));
        black_box(swap_it);
    }

    #[test]
    fn test_load_store_miri() {
        let swap_it = black_box(SwapIt::new(Arc::new(3)));
        let load = black_box(swap_it.load());
        swap_it.store(Arc::new(6));
        black_box(swap_it);
    }

    #[test]
    fn test_load_multi_miri() {
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
