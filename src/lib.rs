#![cfg_attr(
    feature = "no_std",
    no_std,
    allow(internal_features),
    feature(core_intrinsics),
    feature(thread_local)
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
pub use swap_arc::{ArcGuard, SwapArc};
pub use swap_arc_option::SwapArcOption;
pub use swap_box::{BoxGuard, SwapBox};
pub use swap_box_option::SwapBoxOption;
pub use swap_it::SwapIt;
pub use swap_it_option::SwapItOption;

mod epoch;

pub(crate) mod reclamation {
    use crate::epoch::LOCAL_PILE_SIZE;

    // TODO: expose this once adt_const_params got stabilized
    #[allow(dead_code)]
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

mod standard {
    use core::{ptr::NonNull, sync::atomic::Ordering};

    use crate::epoch::{pin, retire_explicit, Guarded, LocalPinGuard};

    pub(crate) struct Swap<T> {
        pub(crate) it: Guarded<T>,
    }

    impl<T> Swap<T> {
        #[inline(always)]
        pub const fn new(ptr: *const T) -> Self {
            Self {
                it: Guarded::new(ptr.cast_mut()),
            }
        }

        pub fn load(&self) -> SwapGuard<T> {
            let pin = pin();
            SwapGuard {
                val: unsafe {
                    NonNull::new_unchecked(pin.load(&self.it, Ordering::Acquire).cast_mut())
                },
                _guard: pin,
            }
        }

        pub fn store(&self, val: *const T, cleanup: fn(*mut T)) {
            let pin = pin();
            let old = pin.swap(&self.it, val, Ordering::AcqRel);
            unsafe {
                retire_explicit(old, cleanup);
            }
        }
    }

    #[cfg(feature = "ptr_ops")]
    impl<T> Swap<T> {
        pub fn compare_exchange(
            &self,
            current: *const T,
            new: *const T,
            cleanup: fn(*mut T),
        ) -> Result<(), *const T> {
            let pin = pin();
            match pin.compare_exchange(&self.it, current, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(val) => {
                    unsafe {
                        retire_explicit(val, cleanup);
                    }
                    Ok(())
                }
                Err(val) => Err(val),
            }
        }
    }

    pub(crate) struct SwapGuard<T> {
        pub(crate) val: NonNull<T>,
        pub(crate) _guard: LocalPinGuard,
    }

    unsafe impl<T> Send for SwapGuard<T> {}
    unsafe impl<T> Sync for SwapGuard<T> {}
}

mod standard_option {
    use core::{ptr::NonNull, sync::atomic::Ordering};

    use crate::{
        epoch::{pin, retire_explicit, Guarded},
        standard::SwapGuard,
    };

    pub(crate) struct SwapOption<T> {
        pub(crate) it: Guarded<T>,
    }

    impl<T> SwapOption<T> {
        #[inline(always)]
        pub const fn new(ptr: *const T) -> Self {
            Self {
                it: Guarded::new(ptr.cast_mut()),
            }
        }

        pub fn load(&self) -> Option<SwapGuard<T>> {
            let pin = pin();
            let ptr = pin.load(&self.it, Ordering::Acquire);
            if ptr.is_null() {
                return None;
            }
            Some(SwapGuard {
                val: unsafe { NonNull::new_unchecked(ptr.cast_mut()) },
                _guard: pin,
            })
        }

        pub fn store(&self, val: *const T, cleanup: fn(*mut T)) {
            let pin = pin();
            let old = pin.swap(&self.it, val, Ordering::AcqRel);
            if old.is_null() {
                // don't cleanup inexistant values
                return;
            }
            unsafe {
                retire_explicit(old, cleanup);
            }
        }
    }

    #[cfg(feature = "ptr_ops")]
    impl<T> SwapOption<T> {
        pub fn compare_exchange(
            &self,
            current: *const T,
            new: *const T,
            cleanup: fn(*mut T),
        ) -> Result<(), *const T> {
            let pin = pin();
            match pin.compare_exchange(&self.it, current, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(val) => {
                    if !val.is_null() {
                        unsafe {
                            retire_explicit(val, cleanup);
                        }
                    }
                    Ok(())
                }
                Err(val) => Err(val),
            }
        }
    }
}

mod swap_box {
    use core::ops::Deref;

    use crate::{
        cleanup_box,
        epoch::retire_explicit,
        standard::{Swap, SwapGuard},
    };

    pub struct SwapBox<T>(Swap<T>);

    impl<T> SwapBox<T> {
        pub fn new(val: Box<T>) -> Self {
            Self(Swap::new(Box::into_raw(val)))
        }

        pub fn load(&self) -> BoxGuard<T> {
            BoxGuard(self.0.load())
        }

        pub fn store(&self, val: Box<T>) {
            let ptr = Box::into_raw(val);
            self.0.store(ptr, cleanup_box::<T>);
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
            match self.0.compare_exchange(current, new_ptr, cleanup_box::<T>) {
                Ok(_) => Ok(()),
                Err(ptr) => Err((unsafe { Box::from_raw(new_ptr) }, ptr)),
            }
        }
    }

    impl<T> Drop for SwapBox<T> {
        fn drop(&mut self) {
            let guard = self.0.load();
            unsafe {
                retire_explicit(guard.val.as_ptr(), cleanup_box::<T>);
            }
        }
    }

    pub struct BoxGuard<T>(pub(crate) SwapGuard<T>);

    #[cfg(feature = "ptr_ops")]
    impl<T> BoxGuard<T> {
        #[inline(always)]
        pub fn get_ptr(&self) -> *const T {
            self.0.val
        }
    }

    impl<T> Deref for BoxGuard<T> {
        type Target = T;

        #[inline]
        fn deref(&self) -> &Self::Target {
            unsafe { self.0.val.as_ref() }
        }
    }
}

mod swap_box_option {
    use core::{
        ptr::{null, NonNull},
        sync::atomic::Ordering,
    };

    use crate::{
        cleanup_box,
        epoch::{pin, retire_explicit},
        standard::SwapGuard,
        standard_option::SwapOption,
        BoxGuard,
    };

    pub struct SwapBoxOption<T>(SwapOption<T>);

    impl<T> SwapBoxOption<T> {
        pub fn new(val: Option<Box<T>>) -> Self {
            let ptr = match val {
                Some(val) => Box::into_raw(val),
                None => null(),
            };
            Self(SwapOption::new(ptr))
        }

        #[inline]
        pub const fn new_empty() -> Self {
            Self(SwapOption::new(null()))
        }

        pub fn load(&self) -> Option<BoxGuard<T>> {
            let pin = pin();
            let ptr = pin.load(&self.0.it, Ordering::Acquire);
            if ptr.is_null() {
                return None;
            }
            Some(BoxGuard(SwapGuard {
                val: unsafe { NonNull::new_unchecked(ptr.cast_mut()) },
                _guard: pin,
            }))
        }

        pub fn store(&self, val: Box<T>) {
            let ptr = Box::into_raw(val);
            self.0.store(ptr, cleanup_box::<T>);
        }
    }

    #[cfg(feature = "ptr_ops")]
    impl<T> SwapBoxOption<T> {
        pub fn compare_exchange(
            &self,
            current: *const T,
            new: Box<T>,
        ) -> Result<(), (Box<T>, *const T)> {
            let new_ptr = Box::into_raw(new);
            match self.0.compare_exchange(current, new_ptr, cleanup_box::<T>) {
                Ok(_) => Ok(()),
                Err(ptr) => Err((unsafe { Box::from_raw(new_ptr) }, ptr)),
            }
        }
    }

    impl<T> Drop for SwapBoxOption<T> {
        fn drop(&mut self) {
            if let Some(guard) = self.0.load() {
                unsafe {
                    retire_explicit(guard.val.as_ptr(), cleanup_box::<T>);
                }
            }
        }
    }
}

mod swap_arc {
    use core::{mem::ManuallyDrop, ops::Deref, sync::atomic::Ordering};

    #[cfg(feature = "no_std")]
    use alloc::sync::Arc;
    #[cfg(not(feature = "no_std"))]
    use std::sync::Arc;

    use crate::{
        cleanup_arc,
        epoch::{pin, retire_explicit, LocalPinGuard},
        standard::Swap,
    };

    pub struct SwapArc<T>(Swap<T>);

    impl<T> SwapArc<T> {
        pub fn new(val: Arc<T>) -> Self {
            Self(Swap::new(Arc::into_raw(val)))
        }

        pub fn load(&self) -> ArcGuard<T> {
            let pin = pin();
            ArcGuard {
                it: ManuallyDrop::new(unsafe {
                    Arc::from_raw(pin.load(&self.0.it, Ordering::Acquire).cast_mut())
                }),
                _guard: pin,
            }
        }

        pub fn store(&self, val: Arc<T>) {
            let ptr = Arc::into_raw(val);
            self.0.store(ptr, cleanup_arc::<T>);
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
            match self.0.compare_exchange(current, new_ptr, cleanup_arc::<T>) {
                Ok(_) => Ok(()),
                Err(ptr) => Err((unsafe { Arc::from_raw(new_ptr) }, ptr)),
            }
        }
    }

    impl<T> Drop for SwapArc<T> {
        fn drop(&mut self) {
            let guard = self.0.load();
            unsafe {
                retire_explicit(guard.val.as_ptr(), cleanup_arc::<T>);
            }
        }
    }

    pub struct ArcGuard<T> {
        pub(crate) it: ManuallyDrop<Arc<T>>,
        pub(crate) _guard: LocalPinGuard,
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
}

mod swap_arc_option {
    use core::{mem::ManuallyDrop, ptr::null, sync::atomic::Ordering};

    #[cfg(feature = "no_std")]
    use alloc::sync::Arc;
    #[cfg(not(feature = "no_std"))]
    use std::sync::Arc;

    use crate::{
        cleanup_arc,
        epoch::{pin, retire_explicit},
        standard_option::SwapOption,
        ArcGuard,
    };

    pub struct SwapArcOption<T>(SwapOption<T>);

    impl<T> SwapArcOption<T> {
        pub fn new(val: Option<Arc<T>>) -> Self {
            Self(SwapOption::new(Self::val_to_ptr(val)))
        }

        #[inline]
        pub const fn new_empty() -> Self {
            Self(SwapOption::new(null()))
        }

        pub fn load(&self) -> Option<ArcGuard<T>> {
            let pin = pin();
            let ptr = pin.load(&self.0.it, Ordering::Acquire);
            if ptr.is_null() {
                return None;
            }
            Some(ArcGuard {
                it: ManuallyDrop::new(unsafe { Arc::from_raw(ptr) }),
                _guard: pin,
            })
        }

        pub fn store(&self, val: Option<Arc<T>>) {
            let ptr = Self::val_to_ptr(val);
            self.0.store(ptr, cleanup_arc::<T>);
        }

        fn val_to_ptr(val: Option<Arc<T>>) -> *const T {
            match val {
                Some(val) => Arc::into_raw(val),
                None => null(),
            }
        }
    }

    #[cfg(feature = "ptr_ops")]
    impl<T> SwapArcOption<T> {
        pub fn compare_exchange(
            &self,
            current: *const T,
            new: Arc<T>,
        ) -> Result<(), (Arc<T>, *const T)> {
            let new_ptr = Arc::into_raw(new);
            match self.0.compare_exchange(current, new_ptr, cleanup_arc::<T>) {
                Ok(_) => Ok(()),
                Err(err_ptr) => Err((unsafe { Arc::from_raw(new_ptr) }, err_ptr)),
            }
        }
    }

    impl<T> Drop for SwapArcOption<T> {
        fn drop(&mut self) {
            if let Some(guard) = self.0.load() {
                unsafe {
                    retire_explicit(guard.val.as_ptr(), cleanup_arc::<T>);
                }
            }
        }
    }
}

mod swap_it {
    use crate::{BoxGuard, SwapBox};

    pub struct SwapIt<T> {
        bx: SwapBox<T>,
    }

    impl<T> SwapIt<T> {
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
}

mod swap_it_option {
    use crate::{BoxGuard, SwapBoxOption};

    pub struct SwapItOption<T> {
        bx: SwapBoxOption<T>,
    }

    impl<T> SwapItOption<T> {
        pub fn new(val: Option<T>) -> Self {
            Self {
                bx: SwapBoxOption::new(val.map(|val| Box::new(val))),
            }
        }

        pub fn load(&self) -> Option<BoxGuard<T>> {
            self.bx.load()
        }

        pub fn store(&self, val: T) {
            self.bx.store(Box::new(val));
        }
    }

    #[cfg(feature = "ptr_ops")]
    impl<T> SwapItOption<T> {
        pub fn compare_exchange(&self, current: *const T, new: T) -> Result<(), (T, *const T)> {
            let new_ptr = Box::new(new);
            match self.bx.compare_exchange(current, new_ptr) {
                Ok(_) => Ok(()),
                Err(err) => Err((err.0.into_inner(), err.1)),
            }
        }
    }
}

fn cleanup_box<T>(ptr: *mut T) {
    let _ = unsafe { Box::from_raw(ptr) };
}

fn cleanup_arc<T>(ptr: *mut T) {
    let _ = unsafe { Arc::from_raw(ptr) };
}

// TODO: use this once `const_type_name` got stabilized
/// We use this count evaluation to store the Some() option count
/*const fn option_layers<T>() -> usize {
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
}*/

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
