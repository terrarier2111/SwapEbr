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

use core::{mem::ManuallyDrop, ops::Deref, ptr::NonNull};

use cfg_if::cfg_if;
use standard::Swap;

mod epoch;

use standard_option::SwapOption;
pub use swap_it::SwapIt;
pub use swap_it_option::SwapItOption;

pub type SwapArc<T> = Swap<Arc<T>, T>;
pub type SwapArcOption<T> = SwapOption<Arc<T>, T>;
pub type SwapBox<T> = Swap<Box<T>, T>;
pub type SwapBoxOption<T> = SwapOption<Box<T>, T>;

pub fn new_unsized<T: ?Sized>(val: Box<T>) -> Swap<Box<Box<T>>, Box<T>> {
    Swap::new(Box::new(val))
}

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
    use core::{marker::PhantomData, ops::Deref, ptr::NonNull, sync::atomic::Ordering};

    use crate::{
        cleanup,
        epoch::{pin, retire_explicit, Guarded, LocalPinGuard},
        PtrConvert,
    };

    pub struct Swap<T: PtrConvert<U>, U> {
        pub(crate) it: Guarded<U>,
        _phantom_data: PhantomData<T>,
    }

    impl<T: PtrConvert<U>, U> Swap<T, U> {
        #[inline]
        pub fn new(val: T) -> Self {
            Self {
                it: Guarded::new(T::into_ptr(val).as_ptr()),
                _phantom_data: PhantomData,
            }
        }

        pub fn load(&self) -> SwapGuard<T, U> {
            let pin = pin();
            SwapGuard {
                val: T::guard_val(unsafe {
                    NonNull::new_unchecked(pin.load(&self.it, Ordering::Acquire).cast_mut())
                }),
                _guard: pin,
                _phantom_data: PhantomData,
            }
        }

        pub fn store(&self, val: T) {
            let pin = pin();
            let old = pin.swap(&self.it, val.into_ptr().as_ptr(), Ordering::AcqRel);
            unsafe {
                retire_explicit(NonNull::new_unchecked(old.cast_mut()), cleanup::<T, U>);
            }
        }
    }

    #[cfg(feature = "ptr_ops")]
    impl<T: PtrConvert<U>, U> Swap<T, U> {
        pub fn compare_exchange(&self, current: *const U, new: T) -> Result<(), (T, *const U)> {
            let pin = pin();
            let new_ptr = new.into_ptr();
            match pin.compare_exchange(
                &self.it,
                current,
                new_ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(val) => {
                    unsafe {
                        retire_explicit(val, cleanup::<T, U>);
                    }
                    Ok(())
                }
                Err(val) => Err((unsafe { T::from_ptr(new_ptr) }, val)),
            }
        }
    }

    impl<T: PtrConvert<U>, U> Drop for Swap<T, U> {
        fn drop(&mut self) {
            let pin = pin();
            unsafe {
                retire_explicit(
                    NonNull::new_unchecked(pin.load(&self.it, Ordering::Acquire).cast_mut()),
                    cleanup::<T, U>,
                );
            }
        }
    }

    pub struct SwapGuard<T: PtrConvert<U>, U> {
        pub(crate) val: T::GuardVal,
        pub(crate) _guard: LocalPinGuard,
        pub(crate) _phantom_data: PhantomData<U>,
    }

    impl<T: PtrConvert<U>, U> Deref for SwapGuard<T, U> {
        type Target = T::GuardDeref;

        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            T::deref_guard(&self.val)
        }
    }

    #[cfg(feature = "ptr_ops")]
    impl<T: PtrConvert<U>, U> SwapGuard<T, U> {
        #[inline(always)]
        pub fn get_ptr(&self) -> NonNull<U> {
            T::guard_to_ptr(&self.val)
        }
    }

    unsafe impl<T: PtrConvert<U>, U> Send for SwapGuard<T, U> {}
    unsafe impl<T: PtrConvert<U>, U> Sync for SwapGuard<T, U> {}
}

mod standard_option {
    use core::{
        marker::PhantomData,
        ptr::{null_mut, NonNull},
        sync::atomic::Ordering,
    };

    use crate::{
        cleanup,
        epoch::{pin, retire_explicit, Guarded},
        standard::SwapGuard,
        PtrConvert,
    };

    pub struct SwapOption<T: PtrConvert<U>, U> {
        pub(crate) it: Guarded<U>,
        _phantom_data: PhantomData<T>,
    }

    impl<T: PtrConvert<U>, U> SwapOption<T, U> {
        #[inline]
        pub fn new(val: Option<T>) -> Self {
            let ptr = match val {
                Some(val) => val.into_ptr().as_ptr(),
                None => null_mut(),
            };
            Self {
                it: Guarded::new(ptr),
                _phantom_data: PhantomData,
            }
        }

        #[inline(always)]
        pub const fn new_empty() -> Self {
            Self {
                it: Guarded::new(null_mut()),
                _phantom_data: PhantomData,
            }
        }

        pub fn load(&self) -> Option<SwapGuard<T, U>> {
            let pin = pin();
            let ptr = pin.load(&self.it, Ordering::Acquire);
            if ptr.is_null() {
                return None;
            }
            Some(SwapGuard {
                val: T::guard_val(unsafe { NonNull::new_unchecked(ptr.cast_mut()) }),
                _guard: pin,
                _phantom_data: PhantomData,
            })
        }

        pub fn store(&self, val: T) {
            let ptr = val.into_ptr();
            let pin = pin();
            let old = pin.swap(&self.it, ptr.as_ptr(), Ordering::AcqRel);
            if old.is_null() {
                // don't cleanup inexistant values
                return;
            }
            unsafe {
                retire_explicit(NonNull::new_unchecked(old.cast_mut()), cleanup::<T, U>);
            }
        }
    }

    #[cfg(feature = "ptr_ops")]
    impl<T> SwapOption<T> {
        pub fn compare_exchange(&self, current: *const T, new: *const T) -> Result<(), *const T> {
            let pin = pin();
            match pin.compare_exchange(&self.it, current, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(val) => {
                    if !val.is_null() {
                        unsafe {
                            retire_explicit(val, cleanup::<T, U>);
                        }
                    }
                    Ok(())
                }
                Err(val) => Err(val),
            }
        }
    }

    impl<T: PtrConvert<U>, U> Drop for SwapOption<T, U> {
        fn drop(&mut self) {
            let pin = pin();
            let ptr = pin.load(&self.it, Ordering::Acquire).cast_mut();
            if ptr.is_null() {
                return;
            }
            unsafe {
                retire_explicit(NonNull::new_unchecked(ptr), cleanup::<T, U>);
            }
        }
    }
}

mod swap_it {
    use crate::{standard::SwapGuard, SwapBox};

    pub struct SwapIt<T> {
        bx: SwapBox<T>,
    }

    impl<T> SwapIt<T> {
        pub fn new(val: T) -> Self {
            Self {
                bx: SwapBox::new(Box::new(val)),
            }
        }

        pub fn load(&self) -> SwapGuard<Box<T>, T> {
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
    use crate::{standard::SwapGuard, SwapBoxOption};

    pub struct SwapItOption<T> {
        bx: SwapBoxOption<T>,
    }

    impl<T> SwapItOption<T> {
        pub fn new(val: Option<T>) -> Self {
            Self {
                bx: SwapBoxOption::new(val.map(|val| Box::new(val))),
            }
        }

        #[inline]
        pub const fn new_empty() -> Self {
            Self {
                bx: SwapBoxOption::new_empty(),
            }
        }

        pub fn load(&self) -> Option<SwapGuard<Box<T>, T>> {
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

fn cleanup<T: PtrConvert<U>, U>(ptr: NonNull<U>) {
    let _ = unsafe { T::from_ptr(ptr) };
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

pub trait PtrConvert<T: ?Sized>: Sized {
    type GuardVal;
    type GuardDeref;

    unsafe fn from_ptr(ptr: NonNull<T>) -> Self;

    fn into_ptr(self) -> NonNull<T>;

    fn guard_val(ptr: NonNull<T>) -> Self::GuardVal;

    fn guard_to_ptr(guard: &Self::GuardVal) -> NonNull<T>;

    fn deref_guard(guard: &Self::GuardVal) -> &Self::GuardDeref;
}

impl<T> PtrConvert<T> for Box<T> {
    type GuardVal = NonNull<T>;
    type GuardDeref = T;

    #[inline(always)]
    unsafe fn from_ptr(ptr: NonNull<T>) -> Self {
        Box::from_raw(ptr.as_ptr())
    }

    #[inline(always)]
    fn into_ptr(self) -> NonNull<T> {
        unsafe { NonNull::new_unchecked(Box::into_raw(self)) }
    }

    #[inline(always)]
    fn guard_val(ptr: NonNull<T>) -> Self::GuardVal {
        ptr
    }

    #[inline(always)]
    fn deref_guard(guard: &Self::GuardVal) -> &Self::GuardDeref {
        unsafe { guard.as_ref() }
    }

    #[inline(always)]
    fn guard_to_ptr(guard: &Self::GuardVal) -> NonNull<T> {
        *guard
    }
}

impl<T> PtrConvert<T> for Arc<T> {
    type GuardVal = ManuallyDrop<Arc<T>>;
    type GuardDeref = Arc<T>;

    #[inline(always)]
    unsafe fn from_ptr(ptr: NonNull<T>) -> Self {
        unsafe { Arc::from_raw(ptr.as_ptr()) }
    }

    #[inline(always)]
    fn into_ptr(self) -> NonNull<T> {
        unsafe { NonNull::new_unchecked(Arc::into_raw(self).cast_mut()) }
    }

    #[inline(always)]
    fn guard_val(ptr: NonNull<T>) -> Self::GuardVal {
        ManuallyDrop::new(unsafe { Arc::from_raw(ptr.as_ptr()) })
    }

    #[inline(always)]
    fn deref_guard(guard: &Self::GuardVal) -> &Self::GuardDeref {
        guard.deref()
    }

    #[inline(always)]
    fn guard_to_ptr(guard: &Self::GuardVal) -> NonNull<T> {
        unsafe { NonNull::new_unchecked((guard.as_ref() as *const T).cast_mut()) }
    }
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
