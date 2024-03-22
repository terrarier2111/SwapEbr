#![feature(sync_unsafe_cell)]
#![feature(thread_local)]
#![feature(cell_update)]
#![feature(vec_into_raw_parts)]

use std::{mem::transmute, sync::{atomic::Ordering, Arc}, thread};

use epoch::Guarded;

use crate::epoch::retire_explicit;

mod epoch;

fn main() {
    let cleanup_fn = cleanup_box::<String> as *const ();
    let cleanup_fn = unsafe { transmute(cleanup_fn) };

    let initial = Box::new("test".to_string());
    let second = Box::new("test2".to_string());
    let guard = Arc::new(Guarded::new(Box::into_raw(initial)));
    let pin = epoch::pin();
    let pinned = pin.load(&guard, Ordering::Acquire);
    println!("pinned: {}", unsafe { &*pinned });
    let move_guard = guard.clone();
    let other = thread::spawn(move || {
        let guard = move_guard;
        let pin = epoch::pin();
        for i in 0..100 {
            let i = 100 + i;
            let curr = Box::new(format!("test{i}"));
            let prev = pin.swap(&guard, Box::into_raw(curr), Ordering::AcqRel);
            unsafe { retire_explicit(prev, cleanup_fn);  }
        }
        println!("finished2");
    });
    for i in 0..100 {
        let curr = Box::new(format!("test{i}"));
        let prev = pin.swap(&guard, Box::into_raw(curr), Ordering::AcqRel);
        unsafe { retire_explicit(prev, cleanup_fn);  }
    }
    println!("finished1");

    other.join().unwrap();
    unsafe { retire_explicit(pin.load(&guard, Ordering::Acquire), cleanup_fn); }
}

unsafe fn cleanup_box<T>(ptr: *mut T) {
    let _ = Box::from_raw(ptr);
}
