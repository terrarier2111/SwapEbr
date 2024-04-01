use std::{
    hint::{black_box, spin_loop},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

use SwapEbr::SwapIt;

fn main() {
    let tmp = Arc::new(SwapIt::new(Arc::new(0)));
    for _ in 0..1000 {
        let started = Arc::new(AtomicBool::new(false));
        let mut threads = vec![];
        for _ in 0..1 {
            let tmp = tmp.clone();
            let started = started.clone();
            threads.push(thread::spawn(move || {
                while !started.load(Ordering::Acquire) {
                    spin_loop();
                }
                for _ in 0..200000 {
                    let l1 = tmp.load();
                    black_box(l1);
                }
            }));
        }
        started.store(true, Ordering::Release);
        threads
            .into_iter()
            .for_each(|thread| thread.join().unwrap());
    }
}
