use std::{
    hint::{black_box, spin_loop},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use swap_it::SwapArc;

fn main() {
    let mut dur = Duration::default();
    let tmp = Arc::new(SwapArc::new(Arc::new(0)));
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
        let start = Instant::now();
        threads
            .into_iter()
            .for_each(|thread| thread.join().unwrap());
        dur += Instant::now().duration_since(start);
    }
    println!("dur: {}", dur.as_millis());
}
