extern crate criterion;

use aarc::Arc as AArc;
use aarc::AtomicArc;
use aarc::Snapshot;
use arc_swap::ArcSwap;
use criterion::Criterion;
use rand::random;
use std::hint::{black_box, spin_loop};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use swap_arc::SwapArc;
use SwapEbr::SwapArc as SwapIt;

// NOTE: AArc seems to leak memory so comparing against it is a bit dangerous and can quickly lead to crashes

fn main() {
    let mut c = Criterion::default().configure_from_args();

    // ---

    c.bench_function("swap_ebr_con/destruct", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(SwapIt::new(Arc::new(5)));
            }
            start.elapsed()
        });
    });

    c.bench_function("swap_arc_con/destruct", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(SwapArc::new(Arc::new(5)));
            }
            start.elapsed()
        });
    });

    c.bench_function("arc_swap_con/destruct", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(ArcSwap::new(Arc::new(5)));
            }
            start.elapsed()
        });
    });

    /*c.bench_function("aarc_con/destruct", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(AtomicArc::new(Some(5)));
            }
            start.elapsed()
        });
    });*/

    // ---

    c.bench_function("swap_ebr_single_single", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_single_single", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_single_single", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_single_single", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            black_box(l1);
                        }
                    }));
                }
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Some(&AArc::new(random())), Ordering::Release);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_single_multi", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = black_box(tmp.load());
                            let l2 = black_box(tmp.load());
                            let l3 = black_box(tmp.load());
                            let l4 = black_box(tmp.load());
                            let l5 = black_box(tmp.load());
                            drop(black_box(l1));
                            drop(black_box(l2));
                            drop(black_box(l3));
                            drop(black_box(l4));
                            drop(black_box(l5));
                        }
                    }));
                }
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_single_multi", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = black_box(tmp.load());
                            let l2 = black_box(tmp.load());
                            let l3 = black_box(tmp.load());
                            let l4 = black_box(tmp.load());
                            let l5 = black_box(tmp.load());
                            drop(black_box(l1));
                            drop(black_box(l2));
                            drop(black_box(l3));
                            drop(black_box(l4));
                            drop(black_box(l5));
                        }
                    }));
                }
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_single_multi", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = black_box(tmp.load());
                            let l2 = black_box(tmp.load());
                            let l3 = black_box(tmp.load());
                            let l4 = black_box(tmp.load());
                            let l5 = black_box(tmp.load());
                            drop(black_box(l1));
                            drop(black_box(l2));
                            drop(black_box(l3));
                            drop(black_box(l4));
                            drop(black_box(l5));
                        }
                    }));
                }
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_single_multi", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = black_box(tmp.load::<Snapshot<i32>>(Ordering::Acquire));
                            let l2 = black_box(tmp.load::<Snapshot<i32>>(Ordering::Acquire));
                            let l3 = black_box(tmp.load::<Snapshot<i32>>(Ordering::Acquire));
                            let l4 = black_box(tmp.load::<Snapshot<i32>>(Ordering::Acquire));
                            let l5 = black_box(tmp.load::<Snapshot<i32>>(Ordering::Acquire));
                            drop(black_box(l1));
                            drop(black_box(l2));
                            drop(black_box(l3));
                            drop(black_box(l4));
                            drop(black_box(l5));
                        }
                    }));
                }
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Some(&AArc::new(rand::random())), Ordering::Release);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_multi_single", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                for _ in 0..20
                /*5*//*1*/
                {
                    // let send = send.clone();
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        // let send = send.clone();
                        for _ in 0..20000
                        /*200*/
                        {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_multi_single", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                for _ in 0..20
                /*5*//*1*/
                {
                    // let send = send.clone();
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        // let send = send.clone();
                        for _ in 0..20000
                        /*200*/
                        {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_multi_single", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                for _ in 0..20
                /*5*//*1*/
                {
                    // let send = send.clone();
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        // let send = send.clone();
                        for _ in 0..20000
                        /*200*/
                        {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_multi_single", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            black_box(l1);
                        }
                    }));
                }
                for _ in 0..20
                /*5*//*1*/
                {
                    // let send = send.clone();
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        // let send = send.clone();
                        for _ in 0..20000
                        /*200*/
                        {
                            tmp.store(Some(&AArc::new(rand::random())), Ordering::Release);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_multi_multi", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load();
                            let l2 = tmp.load();
                            let l3 = tmp.load();
                            let l4 = tmp.load();
                            let l5 = tmp.load();
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    }));
                }
                for _ in 0..20
                /*5*//*1*/
                {
                    // let send = send.clone();
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        // let send = send.clone();
                        for _ in 0..20000
                        /*200*/
                        {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_multi_multi", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load();
                            let l2 = tmp.load();
                            let l3 = tmp.load();
                            let l4 = tmp.load();
                            let l5 = tmp.load();
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    }));
                }
                for _ in 0..20
                /*5*//*1*/
                {
                    // let send = send.clone();
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        // let send = send.clone();
                        for _ in 0..20000
                        /*200*/
                        {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_multi_multi", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load();
                            let l2 = tmp.load();
                            let l3 = tmp.load();
                            let l4 = tmp.load();
                            let l5 = tmp.load();
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    }));
                }
                for _ in 0..20
                /*5*//*1*/
                {
                    // let send = send.clone();
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        // let send = send.clone();
                        for _ in 0..20000
                        /*200*/
                        {
                            tmp.store(Arc::new(rand::random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_multi_multi", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            let l2 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            let l3 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            let l4 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            let l5 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    }));
                }
                for _ in 0..20
                /*5*//*1*/
                {
                    // let send = send.clone();
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        // let send = send.clone();
                        for _ in 0..20000
                        /*200*/
                        {
                            tmp.store(Some(&AArc::new(rand::random())), Ordering::Release);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_read_heavy_single", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_read_heavy_single", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_read_heavy_single", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_read_heavy_single", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20
                /*5*//*1*/
                {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_read_light_single", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_read_light_single", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_read_light_single", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_read_light_single", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_read_light_multi", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_read_light_multi", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_read_light_multi", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_read_light_multi", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            black_box(l1);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_read_heavy_multi", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            let l2 = tmp.load();
                            let l3 = tmp.load();
                            let l4 = tmp.load();
                            let l5 = tmp.load();
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_read_heavy_multi", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            let l2 = tmp.load();
                            let l3 = tmp.load();
                            let l4 = tmp.load();
                            let l5 = tmp.load();
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_read_heavy_multi", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load();
                            let l2 = tmp.load();
                            let l3 = tmp.load();
                            let l4 = tmp.load();
                            let l5 = tmp.load();
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_read_heavy_multi", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            let l1 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            let l2 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            let l3 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            let l4 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            let l5 = tmp.load::<Snapshot<i32>>(Ordering::Acquire);
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_update_single", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_update_single", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_update_single", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("aarc_update_single", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..1 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Some(&AArc::new(random())), Ordering::Release);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    // ---

    c.bench_function("swap_ebr_update_multi", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("swap_arc_update_multi", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    c.bench_function("arc_swap_update_multi", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Arc::new(random()));
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });

    /*c.bench_function("aarc_update_multi", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            let mut diff = Duration::default();
            for _ in 0..iters {
                let started = Arc::new(AtomicBool::new(false));
                let mut threads = vec![];
                for _ in 0..20 {
                    let tmp = tmp.clone();
                    let started = started.clone();
                    threads.push(thread::spawn(move || {
                        while !started.load(Ordering::Acquire) {
                            spin_loop();
                        }
                        for _ in 0..20000 {
                            tmp.store(Some(&AArc::new(random())), Ordering::Release);
                        }
                    }));
                }
                let start = Instant::now();
                started.store(true, Ordering::Release);
                threads
                    .into_iter()
                    .for_each(|thread| thread.join().unwrap());
                diff += start.elapsed();
            }
            diff
        });
    });*/
}
