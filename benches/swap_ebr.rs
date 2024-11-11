extern crate criterion;

use aarc::AtomicArc;
use aarc::Snapshot;
use arc_swap::ArcSwap;
use criterion::Criterion;
use rand::random;
use std::hint::{black_box, spin_loop};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use swap_arc::SwapArc;
use swap_it::SwapArc as SwapIt;

// NOTE: AArc seems to leak memory so comparing against it is a bit dangerous and can quickly lead to crashes

// FIXME: fix duplicated tests!

fn main() {
    let mut c = Criterion::default().configure_from_args();

    // ---

    c.bench_function("swap_ebr_con/destruct", |b| {
        b.iter(|| {
            let _ = black_box(SwapIt::new(Arc::new(5)));
        });
    });

    c.bench_function("swap_arc_con/destruct", |b| {
        b.iter(|| {
            let _ = black_box(SwapArc::new(Arc::new(5)));
        });
    });

    c.bench_function("arc_swap_con/destruct", |b| {
        b.iter(|| {
            let _ = black_box(ArcSwap::new(Arc::new(5)));
        });
    });

    c.bench_function("aarc_con/destruct", |b| {
        b.iter(|| {
            let _ = black_box(AtomicArc::new(Some(5)));
        });
    });

    // ---

    c.bench_function("swap_ebr_read_single_write_none_one", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter(|| {
            let l1 = tmp.load();
            drop(black_box(l1));
            black_box(());
        });
    });

    c.bench_function("swap_arc_read_single_write_none_one", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter(|| {
            let l1 = tmp.load();
            drop(black_box(l1));
            black_box(());
        });
    });

    c.bench_function("arc_swap_read_single_write_none_one", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter(|| {
            let l1 = tmp.load();
            drop(black_box(l1));
            black_box(());
        });
    });

    c.bench_function("aarc_read_single_write_none_one", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter(|| {
            let l1 = tmp.load::<Snapshot<i32>>();
            drop(black_box(l1));
            black_box(());
        });
    });

    // ---

    c.bench_function("swap_ebr_read_single_write_none_many", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter(|| {
            let l1 = tmp.load();
            let l2 = tmp.load();
            let l3 = tmp.load();
            let l4 = tmp.load();
            let l5 = tmp.load();
            drop(black_box(l1));
            black_box(());
            drop(black_box(l2));
            black_box(());
            drop(black_box(l3));
            black_box(());
            drop(black_box(l4));
            black_box(());
            drop(black_box(l5));
            black_box(());
        });
    });

    c.bench_function("swap_arc_read_single_write_none_many", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter(|| {
            let l1 = tmp.load();
            let l2 = tmp.load();
            let l3 = tmp.load();
            let l4 = tmp.load();
            let l5 = tmp.load();
            drop(black_box(l1));
            black_box(());
            drop(black_box(l2));
            black_box(());
            drop(black_box(l3));
            black_box(());
            drop(black_box(l4));
            black_box(());
            drop(black_box(l5));
            black_box(());
        });
    });

    c.bench_function("arc_swap_read_single_write_none_many", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter(|| {
            let l1 = tmp.load();
            let l2 = tmp.load();
            let l3 = tmp.load();
            let l4 = tmp.load();
            let l5 = tmp.load();
            drop(black_box(l1));
            black_box(());
            drop(black_box(l2));
            black_box(());
            drop(black_box(l3));
            black_box(());
            drop(black_box(l4));
            black_box(());
            drop(black_box(l5));
            black_box(());
        });
    });

    c.bench_function("aarc_read_single_write_none_many", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter(|| {
            let l1 = tmp.load::<Snapshot<i32>>();
            let l2 = tmp.load::<Snapshot<i32>>();
            let l3 = tmp.load::<Snapshot<i32>>();
            let l4 = tmp.load::<Snapshot<i32>>();
            let l5 = tmp.load::<Snapshot<i32>>();
            drop(black_box(l1));
            black_box(());
            drop(black_box(l2));
            black_box(());
            drop(black_box(l3));
            black_box(());
            drop(black_box(l4));
            black_box(());
            drop(black_box(l5));
            black_box(());
        });
    });

    // ---

    c.bench_function("swap_ebr_read_multi_write_none_one", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..200000 {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                None,
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_multi_write_none_one", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..200000 {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                None,
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_multi_write_none_one", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..200000 {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                None,
                tmp.clone(),
            )
        });
    });

    c.bench_function("aarc_read_multi_write_none_one", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(
                    multi_threads(),
                    |tmp: Arc<AtomicArc<i32>>| {
                        for _ in 0..200000 {
                            let l1 = tmp.load::<Snapshot<i32>>();
                            black_box(l1);
                        }
                    },
                )),
                None,
                tmp.clone(),
            )
        });
    });

    // ---

    c.bench_function("swap_ebr_read_multi_write_single_one", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..200000 {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(1, |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_multi_write_single_one", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..200000 {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(1, |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_multi_write_single_one", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..200000 {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(1, |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("aarc_read_read_multi_write_single_one", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(
                    multi_threads(),
                    |tmp: Arc<AtomicArc<i32>>| {
                        for _ in 0..200000 {
                            let l1 = tmp.load::<Snapshot<i32>>();
                            black_box(l1);
                        }
                    },
                )),
                Some(Operation::new(1, |tmp: Arc<AtomicArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Some(&Arc::new(random())));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    // ---

    c.bench_function("swap_ebr_read_multi_write_none_many", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..200000 {
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
                })),
                None,
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_multi_write_none_many", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..200000 {
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
                })),
                None,
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_multi_write_none_many", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..200000 {
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
                })),
                None,
                tmp.clone(),
            )
        });
    });

    c.bench_function("aarc_read_multi_write_none_many", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(
                    multi_threads(),
                    |tmp: Arc<AtomicArc<i32>>| {
                        for _ in 0..200000 {
                            let l1 = tmp.load::<Snapshot<i32>>();
                            let l2 = tmp.load::<Snapshot<i32>>();
                            let l3 = tmp.load::<Snapshot<i32>>();
                            let l4 = tmp.load::<Snapshot<i32>>();
                            let l5 = tmp.load::<Snapshot<i32>>();
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    },
                )),
                None,
                tmp.clone(),
            )
        });
    });

    // ---

    c.bench_function("swap_ebr_read_single_write_single_one", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(1, |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000
                    /*200*/
                    {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(1, |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_single_write_single_one", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(1, |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000
                    /*200*/
                    {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(1, |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_single_write_single_one", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(1, |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000
                    /*200*/
                    {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(1, |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("aarc_read_single_write_single_one", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(1, |tmp: Arc<AtomicArc<i32>>| {
                    for _ in 0..20000
                    /*200*/
                    {
                        let l1 = tmp.load::<Snapshot<i32>>();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(1, |tmp: Arc<AtomicArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Some(&Arc::new(random())));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    // ---

    c.bench_function("swap_ebr_read_single_write_single_many", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(1, |tmp: Arc<SwapIt<i32>>| {
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
                })),
                Some(Operation::new(1, |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_single_write_single_many", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(1, |tmp: Arc<SwapArc<i32>>| {
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
                })),
                Some(Operation::new(1, |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_single_write_single_many", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(1, |tmp: Arc<ArcSwap<i32>>| {
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
                })),
                Some(Operation::new(1, |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("aarc_read_single_write_single_many", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(1, |tmp: Arc<AtomicArc<i32>>| {
                    for _ in 0..20000
                    /*200*/
                    {
                        let l1 = tmp.load::<Snapshot<i32>>();
                        let l2 = tmp.load::<Snapshot<i32>>();
                        let l3 = tmp.load::<Snapshot<i32>>();
                        let l4 = tmp.load::<Snapshot<i32>>();
                        let l5 = tmp.load::<Snapshot<i32>>();
                        black_box(l1);
                        black_box(l2);
                        black_box(l3);
                        black_box(l4);
                        black_box(l5);
                    }
                })),
                Some(Operation::new(1, |tmp: Arc<AtomicArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Some(&Arc::new(random())));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    // ---

    c.bench_function("swap_ebr_read_multi_write_multi_one", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000
                    /*200*/
                    {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_multi_write_multi_one", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000
                    /*200*/
                    {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_multi_write_multi_one", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000
                    /*200*/
                    {
                        let l1 = tmp.load();
                        black_box(l1);
                    }
                })),
                Some(Operation::new(multi_threads(), |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("aarc_read_multi_write_multi_one", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(
                    multi_threads(),
                    |tmp: Arc<AtomicArc<i32>>| {
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load::<Snapshot<i32>>();
                            black_box(l1);
                        }
                    },
                )),
                Some(Operation::new(
                    multi_threads(),
                    |tmp: Arc<AtomicArc<i32>>| {
                        for _ in 0..20000 {
                            tmp.store(Some(&Arc::new(random())));
                        }
                    },
                )),
                tmp.clone(),
            )
        });
    });

    // ---

    c.bench_function("swap_ebr_read_multi_write_multi_many", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapIt<i32>>| {
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
                })),
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_multi_write_multi_many", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapArc<i32>>| {
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
                })),
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_multi_write_multi_many", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(multi_threads(), |tmp: Arc<ArcSwap<i32>>| {
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
                })),
                Some(Operation::new(multi_threads(), |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("aarc_read_multi_write_multi_many", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                Some(Operation::new(
                    multi_threads(),
                    |tmp: Arc<AtomicArc<i32>>| {
                        for _ in 0..20000
                        /*200*/
                        {
                            let l1 = tmp.load::<Snapshot<i32>>();
                            let l2 = tmp.load::<Snapshot<i32>>();
                            let l3 = tmp.load::<Snapshot<i32>>();
                            let l4 = tmp.load::<Snapshot<i32>>();
                            let l5 = tmp.load::<Snapshot<i32>>();
                            black_box(l1);
                            black_box(l2);
                            black_box(l3);
                            black_box(l4);
                            black_box(l5);
                        }
                    },
                )),
                Some(Operation::new(
                    multi_threads(),
                    |tmp: Arc<AtomicArc<i32>>| {
                        for _ in 0..20000 {
                            tmp.store(Some(&Arc::new(random())));
                        }
                    },
                )),
                tmp.clone(),
            )
        });
    });

    // ---

    c.bench_function("swap_ebr_read_none_write_single_one", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                None,
                Some(Operation::new(1, |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_none_write_single_one", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                None,
                Some(Operation::new(1, |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_none_write_single_one", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                None,
                Some(Operation::new(1, |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("aarc_read_none_write_single_one", |b| {
        let tmp = Arc::new(AtomicArc::new(Some(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                None,
                Some(Operation::new(1, |tmp: Arc<AtomicArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Some(&Arc::new(random())));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    // ---

    c.bench_function("swap_ebr_read_none_write_multi_one", |b| {
        let tmp = Arc::new(SwapIt::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                None,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapIt<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("swap_arc_read_none_write_multi_one", |b| {
        let tmp = Arc::new(SwapArc::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                None,
                Some(Operation::new(multi_threads(), |tmp: Arc<SwapArc<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
        });
    });

    c.bench_function("arc_swap_read_none_write_multi_one", |b| {
        let tmp = Arc::new(ArcSwap::new(Arc::new(0)));
        b.iter_custom(|iters| {
            measure(
                iters,
                None,
                Some(Operation::new(multi_threads(), |tmp: Arc<ArcSwap<i32>>| {
                    for _ in 0..20000 {
                        tmp.store(Arc::new(random()));
                    }
                })),
                tmp.clone(),
            )
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

fn multi_threads() -> usize {
    20
}

trait ShareFn<T: Send + Sync + 'static>: Fn(T) + Send + Sync + 'static {}

impl<T: Send + Sync + 'static, F: Fn(T) + Send + Sync + 'static> ShareFn<T> for F {}

struct Operation<T: Send + Sync + 'static> {
    threads: usize,
    op: Arc<dyn ShareFn<Arc<T>>>,
    _phantom_data: PhantomData<T>,
}

impl<T: Send + Sync + 'static> Operation<T> {
    fn new<F: Fn(Arc<T>) + Send + Sync + 'static>(threads: usize, op: F) -> Self {
        Self {
            threads,
            op: Arc::new(op),
            _phantom_data: PhantomData,
        }
    }
}

fn measure<T: Send + Sync + 'static>(
    iters: u64,
    read: Option<Operation<T>>,
    write: Option<Operation<T>>,
    val: Arc<T>,
) -> Duration {
    let mut diff = Duration::default();
    for _ in 0..iters {
        let started = Arc::new(AtomicBool::new(false));
        let mut threads = vec![];
        if let Some(read) = read.as_ref() {
            for _ in 0..read.threads {
                let tmp = val.clone();
                let op = read.op.clone();
                let started = started.clone();
                threads.push(thread::spawn(move || {
                    while !started.load(Ordering::Acquire) {
                        spin_loop();
                    }
                    let val = tmp;
                    op(val);
                }));
            }
        }
        if let Some(write) = write.as_ref() {
            for _ in 0..write.threads {
                let tmp = val.clone();
                let op = write.op.clone();
                let started = started.clone();
                threads.push(thread::spawn(move || {
                    while !started.load(Ordering::Acquire) {
                        spin_loop();
                    }
                    let val = tmp;
                    op(val);
                }));
            }
        }
        let start = Instant::now();
        started.store(true, Ordering::Release);
        threads
            .into_iter()
            .for_each(|thread| thread.join().unwrap());
        diff += start.elapsed();
    }
    diff
}
