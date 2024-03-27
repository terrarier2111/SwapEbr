use std::{hint::black_box, sync::Arc};

use SwapEbr::SwapIt;

fn main() {
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
            for i in 0..200 {
                tmp.store(Arc::new(!(9 + i)));
            }
        }));
    }
    threads
        .into_iter()
        .for_each(|thread| thread.join().unwrap());
}
