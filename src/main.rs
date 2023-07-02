use crate::{graph::*, tensor::*};
use std::ops::{Add, Sub};

#[macro_use]
mod graph;
mod tensor;

fn main() {
    let a: Tensor<Const<10_000>> = Tensor::new(vec![1.; 10_000]);
    let b: Tensor<Const<10_000>> = Tensor::new(vec![1.; 10_000]);
    let c: Tensor<Const<10_000>> = Tensor::new(vec![1.; 10_000]);

    let mut graph = Graph::new(a).add(&b).log().exp().exp().log().sub(&c);

    println!("Unoptimized Graph: {graph}");
    graph.debug_optimize();
    println!("Optimized Graph: {graph}");

    graph.compute();
}
