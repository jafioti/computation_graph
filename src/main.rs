use crate::{graph::*, tensor::*};
use std::ops::{Add, Sub};

#[macro_use]
mod graph;
mod tensor;

fn main() {
    let a: Tensor<Const<10_000>> = Tensor::new(vec![1.; 10_000]);
    let b: Tensor<Const<10_000>> = Tensor::new(vec![1.; 10_000]);
    let c: Tensor<Const<10_000>> = Tensor::new(vec![1.; 10_000]);
    let lin = Linear::<10_000, 5_000>::new();

    let graph = a.graph().add(&b).log().exp().exp().log().sub(&c);
    let mut graph = lin.forward(graph);

    graph.debug_optimize();

    graph.compute();
}

struct Linear<const I: usize, const O: usize> {
    weight: Tensor<(Const<I>, Const<O>)>,
}

impl<'a, const I: usize, const O: usize> Linear<I, O> {
    fn new() -> Self {
        Self {
            weight: Tensor::new(vec![1.0; I * O]),
        }
    }

    fn forward(&'a self, input: Graph<'a, Const<I>, Const<I>>) -> Graph<'a, Const<I>, Const<O>> {
        input.matmul(&self.weight)
    }
}
