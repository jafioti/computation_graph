use crate::{graph::*, tensor::*};
use std::ops::{Add, Sub};

#[macro_use]
mod graph;
mod new;
// mod reg;
mod tensor;

fn main() {
    new::main();
}

// struct Linear<const I: usize, const O: usize> {
//     weight: Tensor<(Const<I>, Const<O>)>,
// }

// impl<'a, const I: usize, const O: usize> Linear<I, O> {
//     fn new() -> Self {
//         Self {
//             weight: Tensor::new(vec![1.0; I * O]),
//         }
//     }

//     fn forward(&'a self, input: Graph<'a, Const<I>, Const<I>>) -> Graph<'a, Const<I>, Const<O>> {
//         input.matmul(&self.weight)
//     }
// }
