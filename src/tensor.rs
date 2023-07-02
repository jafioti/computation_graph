use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

use crate::graph::Graph;

pub trait Shape: Debug {
    fn realized_shape() -> Vec<usize>;
    fn num_elements() -> usize {
        Self::realized_shape().iter().sum()
    }
}

#[derive(Debug)]
pub struct Const<const U: usize>;

impl<const A: usize> Shape for Const<A> {
    fn realized_shape() -> Vec<usize> {
        vec![A]
    }
}

impl<const A: usize> Shape for (Const<A>,) {
    fn realized_shape() -> Vec<usize> {
        vec![A]
    }
}

impl<const A: usize, const B: usize> Shape for (Const<A>, Const<B>) {
    fn realized_shape() -> Vec<usize> {
        vec![A, B]
    }
}

#[derive(Debug)]
pub struct Tensor<S: Shape> {
    pub(crate) storage: Vec<f32>,
    _phantom: PhantomData<S>,
}

impl<S: Shape> Tensor<S> {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            storage: data,
            _phantom: Default::default(),
        }
    }

    pub fn graph<'a>(self) -> Graph<'a, S, S> {
        Graph::new(self)
    }
}

impl<S: Shape> Display for Tensor<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tensor {:?}", self.storage)?;
        Ok(())
    }
}
