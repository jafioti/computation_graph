use std::fmt::Debug;

pub trait Shape: Debug + Copy {
    fn realized_shape() -> Vec<usize>;
    fn num_elements() -> usize {
        Self::realized_shape().iter().sum()
    }
}

#[derive(Debug, Clone, Copy)]
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
