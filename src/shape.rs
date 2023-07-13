#![allow(unused)]
use std::fmt::Debug;

pub type R1<const D: usize> = (Const<D>,);
pub type R2<const A: usize, const B: usize> = (Const<A>, Const<B>);
pub type R3<const A: usize, const B: usize, const C: usize> = (Const<A>, Const<B>, Const<C>);

pub trait Shape: Debug + Copy {
    type AddRight<const N: usize>: Shape;
    type AddLeft<const N: usize>: Shape;
    fn realized_shape() -> Vec<usize>;
    fn num_elements() -> usize {
        Self::realized_shape().iter().sum()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Const<const U: usize>;

impl<const A: usize> Shape for Const<A> {
    type AddRight<const N: usize> = R2<A, N>;
    type AddLeft<const N: usize> = R2<N, A>;
    fn realized_shape() -> Vec<usize> {
        vec![A]
    }
}

impl<const A: usize> Shape for (Const<A>,) {
    type AddRight<const N: usize> = R2<A, N>;
    type AddLeft<const N: usize> = R2<N, A>;
    fn realized_shape() -> Vec<usize> {
        vec![A]
    }
}

impl<const A: usize, const B: usize> Shape for (Const<A>, Const<B>) {
    type AddRight<const N: usize> = Const<A>;
    type AddLeft<const N: usize> = Const<A>;
    fn realized_shape() -> Vec<usize> {
        vec![A, B]
    }
}
