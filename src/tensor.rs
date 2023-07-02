use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

pub trait Shape: Debug {
    fn realized_shape() -> Vec<usize>;
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

impl<const A: usize, const B: usize, const C: usize> Shape for (Const<A>, Const<B>, Const<C>) {
    fn realized_shape() -> Vec<usize> {
        vec![A, B, C]
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
}

impl<S: Shape> Display for Tensor<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tensor {:?}", self.storage)?;
        Ok(())
    }
}

impl<S: Shape> AsRef<Tensor<S>> for Tensor<S> {
    fn as_ref(&self) -> &Tensor<S> {
        self
    }
}

// Elementwise operations
impl<S: Shape, T: AsRef<Tensor<S>>> Add<T> for Tensor<S> {
    type Output = Tensor<S>;

    fn add(mut self, rhs: T) -> Self::Output {
        for (ind, i) in self.storage.iter_mut().enumerate() {
            *i += rhs.as_ref().storage[ind];
        }

        self
    }
}

impl<S: Shape, T: AsRef<Tensor<S>>> Sub<T> for Tensor<S> {
    type Output = Tensor<S>;

    fn sub(mut self, rhs: T) -> Self::Output {
        for (ind, i) in self.storage.iter_mut().enumerate() {
            *i -= rhs.as_ref().storage[ind];
        }

        self
    }
}

impl<S: Shape, T: AsRef<Tensor<S>>> Mul<T> for Tensor<S> {
    type Output = Tensor<S>;

    fn mul(mut self, rhs: T) -> Self::Output {
        for (ind, i) in self.storage.iter_mut().enumerate() {
            *i *= rhs.as_ref().storage[ind];
        }

        self
    }
}

impl<S: Shape, T: AsRef<Tensor<S>>> Div<T> for Tensor<S> {
    type Output = Tensor<S>;

    fn div(mut self, rhs: T) -> Self::Output {
        for (ind, i) in self.storage.iter_mut().enumerate() {
            *i /= rhs.as_ref().storage[ind];
        }

        self
    }
}

// Exp and Log
pub trait Exp {
    fn exp(self) -> Self;
}

pub trait Log {
    fn log(self) -> Self;
}

impl<S: Shape> Exp for Tensor<S> {
    fn exp(mut self) -> Self {
        for i in &mut self.storage {
            *i = i.exp();
        }

        self
    }
}

impl<S: Shape> Log for Tensor<S> {
    fn log(mut self) -> Self {
        for i in &mut self.storage {
            *i = i.ln();
        }

        self
    }
}
