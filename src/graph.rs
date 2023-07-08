use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

use strum::IntoStaticStr;

use crate::tensor::*;

#[derive(Debug, IntoStaticStr)]
pub enum GraphOp<'a> {
    Add(Vec<usize>, &'a Vec<f32>),
    Sub(Vec<usize>, &'a Vec<f32>),
    Mul(Vec<usize>, &'a Vec<f32>),
    Div(Vec<usize>, &'a Vec<f32>),
    Log,
    Exp,
    VecMatMul(Vec<usize>, &'a Vec<f32>),
    // Add and subtract
    AddSub(Vec<usize>, &'a Vec<f32>, Vec<usize>, &'a Vec<f32>),
}

pub struct Graph<'a, A: Shape, S: Shape> {
    start: Tensor<A>,
    ops: Vec<GraphOp<'a>>,
    _phantom: PhantomData<S>,
}

impl<'a, A: Shape, S: Shape> Display for Graph<'a, A, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Start")?;
        if !self.ops.is_empty() {
            write!(f, " -> ")?;
        }
        for (i, op) in self.ops.iter().enumerate() {
            let s: &'static str = op.into();
            write!(f, "{s}")?;
            if i < self.ops.len() - 1 {
                write!(f, " -> ")?;
            }
        }
        Ok(())
    }
}

impl<'a, A: Shape> Graph<'a, A, A> {
    pub fn new(inp: Tensor<A>) -> Graph<'a, A, A> {
        Graph {
            start: inp,
            ops: vec![],
            _phantom: Default::default(),
        }
    }
}

impl<'a, A: Shape, S: Shape> Graph<'a, A, S> {
    fn reform<N: Shape>(self) -> Graph<'a, A, N> {
        Graph {
            start: self.start,
            ops: self.ops,
            _phantom: Default::default(),
        }
    }

    /// Execute the graph. `optimize()` should be called before this.
    pub fn compute(self) -> Tensor<S> {
        let mut running_tensor = self.start.storage;
        for op in self.ops {
            match op {
                GraphOp::Add(_, d) => {
                    for (ind, i) in running_tensor.iter_mut().enumerate() {
                        *i += d[ind];
                    }
                }
                GraphOp::Sub(_, d) => {
                    for (ind, i) in running_tensor.iter_mut().enumerate() {
                        *i -= d[ind];
                    }
                }
                GraphOp::Mul(_, d) => {
                    for (ind, i) in running_tensor.iter_mut().enumerate() {
                        *i *= d[ind];
                    }
                }
                GraphOp::Div(_, d) => {
                    for (ind, i) in running_tensor.iter_mut().enumerate() {
                        *i /= d[ind];
                    }
                }
                GraphOp::Log => {
                    running_tensor.iter_mut().for_each(|i| *i = i.ln());
                }
                GraphOp::Exp => {
                    running_tensor.iter_mut().for_each(|i| *i = i.exp());
                }
                GraphOp::AddSub(_, d, _, b) => {
                    for (ind, i) in running_tensor.iter_mut().enumerate() {
                        *i += d[ind] - b[ind];
                    }
                }
                GraphOp::VecMatMul(_, _) => {}
            }
        }

        Tensor::new(running_tensor)
    }

    // Run a single pass of optimization on the graph, returns true if any optimizations were found
    pub fn optimize_pass(&mut self) -> bool {
        let mut found_opt = false;
        // Check for combining adds and subtracts
        for i in crate::check_adjacent_ops!(&self.ops, GraphOp::Add(_, _), GraphOp::Sub(_, _))
            .into_iter()
            .rev()
        {
            let GraphOp::Add(sa, a) = self.ops.remove(i) else {panic!()};
            let GraphOp::Sub(sb, b) = self.ops.remove(i) else {panic!()};
            self.ops.insert(i, GraphOp::AddSub(sa, a, sb, b));
            found_opt = true;
        }
        // Look for adjacent log and exp's
        for i in crate::check_adjacent_ops!(&self.ops, GraphOp::Exp, GraphOp::Log)
            .into_iter()
            .rev()
        {
            self.ops.remove(i);
            self.ops.remove(i);
            found_opt = true;
        }
        for i in crate::check_adjacent_ops!(&self.ops, GraphOp::Log, GraphOp::Exp)
            .into_iter()
            .rev()
        {
            self.ops.remove(i);
            self.ops.remove(i);
            found_opt = true;
        }
        found_opt
    }

    /// Fully optimize graph
    pub fn optimize(&mut self) {
        while self.optimize_pass() {}
    }

    /// Fully optimize graph and print the graph at each step
    pub fn debug_optimize(&mut self) {
        let mut pass = 0;
        println!("Before Optimization: {self}");
        while self.optimize_pass() {
            pass += 1;
            println!("Pass {pass}: {self}");
        }
        println!("Finished optimization after {pass} passes");
    }
}

#[macro_export]
macro_rules! check_adjacent_ops {
    ($ops: expr, $op1: pat, $op2: pat) => {{
        let mut found = vec![];
        let mut skip_next = false;
        for i in 0..$ops.len() {
            if skip_next {
                skip_next = false;
                continue;
            }
            if i + 1 < $ops.len() && (matches!(&$ops[i], $op1) && matches!(&$ops[i + 1], $op2)) {
                found.push(i);
                skip_next = true;
            }
        }

        found
    }};
}

// impl<'a, A: Shape, S: Shape + 'a> Add<&'a Tensor<S>> for Graph<'a, A, S> {
//     type Output = Graph<'a, A, S>;

//     fn add(mut self, rhs: &'a Tensor<S>) -> Self::Output {
//         self.ops
//             .push(GraphOp::Add(S::realized_shape(), &rhs.storage));
//         self
//     }
// }

// impl<'a, A: Shape, S: Shape> Sub<&'a Tensor<S>> for Graph<'a, A, S> {
//     type Output = Graph<'a, A, S>;

//     fn sub(mut self, rhs: &'a Tensor<S>) -> Self::Output {
//         self.ops
//             .push(GraphOp::Sub(S::realized_shape(), &rhs.storage));
//         self
//     }
// }

// impl<'a, A: Shape, S: Shape> Mul<&'a Tensor<S>> for Graph<'a, A, S> {
//     type Output = Graph<'a, A, S>;

//     fn mul(mut self, rhs: &'a Tensor<S>) -> Self::Output {
//         self.ops
//             .push(GraphOp::Mul(S::realized_shape(), &rhs.storage));
//         self
//     }
// }

// impl<'a, A: Shape, S: Shape> Div<&'a Tensor<S>> for Graph<'a, A, S> {
//     type Output = Graph<'a, A, S>;

//     fn div(mut self, rhs: &'a Tensor<S>) -> Self::Output {
//         self.ops
//             .push(GraphOp::Div(S::realized_shape(), &rhs.storage));
//         self
//     }
// }

// // Exp and Log
// pub trait Exp {
//     fn exp(self) -> Self;
// }

// pub trait Log {
//     fn log(self) -> Self;
// }

// impl<'a, A: Shape, S: Shape> Exp for Graph<'a, A, S> {
//     fn exp(mut self) -> Self {
//         self.ops.push(GraphOp::Exp);
//         self
//     }
// }

// impl<'a, A: Shape, S: Shape> Log for Graph<'a, A, S> {
//     fn log(mut self) -> Self {
//         self.ops.push(GraphOp::Log);
//         self
//     }
// }

// impl<'a, A: Shape, const I: usize> Graph<'a, A, Const<I>> {
//     pub fn matmul<const O: usize>(
//         mut self,
//         mat: &'a Tensor<(Const<I>, Const<O>)>,
//     ) -> Graph<'a, A, Const<O>> {
//         self.ops.push(GraphOp::VecMatMul(
//             <(Const<I>, Const<O>)>::realized_shape(),
//             &mat.storage,
//         ));
//         self.reform()
//     }
// }
