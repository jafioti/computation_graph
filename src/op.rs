use std::fmt::Debug;

use crate::new::Tensor;

pub trait Operator: Debug {
    fn name(&self) -> &'static str;
    fn process(&self, inp: Vec<&Tensor>) -> Tensor;
}

#[derive(Debug, Clone)]
pub struct Input;
impl Operator for Input {
    fn name(&self) -> &'static str {
        "Input"
    }
    fn process(&self, _: Vec<&Tensor>) -> Tensor {
        panic!("You Fool.")
    }
}

#[derive(Debug, Clone)]
pub struct Add;
impl Operator for Add {
    fn name(&self) -> &'static str {
        "Add"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for (a, b) in t.data.iter_mut().zip(tensors[1].data.iter()) {
            *a += b;
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Subtract;
impl Operator for Subtract {
    fn name(&self) -> &'static str {
        "Subtract"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for (a, b) in t.data.iter_mut().zip(tensors[1].data.iter()) {
            *a -= b;
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Multiply;
impl Operator for Multiply {
    fn name(&self) -> &'static str {
        "Multiply"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for (a, b) in t.data.iter_mut().zip(tensors[1].data.iter()) {
            *a *= b;
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Division;
impl Operator for Division {
    fn name(&self) -> &'static str {
        "Divide"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for (a, b) in t.data.iter_mut().zip(tensors[1].data.iter()) {
            *a /= b;
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Log;
impl Operator for Log {
    fn name(&self) -> &'static str {
        "Log"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for a in t.data.iter_mut() {
            *a = a.ln();
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Exp;
impl Operator for Exp {
    fn name(&self) -> &'static str {
        "Exp"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for a in t.data.iter_mut() {
            *a = a.exp();
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct VecMatMul;
impl Operator for VecMatMul {
    fn name(&self) -> &'static str {
        "VecMatMul"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let vector = tensors[0];
        let mut matrix = tensors[1].clone();
        for out in 0..matrix.shape[0] {
            for inner in 0..matrix.shape[1] {
                matrix.data[out * matrix.shape[1] + inner] *= vector.data[inner];
            }
        }
        matrix
    }
}

#[derive(Debug, Clone)]
pub struct RepeatStart(pub usize);
impl Operator for RepeatStart {
    fn name(&self) -> &'static str {
        "RepeatStart"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        // Repeat this data along a new dimension at the beginning
        let mut tensor = tensors[0].clone();
        let len = tensor.data.len();
        tensor.data = tensor.data.into_iter().cycle().take(len * self.0).collect();
        tensor.strides.insert(0, tensor.shape.iter().product());
        tensor.shape.insert(0, self.0);
        tensor
    }
}

#[derive(Debug, Clone)]
pub struct RepeatEnd(pub usize);
impl Operator for RepeatEnd {
    fn name(&self) -> &'static str {
        "RepeatEnd"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        // Repeat this data along a new dimension at the end
        let mut tensor = tensors[0].clone();
        tensor.data = tensor
            .data
            .into_iter()
            .flat_map(|i| std::iter::repeat(i).take(self.0))
            .collect();
        tensor.shape.push(self.0);
        tensor.strides.push(1);
        tensor
    }
}
