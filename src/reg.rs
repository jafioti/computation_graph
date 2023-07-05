use std::{collections::HashMap, marker::PhantomData, ops::Add};

use crate::tensor::{Const, Shape};

use uuid::Uuid;

pub fn main() {
    let a = GraphTensor::<Const<1>>::new();
    let b = GraphTensor::<Const<1>>::new();

    let c = a + &b;
    let d = b + &c;

    // We now have graphs for both c and d. We can now finalize them with data, and then feed them into the engine
    let final_c = c.finalize(vec![1.0]);
    let final_d = d.finalize(vec![2.0]);

    println!("Final Answers: {:?}", calculate(vec![final_c, final_d]));
}

#[derive(Debug)]
struct GraphTensor<S: Shape> {
    id: Uuid,
    ops: Vec<(Uuid, Vec<usize>, GraphOp)>, // A list of the previous id, shape of the tensor and the op we did to get here
    _phantom: PhantomData<S>,
}

/// Same as graph tensor but no shape, with data
#[derive(Debug)]
struct FinalizedGraphTensor {
    id: Uuid,
    ops: Vec<(Uuid, Vec<usize>, GraphOp)>, // A list of the previous shape of the tensor and the op we did to get here
    starting_data: Option<Vec<f32>>,       // The actual data of the tensor
}

impl<S: Shape> GraphTensor<S> {
    fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            ops: vec![],
            _phantom: Default::default(),
        }
    }

    fn finalize(self, starting_data: Vec<f32>) -> FinalizedGraphTensor {
        if let Some((_, first_shape, _)) = self.ops.first() {
            assert_eq!(starting_data.len(), first_shape.iter().sum());
        } else {
            assert_eq!(starting_data.len(), S::num_elements());
        }
        FinalizedGraphTensor {
            id: self.id,
            ops: self.ops,
            starting_data: Some(starting_data),
        }
    }
}

#[derive(Debug)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug)]
enum UnaryOp {
    Log,
    Exp,
}

#[derive(Debug)]
enum GraphOp {
    Binary(BinaryOp, Uuid),
    Unary(UnaryOp),
}

impl<S: Shape> Add<&GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(mut self, rhs: &GraphTensor<S>) -> Self::Output {
        self.ops.push((
            self.id,
            S::realized_shape(),
            GraphOp::Binary(BinaryOp::Add, rhs.id),
        ));
        self.id = Uuid::new_v4();
        self
    }
}

fn calculate(mut graphs: Vec<FinalizedGraphTensor>) -> Vec<Vec<f32>> {
    // Combine all graphs into a final graph, optimize, and execute it. We have the full graphs and all data required.
    let output_tensors = graphs.iter().map(|g| g.id).collect::<Vec<_>>();
    // Keep track of the tensors we currently have available
    let mut tensors = graphs
        .iter_mut()
        .map(|t| {
            if let Some((first_id, _, _)) = t.ops.first() {
                (*first_id, t.starting_data.take().unwrap())
            } else {
                (t.id, t.starting_data.take().unwrap())
            }
        })
        .collect::<HashMap<Uuid, Vec<f32>>>();

    loop {
        // Run through each graph as far as possible
        for graph in &mut graphs {
            loop {
                let Some((first_id, _, first_op)) = graph.ops.first() else {
                    break;
                };
                // Check for required ids
                if !tensors.contains_key(first_id) {
                    break;
                }
                if match first_op {
                    GraphOp::Unary(_) => false,
                    GraphOp::Binary(_, id) => !tensors.contains_key(id),
                } {
                    break;
                }

                let (tensor_id, shape, op) = graph.ops.remove(0);
                let tensor = tensors.remove(&tensor_id).unwrap();

                match op {
                    GraphOp::Unary(op) => match op {
                        UnaryOp::Exp => {}
                        UnaryOp::Log => {}
                    },
                    GraphOp::Binary(op, id) => match op {
                        BinaryOp::Add => {}
                    },
                }
            }
        }
    }
    vec![]
}
