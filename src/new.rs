#![allow(clippy::needless_range_loop)]

use crate::{
    op::{self, Operator},
    shape::*,
};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph, visit::EdgeRef, Directed, Direction};

pub fn main() {
    let mut cx = Graph::new();
    let b = cx.new_tensor::<R1<3>>();
    let c = cx.new_tensor::<R1<3>>();
    let g = cx.new_tensor::<R1<3>>();
    let e = cx.new_tensor::<R1<3>>();

    let a = b * c + g;
    let d = (b * c / e).exp().log().repeat_start::<2>();
    let a = a.vec_mat_mul(d);

    b.set(vec![1.0, 2.0, 3.0]);
    c.set(vec![1.0, 2.0, 3.0]);
    g.set(vec![1.0, 2.0, 3.0]);
    e.set(vec![1.0, 2.0, 3.0]);

    a.mark();
    d.mark();

    cx.execute();

    let unoptimized_a = a.retrieve().unwrap();
    let unoptimized_d = d.retrieve().unwrap();

    let pre_optimized = cx.petgraph();

    cx.optimize();

    display_petgraph(&pre_optimized.join(&cx.petgraph()));

    cx.execute();
    assert_close(&unoptimized_a, &a.retrieve().unwrap());
    assert_close(&unoptimized_d, &d.retrieve().unwrap());
}

fn assert_close(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape, b.shape, "Shapes don't match");
    assert_eq!(a.strides, b.strides, "Strides don't match");
    for (a, b) in a.data.iter().zip(b.data.iter()) {
        if (a - b).abs() > 0.01 {
            panic!("{a} is not close to {b}");
        }
    }
}

#[derive(Debug, Default)]
pub struct Graph {
    tensors: HashMap<NodeIndex, Tensor>,
    id_remap: HashMap<NodeIndex, NodeIndex>,
    graph: StableGraph<Box<dyn Operator>, u8, Directed, u32>,
    no_delete: HashSet<NodeIndex>,
}

#[derive(Clone, Copy)]
pub struct GraphTensor<S: Shape> {
    id: NodeIndex,
    graph_ref: *mut Graph,
    _phantom: PhantomData<S>,
}

/// An entirely dynamic tensor with data
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub strides: Vec<usize>,
    pub shape: Vec<usize>,
}

impl<S: Shape> GraphTensor<S> {
    fn from_id(id: NodeIndex, graph_ref: *mut Graph) -> Self {
        Self {
            id,
            graph_ref,
            _phantom: Default::default(),
        }
    }

    /// Mark this tensor to be retrieved later
    fn mark(&self) {
        unsafe { self.graph_ref.as_mut().unwrap().mark(*self) }
    }

    fn retrieve(self) -> Option<Tensor> {
        unsafe { self.graph_ref.as_mut().unwrap().get_tensor(self.id) }
    }

    fn set(&self, data: Vec<f32>) {
        unsafe { self.graph_ref.as_mut().unwrap().set_tensor(*self, data) }
    }

    fn log(self) -> GraphTensor<S> {
        let graph = unsafe { &mut self.graph_ref.as_mut().unwrap().graph };
        let new_id = graph.add_node(Box::new(op::Log));
        graph.add_edge(self.id, new_id, 0);
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    fn exp(self) -> GraphTensor<S> {
        let graph = unsafe { &mut self.graph_ref.as_mut().unwrap().graph };
        let new_id = graph.add_node(Box::new(op::Exp));
        graph.add_edge(self.id, new_id, 0);
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    fn repeat_start<const R: usize>(self) -> GraphTensor<<S as Shape>::AddLeft<R>> {
        let graph = unsafe { &mut self.graph_ref.as_mut().unwrap().graph };
        let new_id = graph.add_node(Box::new(op::RepeatStart(R)));
        graph.add_edge(self.id, new_id, 0);
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    fn repeat_end<const R: usize>(self) -> GraphTensor<<S as Shape>::AddRight<R>> {
        let graph = unsafe { &mut self.graph_ref.as_mut().unwrap().graph };
        let new_id = graph.add_node(Box::new(op::RepeatEnd(R)));
        graph.add_edge(self.id, new_id, 0);
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<const C: usize> GraphTensor<R1<C>> {
    fn vec_mat_mul<const R: usize>(self, rhs: GraphTensor<R2<R, C>>) -> GraphTensor<R2<R, C>> {
        unsafe { self.graph_ref.as_mut().unwrap().vec_mat_mul(self, rhs) }
    }
}

impl Graph {
    fn new() -> Graph {
        Graph::default()
    }

    fn get_tensor(&mut self, mut id: NodeIndex) -> Option<Tensor> {
        // Walk through remaps
        while let Some(new_id) = self.id_remap.get(&id) {
            id = *new_id;
        }

        self.tensors.remove(&id)
    }

    fn new_tensor<S: Shape>(&mut self) -> GraphTensor<S> {
        let tensor = GraphTensor {
            id: self.graph.add_node(Box::new(op::Input)),
            graph_ref: self,
            _phantom: Default::default(),
        };
        self.no_delete.insert(tensor.id);
        tensor
    }

    fn vec_mat_mul<const C: usize, const R: usize>(
        &mut self,
        vector: GraphTensor<R1<C>>,
        matrix: GraphTensor<R2<R, C>>,
    ) -> GraphTensor<R2<R, C>> {
        // Multiply the matrix by the vector on each row
        let new_id = self.graph.add_node(Box::new(op::VecMatMul));
        self.graph.add_edge(vector.id, new_id, 0);
        self.graph.add_edge(matrix.id, new_id, 1);
        GraphTensor::from_id(new_id, vector.graph_ref)
    }

    fn set_tensor<S: Shape>(&mut self, graph_tensor: GraphTensor<S>, data: Vec<f32>) {
        let mut strides = S::realized_shape()
            .into_iter()
            .rev()
            .scan(1, |acc, x| {
                let before = *acc;
                *acc *= x;
                Some(before)
            })
            .collect_vec();
        strides.reverse();
        self.tensors.insert(
            graph_tensor.id,
            Tensor {
                data,
                strides,
                shape: S::realized_shape(),
            },
        );
    }

    /// Mark a tensor for retrival (memory won't free after executed)
    fn mark<S: Shape>(&mut self, graph_tensor: GraphTensor<S>) {
        self.no_delete.insert(graph_tensor.id);
    }

    /// Run the full suite of optimizations
    fn optimize(&mut self) {
        self.unary_sequential_opt();
        self.cse_opt();
    }

    /// Eliminate complementary unary sequential operations like `x.log().exp()`
    fn unary_sequential_opt(&mut self) {
        // Scan through unary sequential eliminations
        for id in self.graph.node_indices().collect_vec() {
            if self.no_delete.contains(&id) {
                continue;
            }
            for outgoing_target in self
                .graph
                .edges_directed(id, petgraph::Direction::Outgoing)
                .map(|i| i.target())
                .collect_vec()
            {
                let op = self.graph.node_weight(id).unwrap();
                if (op.name() == "Exp"
                    && self.graph.node_weight(outgoing_target).unwrap().name() == "Log")
                    || (op.name() == "Log"
                        && self.graph.node_weight(outgoing_target).unwrap().name() == "Exp")
                {
                    // Remove current node and next node
                    let pre_node = self
                        .graph
                        .edges_directed(id, petgraph::Direction::Incoming)
                        .next()
                        .unwrap()
                        .source();

                    for (edge_weight, outgoing_edge_target) in self
                        .graph
                        .edges_directed(outgoing_target, Direction::Outgoing)
                        .map(|e| (*e.weight(), e.target()))
                        .collect_vec()
                    {
                        self.graph
                            .add_edge(pre_node, outgoing_edge_target, edge_weight);
                    }

                    Self::move_references(
                        &mut self.id_remap,
                        &mut self.no_delete,
                        outgoing_target,
                        pre_node,
                    );
                    self.graph.remove_node(id);
                    self.graph.remove_node(outgoing_target);
                }
            }
        }
    }

    /// Common subexpression elimination (https://en.wikipedia.org/wiki/Common_subexpression_elimination)
    fn cse_opt(&mut self) {
        // Look for nodes that have the exact same srcs
        // Loop cause I'm lazy
        loop {
            let mut eliminated = false;
            let mut srcs_set = HashMap::new();
            for node in self.graph.node_indices().collect_vec() {
                let mut srcs = self
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .map(|e| e.source())
                    .collect_vec();

                if srcs.is_empty() || self.graph.node_weight(node).unwrap().name() == "Input" {
                    continue;
                }

                // If order doesn't matter, make sure  different ordered srcs match by sorting
                srcs.sort();

                if let Some(other_node) = srcs_set.get(&srcs) {
                    if self.graph.node_weight(node).unwrap().name()
                        == self.graph.node_weight(*other_node).unwrap().name()
                    {
                        // Carry over outgoing edges from node to other_node
                        for (weight, target) in self
                            .graph
                            .edges_directed(node, petgraph::Direction::Outgoing)
                            .map(|e| (*e.weight(), e.target()))
                            .collect_vec()
                        {
                            self.graph.add_edge(*other_node, target, weight);
                        }
                        // Transfer all references to node over to other node
                        Self::move_references(
                            &mut self.id_remap,
                            &mut self.no_delete,
                            node,
                            *other_node,
                        );
                        // Remove node
                        self.graph.remove_node(node);
                        eliminated = true;
                        break;
                    }
                }
                srcs_set.insert(srcs, node);
            }

            if !eliminated {
                break;
            }
        }
    }

    /// Execute the graph.
    fn execute(&mut self) {
        loop {
            let mut new_tensors = vec![];
            // Find all executable ops
            for (node, srcs) in self
                .graph
                .node_indices()
                .filter_map(|n| {
                    if self.tensors.contains_key(&n) {
                        return None;
                    }
                    let mut data = vec![];
                    for e in self
                        .graph
                        .edges_directed(n, petgraph::Direction::Incoming)
                        .sorted_by_key(|e| e.weight())
                    {
                        if let Some(e) = self.tensors.get(&e.source()) {
                            data.push(e);
                        } else {
                            return None;
                        }
                    }
                    Some((n, data))
                })
                .collect_vec()
            {
                // All sources are ready, execute
                let f = self.graph.node_weight(node).unwrap().process(srcs);
                new_tensors.push((node, f));
            }

            // Check if we can delete the source tensors now
            for node in new_tensors.iter().map(|(t, _)| t) {
                // Check we have incoming edges (don't want to remove the sources)
                for source in self
                    .graph
                    .edges_directed(*node, Direction::Incoming)
                    .map(|e| e.source())
                    .filter(|e| self.graph.edges_directed(*e, Direction::Outgoing).count() == 1)
                    .collect_vec()
                {
                    if !self.no_delete.contains(&source) {
                        // Delete tensor and node
                        self.tensors.remove(&source);
                    }
                }
            }

            if new_tensors.is_empty() {
                break;
            }

            for (k, v) in new_tensors {
                self.tensors.insert(k, v);
            }
        }
    }

    // Convert to petgraph
    fn petgraph(&self) -> petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32> {
        let mut new_graph = petgraph::stable_graph::StableGraph::default();
        let mut id_map = HashMap::new();
        for (id, node) in self.graph.node_indices().zip(self.graph.node_weights()) {
            id_map.insert(id, new_graph.add_node(format!("{node:?}")));
        }

        for node in self.graph.node_indices() {
            for edge in self
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing)
            {
                new_graph.add_edge(
                    id_map[&edge.source()],
                    id_map[&edge.target()],
                    *edge.weight(),
                );
            }
        }

        new_graph
    }

    /// Transfer all external references from one node to another (this may happen because one node is about to be removed / merged into another)
    fn move_references(
        id_remap: &mut HashMap<NodeIndex, NodeIndex>,
        no_delete: &mut HashSet<NodeIndex<u32>>,
        src: NodeIndex,
        trg: NodeIndex,
    ) {
        // Create remap
        id_remap.insert(src, trg);
        // Transfer no_delete
        if no_delete.remove(&src) {
            no_delete.insert(trg);
        }
    }
}

fn display_petgraph(
    graph: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
) {
    // Display in browser
    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(
            &petgraph::dot::Dot::with_config(&graph, &[petgraph::dot::Config::EdgeNoLabel,])
                .to_string()
        )
    );
    if let Err(e) = webbrowser::open(&url) {
        println!("Error displaying graph: {:?}", e);
    }
}

trait JoinGraph {
    fn join(
        self,
        rhs: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
    ) -> Self;
}

impl JoinGraph for petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32> {
    fn join(
        mut self,
        rhs: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
    ) -> Self {
        let mut id_map = HashMap::new(); // We track the node id remapping here so they don't overlap
        for (index, node) in rhs.node_indices().zip(rhs.node_weights()) {
            id_map.insert(index, self.add_node(node.clone()));
        }

        for node in rhs.node_indices() {
            for edge in rhs.edges_directed(node, petgraph::Direction::Outgoing) {
                self.add_edge(
                    id_map[&edge.source()],
                    id_map[&edge.target()],
                    *edge.weight(),
                );
            }
        }

        self
    }
}

impl<S: Shape> Add<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { &mut self.graph_ref.as_mut().unwrap().graph };
        let new_id = graph.add_node(Box::new(op::Add));
        graph.add_edge(self.id, new_id, 0);
        graph.add_edge(rhs.id, new_id, 1);
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: Shape> Sub<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { &mut self.graph_ref.as_mut().unwrap().graph };
        let new_id = graph.add_node(Box::new(op::Subtract));
        graph.add_edge(self.id, new_id, 0);
        graph.add_edge(rhs.id, new_id, 1);
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: Shape> Mul<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { &mut self.graph_ref.as_mut().unwrap().graph };
        let new_id = graph.add_node(Box::new(op::Multiply));
        graph.add_edge(self.id, new_id, 0);
        graph.add_edge(rhs.id, new_id, 1);
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: Shape> Div<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { &mut self.graph_ref.as_mut().unwrap().graph };
        let new_id = graph.add_node(Box::new(op::Division));
        graph.add_edge(self.id, new_id, 0);
        graph.add_edge(rhs.id, new_id, 1);
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}
