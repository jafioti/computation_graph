use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

use itertools::Itertools;
use petgraph::{
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgesDirected},
    Directed, Direction,
};
use strum::IntoStaticStr;

use crate::shape::{Const, Shape};

type R1<const D: usize> = (Const<D>,);
type R2<const A: usize, const B: usize> = (Const<A>, Const<B>);

pub fn main() {
    let mut cx = Graph::new();
    let b = cx.new_tensor::<R1<3>>();
    let c = cx.new_tensor::<R1<3>>();
    let g = cx.new_tensor::<R1<3>>();
    let e = cx.new_tensor::<R1<3>>();
    let f = cx.new_tensor::<R2<2, 3>>();

    let a = b * c + g;
    let d = (b * c / e).exp().log() - a;
    let z = d.vec_mat_mul(f);

    let pre_optimized = cx.petgraph();

    cx.optimize();

    display_petgraph(&pre_optimized.join(&cx.petgraph()));

    b.set(vec![1.0, 2.0, 3.0]);
    c.set(vec![1.0, 2.0, 3.0]);
    g.set(vec![1.0, 2.0, 3.0]);
    e.set(vec![1.0, 2.0, 3.0]);
    f.set(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);

    a.mark();
    d.mark();
    z.mark();

    cx.execute();

    let a = a.retrieve().unwrap();
    let d = d.retrieve().unwrap();
    let z = z.retrieve().unwrap();

    println!("A: {:?}", a);
    println!("D: {:?}", d);
    println!("Z: {:?}", z);
}

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Log,
    Exp,
    VecMatMul,
    /// Repeat a tensor along a new dimension at the start or end
    Repeat(RepeatSide),
}

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq)]
enum RepeatSide {
    Start,
    End,
}

#[derive(Debug, Default)]
struct Graph {
    tensors: HashMap<NodeIndex, (Vec<f32>, Vec<usize>)>,
    op_nodes: HashMap<NodeIndex, Op>,
    graph: StableGraph<String, bool, Directed, u32>,
    no_delete: HashSet<NodeIndex>,
}

#[derive(Clone, Copy)]
struct GraphTensor<S: Shape> {
    id: NodeIndex,
    graph_ref: *mut Graph,
    _phantom: PhantomData<S>,
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

    fn retrieve(self) -> Option<(Vec<f32>, Vec<usize>)> {
        unsafe { self.graph_ref.as_mut().unwrap().tensors.remove(&self.id) }
    }

    fn set(&self, data: Vec<f32>) {
        unsafe { self.graph_ref.as_mut().unwrap().set_tensor(*self, data) }
    }

    fn log(self) -> GraphTensor<S> {
        unsafe { self.graph_ref.as_mut().unwrap().log(self) }
    }

    fn exp(self) -> GraphTensor<S> {
        unsafe { self.graph_ref.as_mut().unwrap().exp(self) }
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

    fn new_tensor<S: Shape>(&mut self) -> GraphTensor<S> {
        GraphTensor {
            id: self.graph.add_node((self.tensors.len() + 1).to_string()),
            graph_ref: self,
            _phantom: Default::default(),
        }
    }

    fn add<S: Shape>(&mut self, t1: GraphTensor<S>, t2: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = self.graph.add_node("Add".to_string());
        self.op_nodes.insert(new_id, Op::Add);
        self.graph.add_edge(t1.id, new_id, false);
        self.graph.add_edge(t2.id, new_id, false);
        GraphTensor::from_id(new_id, t1.graph_ref)
    }

    fn sub<S: Shape>(&mut self, t1: GraphTensor<S>, t2: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = self.graph.add_node("Sub".to_string());
        self.op_nodes.insert(new_id, Op::Sub);
        self.graph.add_edge(t1.id, new_id, false);
        self.graph.add_edge(t2.id, new_id, false);
        GraphTensor::from_id(new_id, t1.graph_ref)
    }

    fn mul<S: Shape>(&mut self, t1: GraphTensor<S>, t2: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = self.graph.add_node("Mul".to_string());
        self.op_nodes.insert(new_id, Op::Mul);
        self.graph.add_edge(t1.id, new_id, false);
        self.graph.add_edge(t2.id, new_id, false);
        GraphTensor::from_id(new_id, t1.graph_ref)
    }

    fn div<S: Shape>(&mut self, t1: GraphTensor<S>, t2: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = self.graph.add_node("Div".to_string());
        self.op_nodes.insert(new_id, Op::Div);
        self.graph.add_edge(t1.id, new_id, false);
        self.graph.add_edge(t2.id, new_id, false);
        GraphTensor::from_id(new_id, t1.graph_ref)
    }

    fn log<S: Shape>(&mut self, t1: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = self.graph.add_node("Log".to_string());
        self.op_nodes.insert(new_id, Op::Log);
        self.graph.add_edge(t1.id, new_id, false);
        GraphTensor::from_id(new_id, t1.graph_ref)
    }

    fn exp<S: Shape>(&mut self, t1: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = self.graph.add_node("Exp".to_string());
        self.op_nodes.insert(new_id, Op::Exp);
        self.graph.add_edge(t1.id, new_id, false);
        GraphTensor::from_id(new_id, t1.graph_ref)
    }

    fn vec_mat_mul<const C: usize, const R: usize>(
        &mut self,
        vector: GraphTensor<R1<C>>,
        matrix: GraphTensor<R2<R, C>>,
    ) -> GraphTensor<R2<R, C>> {
        // Multiply the matrix by the vector on each row
        let new_id = self.graph.add_node("VecMatMul".to_string());
        self.op_nodes.insert(new_id, Op::VecMatMul);
        self.graph.add_edge(vector.id, new_id, false);
        self.graph.add_edge(matrix.id, new_id, false);
        GraphTensor::from_id(new_id, vector.graph_ref)
    }

    fn set_tensor<S: Shape>(&mut self, graph_tensor: GraphTensor<S>, data: Vec<f32>) {
        self.tensors
            .insert(graph_tensor.id, (data, S::realized_shape()));
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
        for (id, op_node) in self.op_nodes.clone() {
            for outgoing_target in self
                .graph
                .edges_directed(id, petgraph::Direction::Outgoing)
                .map(|i| i.target())
                .collect_vec()
            {
                if (matches!(op_node, Op::Exp)
                    && matches!(self.op_nodes[&outgoing_target], Op::Log))
                    || (matches!(op_node, Op::Log)
                        && matches!(self.op_nodes[&outgoing_target], Op::Exp))
                {
                    // Remove current node and next node
                    // Take outgoing connections from second node and connect them to node before first
                    let pre_node = self
                        .graph
                        .edges_directed(id, petgraph::Direction::Incoming)
                        .next()
                        .unwrap()
                        .source();

                    self.graph.move_incoming_edges(pre_node, outgoing_target);
                    *self.graph.node_weight_mut(outgoing_target).unwrap() =
                        self.graph.node_weight(pre_node).unwrap().clone();
                    self.graph.remove_node(pre_node);
                    self.graph.remove_node(id);
                    self.op_nodes
                        .insert(outgoing_target, self.op_nodes[&pre_node]);
                }
            }
        }
    }

    /// Common subexpression elimination (https://en.wikipedia.org/wiki/Common_subexpression_elimination)
    fn cse_opt(&mut self) {
        // Look for nodes that have the exact same srcs
        loop {
            // Loop cause I'm lazy
            let mut eliminated = false;
            let mut srcs_set = HashMap::new();
            for node in self.graph.node_indices().collect_vec() {
                let mut srcs = self
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .map(|e| e.source())
                    .collect_vec();

                if srcs.is_empty() || self.op_nodes.get(&node).is_none() {
                    continue;
                }

                // If order doesn't matter, make sure  different ordered srcs match by sorting
                let order_matters = match self.op_nodes[&node] {
                    Op::Add | Op::Exp | Op::Log | Op::Mul | Op::Repeat(_) => false,
                    Op::Div | Op::Sub | Op::VecMatMul => true,
                };
                if !order_matters {
                    srcs.sort();
                }

                if let Some(other_node) = srcs_set.get(&srcs) {
                    if self.op_nodes[&node] == self.op_nodes[other_node] {
                        // Carry over outgoing edges from node to other_node
                        for target in self
                            .graph
                            .edges_directed(node, petgraph::Direction::Outgoing)
                            .map(|e| e.target())
                            .collect_vec()
                        {
                            self.graph.add_edge(*other_node, target, false);
                        }
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
                        .sorted_by_key(|e| e.id())
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
                let f = match self.op_nodes[&node] {
                    Op::Add => {
                        let (mut f, shape) = srcs[0].clone();
                        for (src, _) in srcs.iter().skip(1) {
                            for j in 0..src.len() {
                                f[j] += src[j];
                            }
                        }
                        (f, shape)
                    }
                    Op::Sub => {
                        let (mut f, shape) = srcs[0].clone();
                        for (src, _) in srcs.iter().skip(1) {
                            for j in 0..src.len() {
                                f[j] -= src[j];
                            }
                        }
                        (f, shape)
                    }
                    Op::Mul => {
                        let (mut f, shape) = srcs[0].clone();
                        for (src, _) in srcs.iter().skip(1) {
                            for j in 0..src.len() {
                                f[j] *= src[j];
                            }
                        }
                        (f, shape)
                    }
                    Op::Div => {
                        let (mut f, shape) = srcs[0].clone();
                        for (src, _) in srcs.iter().skip(1) {
                            for j in 0..src.len() {
                                f[j] /= src[j];
                            }
                        }
                        (f, shape)
                    }
                    Op::Log => {
                        let (mut f, shape) = srcs[0].clone();
                        for i in &mut f {
                            *i = i.ln();
                        }
                        (f, shape)
                    }
                    Op::Exp => {
                        let (mut f, shape) = srcs[0].clone();
                        for i in &mut f {
                            *i = i.exp();
                        }
                        (f, shape)
                    }
                    Op::VecMatMul => {
                        let (vector, _) = srcs[0];
                        let (mut matrix, mat_shape) = srcs[1].clone();
                        for out in 0..mat_shape[0] {
                            for inner in 0..mat_shape[1] {
                                matrix[out * mat_shape[1] + inner] *= vector[inner];
                            }
                        }
                        (matrix, mat_shape)
                    }
                    Op::Repeat(_) => {
                        let (mut f, shape) = srcs[0].clone();
                        for i in &mut f {
                            *i = i.ln();
                        }
                        (f, shape)
                    }
                };
                new_tensors.push((node, f));
            }

            // Check if we can delete the source tensors now
            for node in new_tensors.iter().map(|(t, _)| t) {
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
                        self.graph.remove_node(source);
                    }
                }

                // Remove edges going to current node
                for edge in self
                    .graph
                    .edges_directed(*node, Direction::Incoming)
                    .map(|e| e.id())
                    .collect_vec()
                {
                    self.graph.remove_edge(edge);
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
    fn petgraph(
        &self,
    ) -> petgraph::stable_graph::StableGraph<String, bool, petgraph::Directed, u32> {
        self.graph.clone()
    }
}

fn display_petgraph(
    graph: &petgraph::stable_graph::StableGraph<String, bool, petgraph::Directed, u32>,
) {
    // Display in browser
    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(
            &petgraph::dot::Dot::with_config(&graph, &[petgraph::dot::Config::EdgeNoLabel])
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
        rhs: &petgraph::stable_graph::StableGraph<String, bool, petgraph::Directed, u32>,
    ) -> Self;
}

impl JoinGraph for petgraph::stable_graph::StableGraph<String, bool, petgraph::Directed, u32> {
    fn join(
        mut self,
        rhs: &petgraph::stable_graph::StableGraph<String, bool, petgraph::Directed, u32>,
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
        unsafe { self.graph_ref.as_mut().unwrap().add(self, rhs) }
    }
}

impl<S: Shape> Sub<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: GraphTensor<S>) -> Self::Output {
        unsafe { self.graph_ref.as_mut().unwrap().sub(self, rhs) }
    }
}

impl<S: Shape> Mul<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: GraphTensor<S>) -> Self::Output {
        unsafe { self.graph_ref.as_mut().unwrap().mul(self, rhs) }
    }
}

impl<S: Shape> Div<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: GraphTensor<S>) -> Self::Output {
        unsafe { self.graph_ref.as_mut().unwrap().div(self, rhs) }
    }
}

trait MoveIncomingEdges {
    fn move_incoming_edges(&mut self, orig_id: NodeIndex, new_id: NodeIndex);
}

impl MoveIncomingEdges for StableGraph<String, bool, petgraph::Directed, u32> {
    fn move_incoming_edges(&mut self, orig_id: NodeIndex, new_id: NodeIndex) {
        // Clear incoming edges off new node
        for edge in self
            .edges_directed(new_id, Direction::Incoming)
            .map(|e| e.id())
            .collect_vec()
        {
            self.remove_edge(edge);
        }

        // Create new edges
        for source in self
            .edges_directed(orig_id, Direction::Incoming)
            .map(|e| e.source())
            .collect_vec()
        {
            self.add_edge(source, new_id, false);
        }
    }
}
