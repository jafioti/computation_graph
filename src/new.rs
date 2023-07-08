use std::{
    collections::HashMap,
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph, visit::EdgeRef, Directed, Direction};
use strum::IntoStaticStr;

use crate::tensor::{Const, Shape};

type R1<const D: usize> = (Const<D>,);

pub fn main() {
    let mut cx = Graph::new();
    let b = cx.new_tensor::<R1<3>>();
    let c = cx.new_tensor::<R1<3>>();
    let g = cx.new_tensor::<R1<3>>();
    let e = cx.new_tensor::<R1<3>>();

    let a = b * c + g;
    let d = (b * c / e).exp().log() - a;

    let pre_optimized = cx.petgraph();

    cx.optimize();

    display_petgraph(&pre_optimized.join(&cx.petgraph()));

    cx.set_tensor(b, vec![1.0, 2.0, 3.0]);
    cx.set_tensor(c, vec![1.0, 2.0, 3.0]);
    cx.set_tensor(g, vec![1.0, 2.0, 3.0]);
    cx.set_tensor(e, vec![1.0, 2.0, 3.0]);

    let mut cx = cx.execute();

    let a = cx.get(a).unwrap();
    let d = cx.get(d).unwrap();

    println!("A: {:?}", a);
    println!("D: {:?}", d);
}

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Log,
    Exp,
}

#[derive(Debug)]
struct Graph {
    tensors: HashMap<NodeIndex, Option<Vec<f32>>>,
    op_nodes: HashMap<NodeIndex, Op>, // The Uuid of the resulting tensors after each op
    graph: StableGraph<String, bool, Directed, u32>,
}

struct ExecutedGraph {
    tensors: HashMap<NodeIndex, Vec<f32>>,
}

impl ExecutedGraph {
    pub fn get<S: Shape>(&mut self, tensor: GraphTensor<S>) -> Option<Vec<f32>> {
        self.tensors.remove(&tensor.id)
    }
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

    fn log(self) -> GraphTensor<S> {
        unsafe { self.graph_ref.as_mut().unwrap().log(self) }
    }

    fn exp(self) -> GraphTensor<S> {
        unsafe { self.graph_ref.as_mut().unwrap().exp(self) }
    }
}

impl Graph {
    fn new() -> Graph {
        Graph {
            tensors: HashMap::default(),
            op_nodes: HashMap::default(),
            graph: petgraph::stable_graph::StableGraph::default(),
        }
    }

    fn new_tensor<S: Shape>(&mut self) -> GraphTensor<S> {
        let id = self.graph.add_node((self.tensors.len() + 1).to_string());
        self.tensors.insert(id, None);
        GraphTensor {
            id,
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

    fn set_tensor<S: Shape>(&mut self, graph_tensor: GraphTensor<S>, data: Vec<f32>) {
        *self.tensors.get_mut(&graph_tensor.id).unwrap() = Some(data);
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
                    Op::Add | Op::Exp | Op::Log | Op::Mul => false,
                    Op::Div | Op::Sub => true,
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

    fn execute(mut self) -> ExecutedGraph {
        loop {
            let mut new_tensors = vec![];
            // Find all executable ops
            for (node, srcs) in self.graph.node_indices().filter_map(|n| {
                if self.tensors.contains_key(&n) {
                    return None;
                }
                let mut data = vec![];
                let mut missed = false;
                for e in self.graph.edges_directed(n, petgraph::Direction::Incoming) {
                    if let Some(Some(e)) = self.tensors.get(&e.source()) {
                        data.push(e);
                    } else {
                        missed = true;
                        break;
                    }
                }
                if missed {
                    None
                } else {
                    Some((n, data))
                }
            }) {
                // All sources are ready, execute
                let f = match self.op_nodes[&node] {
                    Op::Add => {
                        let mut f = srcs[0].clone();
                        for src in srcs.iter().skip(1) {
                            for j in 0..src.len() {
                                f[j] += src[j];
                            }
                        }
                        f
                    }
                    Op::Sub => {
                        let mut f = srcs[0].clone();
                        for src in srcs.iter().skip(1) {
                            for j in 0..src.len() {
                                f[j] -= src[j];
                            }
                        }
                        f
                    }
                    Op::Mul => {
                        let mut f = srcs[0].clone();
                        for src in srcs.iter().skip(1) {
                            for j in 0..src.len() {
                                f[j] *= src[j];
                            }
                        }
                        f
                    }
                    Op::Div => {
                        let mut f = srcs[0].clone();
                        for src in srcs.iter().skip(1) {
                            for j in 0..src.len() {
                                f[j] /= src[j];
                            }
                        }
                        f
                    }
                    Op::Log => {
                        let mut f = srcs[0].clone();
                        for i in &mut f {
                            *i = i.ln();
                        }
                        f
                    }
                    Op::Exp => {
                        let mut f = srcs[0].clone();
                        for i in &mut f {
                            *i = i.exp();
                        }
                        f
                    }
                };
                new_tensors.push((node, Some(f)));
            }

            if new_tensors.is_empty() {
                break;
            }

            for (k, v) in new_tensors {
                self.tensors.insert(k, v);
            }
        }

        ExecutedGraph {
            tensors: self
                .tensors
                .into_iter()
                .flat_map(|(id, data)| data.map(|data| (id, data)))
                .collect(),
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
