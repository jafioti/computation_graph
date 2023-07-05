use std::{collections::HashMap, marker::PhantomData};

use itertools::Itertools;
use strum::IntoStaticStr;
use uuid::Uuid;

use crate::tensor::{Const, Shape};

type R1<const D: usize> = (Const<D>,);

pub fn main() {
    let mut cx = Graph::new();
    let b = cx.tensor::<R1<3>>();
    let c = cx.tensor::<R1<3>>();
    let g = cx.tensor::<R1<3>>();
    let e = cx.tensor::<R1<3>>();

    let a = b.mul(&mut cx, c).add(&mut cx, g);
    let d = b
        .mul(&mut cx, c)
        .mul(&mut cx, e)
        .exp(&mut cx)
        .log(&mut cx)
        .sub(&mut cx, a);

    cx.set_tensor(b, vec![1.0, 2.0, 3.0]);
    cx.set_tensor(c, vec![1.0, 2.0, 3.0]);
    cx.set_tensor(g, vec![1.0, 2.0, 3.0]);
    cx.set_tensor(e, vec![1.0, 2.0, 3.0]);

    let pre_optimized = cx.petgraph();
    cx.optimize();

    display_petgraph(&pre_optimized.join(&cx.petgraph()));

    let mut cx = cx.execute();

    let a = cx.get(a).unwrap();
    let d = cx.get(d).unwrap();

    println!("A: {:?}", a);
    println!("D: {:?}", d);
}

#[derive(Clone, Copy, Debug, IntoStaticStr)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Log,
    Exp,
}

struct Graph {
    tensors: HashMap<Uuid, Option<Vec<f32>>>,
    op_nodes: HashMap<Uuid, Op>, // The Uuid of the resulting tensors after each op
    connections: HashMap<Uuid, Vec<Uuid>>, // The edges of the computation graph
}

struct ExecutedGraph {
    tensors: HashMap<Uuid, Vec<f32>>,
}

impl ExecutedGraph {
    pub fn get<S: Shape>(&mut self, tensor: GraphTensor<S>) -> Option<Vec<f32>> {
        self.tensors.remove(&tensor.id)
    }
}

#[derive(Clone, Copy)]
struct GraphTensor<S: Shape> {
    id: Uuid,
    _phantom: PhantomData<S>,
}

impl<S: Shape> GraphTensor<S> {
    fn from_id(id: Uuid) -> Self {
        Self {
            id,
            _phantom: Default::default(),
        }
    }

    fn add(self, cx: &mut Graph, rhs: GraphTensor<S>) -> GraphTensor<S> {
        cx.add(self, rhs)
    }

    fn sub(self, cx: &mut Graph, rhs: GraphTensor<S>) -> GraphTensor<S> {
        cx.sub(self, rhs)
    }

    fn mul(self, cx: &mut Graph, rhs: GraphTensor<S>) -> GraphTensor<S> {
        cx.mul(self, rhs)
    }

    fn div(self, cx: &mut Graph, rhs: GraphTensor<S>) -> GraphTensor<S> {
        cx.div(self, rhs)
    }

    fn log(self, cx: &mut Graph) -> GraphTensor<S> {
        cx.log(self)
    }

    fn exp(self, cx: &mut Graph) -> GraphTensor<S> {
        cx.exp(self)
    }
}

impl Graph {
    fn new() -> Graph {
        Graph {
            tensors: HashMap::default(),
            op_nodes: HashMap::default(),
            connections: HashMap::default(),
        }
    }

    fn tensor<S: Shape>(&mut self) -> GraphTensor<S> {
        let tensor = GraphTensor {
            id: Uuid::new_v4(),
            _phantom: Default::default(),
        };
        self.tensors.insert(tensor.id, None);
        tensor
    }

    fn add<S: Shape>(&mut self, t1: GraphTensor<S>, t2: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = Uuid::new_v4();
        self.op_nodes.insert(new_id, Op::Add);
        if let Some(v) = self.connections.get_mut(&t1.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t1.id, vec![new_id]);
        }
        if let Some(v) = self.connections.get_mut(&t2.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t2.id, vec![new_id]);
        }
        GraphTensor::from_id(new_id)
    }

    fn sub<S: Shape>(&mut self, t1: GraphTensor<S>, t2: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = Uuid::new_v4();
        self.op_nodes.insert(new_id, Op::Sub);
        if let Some(v) = self.connections.get_mut(&t1.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t1.id, vec![new_id]);
        }
        if let Some(v) = self.connections.get_mut(&t2.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t2.id, vec![new_id]);
        }
        GraphTensor::from_id(new_id)
    }

    fn mul<S: Shape>(&mut self, t1: GraphTensor<S>, t2: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = Uuid::new_v4();
        self.op_nodes.insert(new_id, Op::Mul);
        if let Some(v) = self.connections.get_mut(&t1.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t1.id, vec![new_id]);
        }
        if let Some(v) = self.connections.get_mut(&t2.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t2.id, vec![new_id]);
        }
        GraphTensor::from_id(new_id)
    }

    fn div<S: Shape>(&mut self, t1: GraphTensor<S>, t2: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = Uuid::new_v4();
        self.op_nodes.insert(new_id, Op::Div);
        if let Some(v) = self.connections.get_mut(&t1.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t1.id, vec![new_id]);
        }
        if let Some(v) = self.connections.get_mut(&t2.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t2.id, vec![new_id]);
        }
        GraphTensor::from_id(new_id)
    }

    fn log<S: Shape>(&mut self, t1: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = Uuid::new_v4();
        self.op_nodes.insert(new_id, Op::Log);
        if let Some(v) = self.connections.get_mut(&t1.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t1.id, vec![new_id]);
        }
        GraphTensor::from_id(new_id)
    }

    fn exp<S: Shape>(&mut self, t1: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = Uuid::new_v4();
        self.op_nodes.insert(new_id, Op::Exp);
        if let Some(v) = self.connections.get_mut(&t1.id) {
            v.push(new_id);
        } else {
            self.connections.insert(t1.id, vec![new_id]);
        }
        GraphTensor::from_id(new_id)
    }

    fn set_tensor<S: Shape>(&mut self, graph_tensor: GraphTensor<S>, data: Vec<f32>) {
        *self.tensors.get_mut(&graph_tensor.id).unwrap() = Some(data);
    }

    fn optimize(&mut self) {
        let mut reverse_edges: HashMap<Uuid, Vec<Uuid>> = self
            .connections
            .clone()
            .into_iter()
            .flat_map(|(src, dests)| dests.iter().map(|dest| (*dest, src)).collect::<Vec<_>>())
            .into_group_map();
        // Scan through unary sequential eliminations
        for (id, op_node) in self.op_nodes.clone() {
            if self.op_nodes.contains_key(&id) {
                for outgoing_connection in self.connections.get(&id).cloned().unwrap_or_default() {
                    if (matches!(op_node, Op::Exp)
                        && matches!(self.op_nodes[&outgoing_connection], Op::Log))
                        || (matches!(op_node, Op::Log)
                            && matches!(self.op_nodes[&outgoing_connection], Op::Exp))
                    {
                        // Remove current node and next node
                        *self.connections.get_mut(&reverse_edges[&id][0]).unwrap() =
                            self.connections[&outgoing_connection].clone();
                        reverse_edges.remove(&self.connections[&outgoing_connection][0]);
                        reverse_edges.remove(&self.connections[&outgoing_connection][0]);
                        self.connections.remove(&outgoing_connection);
                        self.op_nodes.remove(&outgoing_connection);
                        self.op_nodes.remove(&id);
                        self.connections.remove(&id);
                    }
                }
            }
        }
    }

    fn execute(self) -> ExecutedGraph {
        let mut ready_nodes = self.tensors;
        let mut new_ready_nodes: HashMap<Uuid, Option<Vec<f32>>> = HashMap::default();
        let mut edges: HashMap<Uuid, Vec<Uuid>> = self.connections;
        let mut reverse_edges: HashMap<Uuid, Vec<Uuid>> = edges
            .iter()
            .flat_map(|(src, dests)| dests.iter().map(|dest| (*dest, *src)).collect::<Vec<_>>())
            .into_group_map();
        while !edges.is_empty() {
            // Find all executable ops
            for (id, data) in &ready_nodes {
                if let Some(data) = data {
                    if let Some(node_edges) = edges.get(id).cloned() {
                        for edge in node_edges {
                            let srcs = reverse_edges[&edge]
                                .iter()
                                .flat_map(|i| ready_nodes.get(i))
                                .flatten()
                                .collect::<Vec<_>>();
                            if srcs.len() == reverse_edges[&edge].len() {
                                // Remove the edges
                                for edge in &reverse_edges[&edge] {
                                    edges.remove(edge);
                                }
                                reverse_edges.remove(&edge);
                                // All sources are ready, execute
                                match self.op_nodes[&edge] {
                                    Op::Add => {
                                        let mut f = data.clone();
                                        for src in srcs.iter().skip(1) {
                                            for j in 0..src.len() {
                                                f[j] += src[j];
                                            }
                                        }
                                        new_ready_nodes.insert(edge, Some(f));
                                    }
                                    Op::Sub => {
                                        let mut f = data.clone();
                                        for src in srcs.iter().skip(1) {
                                            for j in 0..src.len() {
                                                f[j] -= src[j];
                                            }
                                        }
                                        new_ready_nodes.insert(edge, Some(f));
                                    }
                                    Op::Mul => {
                                        let mut f = data.clone();
                                        for src in srcs.iter().skip(1) {
                                            for j in 0..src.len() {
                                                f[j] *= src[j];
                                            }
                                        }
                                        new_ready_nodes.insert(edge, Some(f));
                                    }
                                    Op::Div => {
                                        let mut f = data.clone();
                                        for src in srcs.iter().skip(1) {
                                            for j in 0..src.len() {
                                                f[j] /= src[j];
                                            }
                                        }
                                        new_ready_nodes.insert(edge, Some(f));
                                    }
                                    Op::Log => {
                                        let mut f = data.clone();
                                        for i in &mut f {
                                            *i = i.ln();
                                        }
                                        new_ready_nodes.insert(edge, Some(f));
                                    }
                                    Op::Exp => {
                                        let mut f = data.clone();
                                        for i in &mut f {
                                            *i = i.exp();
                                        }
                                        new_ready_nodes.insert(edge, Some(f));
                                    }
                                }
                            } else {
                                let x: &'static str = self.op_nodes[&edge].into();
                                println!(
                                    "Op: {} Sources: {} Expected Sources: {}",
                                    x,
                                    srcs.len(),
                                    reverse_edges[&edge].len()
                                );
                            }
                        }
                    }
                }
            }
            for (id, node) in new_ready_nodes {
                ready_nodes.insert(id, node);
            }
            new_ready_nodes = HashMap::default();
        }

        ExecutedGraph {
            tensors: ready_nodes
                .into_iter()
                .flat_map(|(id, data)| data.map(|data| (id, data)))
                .collect(),
        }
    }

    // Convert to petgraph
    fn petgraph(&self) -> petgraph::Graph<String, bool, petgraph::Directed, u32> {
        let mut graph = petgraph::Graph::new();
        let mut nodes = HashMap::new();

        // Initial tensors
        for (i, (id, _)) in self.tensors.iter().enumerate() {
            nodes.insert(*id, graph.add_node((i + 1).to_string()));
        }

        // Operations
        for (id, op) in &self.op_nodes {
            let op: &'static str = op.into();
            nodes.insert(*id, graph.add_node(op.to_string()));
        }

        // Edges
        for (from, to) in &self.connections {
            for to in to {
                graph.add_edge(nodes[from], nodes[to], false);
            }
        }

        graph
    }
}

fn display_petgraph(graph: &petgraph::Graph<String, bool, petgraph::Directed, u32>) {
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
    fn join(self, rhs: &petgraph::Graph<String, bool, petgraph::Directed, u32>) -> Self;
}

impl JoinGraph for petgraph::Graph<String, bool, petgraph::Directed, u32> {
    fn join(mut self, rhs: &petgraph::Graph<String, bool, petgraph::Directed, u32>) -> Self {
        let mut id_map = HashMap::new(); // We track the node id remapping here so they don't overlap
        for (index, node) in rhs.node_indices().zip(rhs.node_weights()) {
            id_map.insert(index, self.add_node(node.clone()));
        }

        for edge in rhs.raw_edges() {
            self.add_edge(id_map[&edge.source()], id_map[&edge.target()], edge.weight);
        }

        self
    }
}
