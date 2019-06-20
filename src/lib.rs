use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::str::FromStr;
use std::string::ParseError;

/// This defines the acceptable attribute type: Int, Float Or Text
/// We define the type that does not exist in the enumeration as a String.
/// You can refine String to some more specific type.
#[derive(Clone)]
pub enum Attr {
    Int(i64),
    Float(f64),
    Text(String),
    //...other types
}
/// Parsing `Attr` from file format attributes.
///
/// In the function `from_str`, we take a String as a input, and trying to parsing it to the types
/// defined in enum `Attr` in proper order. If we can't parse it to any types above, we regard this
/// attribute as a plain text(as a String).
/// Maybe we should drop the attributes we can't parse. That should have our actual needs to decide.
/// While we just regard it as a string here.
///
/// If there are no hints for the attributes' type, we can only verifying the origin type of
/// the attributes by trying to parsing string to a type below successively.
///
/// Maybe we can do some pretreatments for the data file by the prior knowledge.
impl FromStr for Attr {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.parse::<i64>().is_ok() {
            return Ok(Attr::Int(s.parse().unwrap()));
        }
        if s.parse::<f64>().is_ok() {
            return Ok(Attr::Int(s.parse().unwrap()));
        }
        return Ok(Attr::Text(s.to_string()));
    }
}

#[derive(Clone)]
pub enum GraphType {
    Undirected,
    Directed,
}

#[derive(Clone)]
struct Edge<T> {
    source: T,
    target: T,
    attrs: HashMap<String, Attr>,
}

#[derive(Clone)]
struct Vertex<T> {
    id: T,
    attrs: HashMap<String, Attr>,
}

impl<T> Vertex<T> {
    fn new(id: T) -> Vertex<T> {
        Vertex {
            id,
            attrs: HashMap::new(),
        }
    }
}

/// Structure `Graph` used to store a graph structure(edges,vertices,etc) and some basic infomations.
///
/// The constructor designed by `Builder Pattern`,so you need to call function continuously for
/// supplying required informations when constructing a `Graph` instance.
///
/// The structure `Graph` used `vec` to store edges and nodes and using a `Hashmap` for indexing
/// node's neighbours pos in `vec`.(key = node's id, value = (start index in vec,end index in vec)).
///
/// Here are the descriptions for the fields in structure `Graph`:
///     1. graph_type: identifing the graph is directed or not,
///     2. schema(Unimplemented): pre-defined the attributes' name for nodes and edges,
///     3. node_file: optional csv node file path,
///     4. edge_file: csv edge file path,
///     5. separate:  csv files separate where each term is separated with sep,
///     6. vertices: `vec` for storing nodes which reading from the node file with separate,
///     7. edges: `vec` for storing edges which reading from the edge file with separate,
///     8. neighbour_index: `HashMap` for indexing node's neighbours pos in `vec`.
///
/// Here are the descriptions for the functions:
///     1. default(): create a default graph data;
///     2. with_dir(GraphType::Directed(or Undirected)): the graph constructing is directed or not;
///     3. with_schema(): identifing the schema of the graph;
///     4. from_file(edge_file:String,node_file:Option<String>,sep:String): node_file path,
///             and optional edge file path, both are csv files where each term is separated with sep.
///     5. build(): return a graph instance according to the iniformations gave above.
///
/// # Example
///
/// ```
///  use graph_rust::{Graph, GraphType};
///  let graph = Graph::default()
///        .with_dir(GraphType::Directed)
///        .with_schema(String::from("..."))
///        .from_file(String::from("data_edges.csv"), None, ",")
///        .build();
/// ```
#[derive(Clone)]
pub struct Graph<T> {
    graph_type: GraphType,
    schema: String,
    node_file: Option<String>,
    edge_file: String,
    separate: String,
    vertices: Vec<Vertex<T>>,
    edges: Vec<Edge<T>>,
    neighbour_index: HashMap<T, (usize, usize)>,
}

impl<T> Graph<T>
    where
        T: Eq + Hash + FromStr + Display + Clone + Copy,
        <T as FromStr>::Err: Debug,
{
    pub fn default() -> Graph<T> {
        return Graph {
            graph_type: GraphType::Directed,
            schema: "".to_string(),
            node_file: None,
            edge_file: "".to_string(),
            separate: "".to_string(),
            vertices: vec![],
            edges: vec![],
            neighbour_index: HashMap::new(),
        };
    }

    pub fn with_dir(&mut self, dir: GraphType) -> &mut Graph<T> {
        self.graph_type = dir;
        self
    }

    pub fn with_schema(&mut self, schema: String) -> &mut Graph<T> {
        self.schema = schema;
        self
    }

    pub fn from_file(
        &mut self,
        edge_file: String,
        node_file: Option<String>,
        sep: &str,
    ) -> &mut Graph<T> {
        self.node_file = node_file;
        self.edge_file = edge_file;
        self.separate = sep.to_string();
        self
    }

    pub fn build(&mut self) -> Graph<T> {
        //Init node and edge container
        let mut vertices = HashMap::new();
        let mut edges = HashMap::new();

        let directed = match self.graph_type {
            GraphType::Directed => true,
            GraphType::Undirected => false,
        };

        let no_node_file = match self.node_file {
            None => true,
            _ => false,
        };

        // Open file and read line by line
        let edge_file = File::open(&self.edge_file).expect("Please check `edge_file` path");

        let buf_reader = BufReader::new(edge_file);
        for origin_line in buf_reader.lines() {
            // Split file data into structure
            if let Err(_e) = origin_line {
                continue;
            }
            let line = origin_line.unwrap();
            let edge_info: Vec<&str> = line.split(&self.separate).collect();
            let source: T = edge_info[0].parse().unwrap();
            let target: T = edge_info[1].parse().unwrap();

            // Generate edge
            let mut attrs: HashMap<String, Attr> = HashMap::new();
            for i in 2..edge_info.len() {
                let attr: Vec<&str> = edge_info[i].split("=").collect();
                // Invalidate line data: Skip
                if attr.len() != 2 {
                    continue;
                }
                attrs.insert(attr[0].to_string(), attr[1].parse().unwrap());
            }

            // Update edge set
            if !directed {
                update_edge_set(&mut edges, target, source, attrs.clone())
            }
            update_edge_set(&mut edges, source, target, attrs);

            if no_node_file {
                if !vertices.contains_key(&source) {
                    vertices.insert(source, Vertex::new(source));
                }
                if !vertices.contains_key(&target) {
                    vertices.insert(target, Vertex::new(target));
                }
            }
        }

        //Read vertices if node_file not `None`
        if !no_node_file {
            let path = self.node_file.clone().unwrap();
            let node_file = File::open(path).expect("Please check `node_file` path");

            let buf_reader = BufReader::new(node_file);
            for origin_line in buf_reader.lines() {
                // Split file data into structure
                if let Err(_e) = origin_line {
                    continue;
                }

                let line = origin_line.unwrap();
                let edge_info: Vec<&str> = line.split(&self.separate).collect();
                let id: T = edge_info[0].parse().unwrap();

                // Generate node
                let mut attrs: HashMap<String, Attr> = HashMap::new();
                for i in 1..edge_info.len() {
                    let attr: Vec<&str> = edge_info[i].split("=").collect();
                    // Invalidate line data: Skip
                    if attr.len() != 2 {
                        continue;
                    }
                    attrs.insert(attr[0].to_string(), attr[1].parse().unwrap());
                }
                vertices.insert(id, Vertex { id, attrs });
            }
        }

        let mut g = Graph {
            graph_type: self.graph_type.clone(),
            schema: self.schema.clone(),
            node_file: self.node_file.clone(),
            edge_file: self.edge_file.clone(),
            separate: self.separate.clone(),
            vertices: vec![],
            edges: vec![],
            neighbour_index: HashMap::new(),
        };

        // Generate edge index
        for (id, v) in vertices {
            let edge_opt = edges.get_mut(&id);
            if let None = edge_opt {
                continue;
            }
            let mut edge_vec = edge_opt.unwrap();
            g.neighbour_index.insert(
                id,
                (
                    self.neighbour_index.len(),
                    edge_vec.len() + self.neighbour_index.len(),
                ),
            );
            g.edges.append(&mut edge_vec);
            g.vertices.push(v);
        }
        return g;
    }

    pub fn bfs(&self, source: T) {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        let edge_index = self.neighbour_index.get(&source);
        if let None = edge_index {
            return;
        }
        queue.push_back(source.clone());
        visited.insert(source);
        while !queue.is_empty() {
            let v = queue.pop_front().unwrap();
            print!("{}->", v);
            let neighbour_index_opt = self.neighbour_index.get(&v);
            if let None = neighbour_index_opt {
                continue;
            }
            let (neighbour_index, neighbour_end) = neighbour_index_opt.unwrap();
            for i in neighbour_index.clone()..neighbour_end.clone() {
                let edge = self.edges.get(i).unwrap();
                if visited.contains(&edge.target) {
                    continue;
                }
                queue.push_back(edge.target.clone());
                visited.insert(edge.target.clone());
            }
        }
    }

    pub fn dfs(&self, source: T) {
        let mut stack = VecDeque::new();
        let mut visited = HashSet::new();

        print!("{}->", source);
        visited.insert(source.clone());
        stack.push_front(source);

        while !stack.is_empty() {
            let top = stack.front();
            if let None = top {
                continue;
            }
            let pre = top.unwrap();

            let neighbour_index_opt = self.neighbour_index.get(pre);
            if let None = neighbour_index_opt {
                stack.pop_front();
                continue;
            }
            let (neighbour_index, neighbour_end) = neighbour_index_opt.unwrap();
            let mut i = neighbour_index.clone();
            while i < neighbour_end.clone() {
                let cur = self.edges.get(i).unwrap();
                if !visited.contains(&cur.target) {
                    print!("{}->", cur.target);
                    visited.insert(cur.target.clone());
                    stack.push_front(cur.target.clone());
                    break;
                }
                i += 1;
            }
            if i == neighbour_end.clone() {
                stack.pop_front();
            }
        }
    }
}

/// Assistance function for initial edge set
fn update_edge_set<T>(
    edges: &mut HashMap<T, Vec<Edge<T>>>,
    from: T,
    target: T,
    attrs: HashMap<String, Attr>,
) where
    T: Eq + Hash + Clone,
{
    match edges.get(&from) {
        Some(_v) => {
            let mut edge_vec = edges.remove(&from).unwrap();
            edge_vec.push(Edge {
                source: from.clone(),
                target,
                attrs,
            });
            edges.insert(from.clone(), edge_vec);
        }
        None => {
            let mut vec = Vec::new();
            vec.push(Edge {
                source: from.clone(),
                target,
                attrs,
            });
            edges.insert(from.clone(), vec);
        }
    }
}

/// Structure `Bfs` used to bind a graph and visit nodes of the bind graph in a breadth-first-search (BFScc).
///
/// The Structure implements trait `Iterator`, so you can use a iterator style for accessing node.
///
/// This algorithm binding a reference of the first given param graph which is the target traversal graph.
///
/// This algorithm start at the second given param `start` and traverses nodes by BFS.
///
/// Traversal status are storing in structure `Bfs`.
///
/// # Example
///
/// ```
///  use graph_rust::{Bfs, Graph, GraphType};
///  let graph = Graph::default()
///        .with_dir(GraphType::Directed)
///        .with_schema(String::from("..."))
///        .from_file(String::from("data_edges.csv"), None, ",")
///        .build();
///  let mut bfs = Bfs::new(&graph, 1);
///  while let Some(node) = bfs.next() {
///        print!("{}->", node);
///  }
/// ```
pub struct Bfs<'a, T> {
    pub bind_graph: &'a Graph<T>,
    pub stack: VecDeque<T>,
    pub visited: HashSet<T>,
}

impl<'a, T> Bfs<'a, T>
    where
        T: Copy + PartialEq + Eq + Hash,
{
    pub fn new(graph: &'a Graph<T>, start: T) -> Self {
        let mut bfs = Bfs {
            bind_graph: graph,
            stack: VecDeque::new(),
            visited: HashSet::new(),
        };
        bfs.visited.insert(start);
        bfs.stack.clear();
        bfs.stack.push_front(start);
        bfs
    }
}

impl<'a, T> Iterator for Bfs<'a, T>
    where
        T: Copy + PartialEq + Eq + Hash,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let top_node_opt = self.stack.pop_front();
        if let None = top_node_opt {
            return None;
        }
        let node = top_node_opt.unwrap();
        let neighbour_index_opt = self.bind_graph.neighbour_index.get(&node);
        if let None = neighbour_index_opt {
            return Some(node);
        }
        let (neighbour_index, neighbour_end) = neighbour_index_opt.unwrap();
        let mut i = neighbour_index.clone();
        while i < neighbour_end.clone() {
            let cur = self.bind_graph.edges.get(i).unwrap();
            if !self.visited.contains(&cur.target) {
                self.visited.insert(cur.target);
                self.stack.push_back(cur.target);
            }
            i += 1;
        }
        return Some(node);
    }
}

/// Structure `Dfs` used to bind a graph and visit nodes of the bind graph in a depth-first-search (DFS).
///
/// The Structure implements trait `Iterator`, so you can use a iterator style for accessing node.
///
/// This algorithm binding a reference of the first given param graph which is the target traversal graph.
///
/// This algorithm start at the second given param `start` and traverses nodes by DFS.
///
/// Traversal status are storing in structure `Dfs`.
///
/// # Example
///
/// ```
///  use graph_rust::{Dfs, Graph, GraphType};
///  let graph = Graph::default()
///        .with_dir(GraphType::Directed)
///        .with_schema(String::from("..."))
///        .from_file(String::from("data_edges.csv"), None, ",")
///        .build();
///  let mut dfs = Dfs::new(&graph, 1);
///  while let Some(node) = dfs.next() {
///        print!("{}->", node);
///  }
/// ```
pub struct Dfs<'a, T> {
    pub bind_graph: &'a Graph<T>,
    pub stack: Vec<T>,
    pub visited: HashSet<T>,
}

impl<'a, T> Dfs<'a, T>
    where
        T: Copy + PartialEq + Eq + Hash,
{
    pub fn new(graph: &'a Graph<T>, start: T) -> Self {
        let mut dfs = Dfs {
            bind_graph: graph,
            stack: Vec::new(),
            visited: HashSet::new(),
        };
        dfs.visited.insert(start);
        dfs.stack.clear();
        dfs.stack.push(start);
        dfs
    }

    pub fn next(&mut self) -> Option<T> {
        let top_node_opt = self.stack.pop();
        if let None = top_node_opt {
            return None;
        }
        let node = top_node_opt.unwrap();
        let neighbour_index_opt = self.bind_graph.neighbour_index.get(&node);
        if let None = neighbour_index_opt {
            return Some(node);
        }
        let (neighbour_index, neighbour_end) = neighbour_index_opt.unwrap();
        let mut i = neighbour_index.clone();
        while i < neighbour_end.clone() {
            let cur = self.bind_graph.edges.get(i).unwrap();
            if !self.visited.contains(&cur.target) {
                self.visited.insert(cur.target);
                self.stack.push(cur.target);
            }
            i += 1;
        }
        return Some(node);
    }
}

impl<'a, T> Iterator for Dfs<'a, T>
    where
        T: Copy + PartialEq + Eq + Hash,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let top_node_opt = self.stack.pop();
        if let None = top_node_opt {
            return None;
        }
        let node = top_node_opt.unwrap();
        let neighbour_index_opt = self.bind_graph.neighbour_index.get(&node);
        if let None = neighbour_index_opt {
            return Some(node);
        }
        let (neighbour_index, neighbour_end) = neighbour_index_opt.unwrap();
        let mut i = neighbour_index.clone();
        while i < neighbour_end.clone() {
            let cur = self.bind_graph.edges.get(i).unwrap();
            if !self.visited.contains(&cur.target) {
                self.visited.insert(cur.target);
                self.stack.push(cur.target);
            }
            i += 1;
        }
        return Some(node);
    }
}

///TODO: Other traversal algorithms here
fn other_traversal_algorithms() {}
