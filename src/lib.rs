#[macro_use]
extern crate serde_derive;

use csv::{Position, ReaderBuilder};
use csv_index::RandomAccessSimple;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::io::Cursor;
use std::str::FromStr;
use std::string::ParseError;
use std::sync::mpsc::{channel, Receiver, Sender};

/// This defines the acceptable attribute type: Int, Float Or Text
/// We define the type that does not exist in the enumeration as a String.
/// You can refine String to some more specific type.
#[derive(Clone, Deserialize)]
#[serde(untagged)]
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
/// Maybe we should drop the attributes we can't parse. That should have our actual requirements to decide.
/// While we just regard it as a string here.
///
/// If there are no hints for the attributes' type, we can only verifying the origin type of
/// the attributes by trying to parsing string to a type below successively.
///
/// Maybe we can do some pretreatments for the data file by the prior knowledge to speedup the process.
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

#[derive(Clone, Deserialize)]
struct Edge<T> {
    source: T,
    target: T,
    attrs: Vec<Attr>,
}

#[derive(Clone, Deserialize)]
struct Vertex<T> {
    id: T,
    attrs: Vec<Attr>,
}

impl<T> Vertex<T> {
    fn new(id: T) -> Vertex<T> {
        Vertex { id, attrs: vec![] }
    }
}

/// Structure `Graph` used to store a graph structure(edges,vertices,etc) and some basic infomations.
///
/// The constructor designed by `Builder Pattern`,so you need to call function continuously for
/// supplying required information when constructing a `Graph` instance.
///
/// The structure `Graph` used `vec` to store edges and nodes and using a `Hashmap` for indexing
/// node's neighbours pos in `vec`.(key = node's id, value = (start index in vec,end index in vec)).
///
/// Here are the descriptions for the fields in structure `Graph`:
///     1. graph_type: identifying the graph is directed or not,
///     2. vertices: `vec` for storing nodes which reading from the node file with separate,
///     3. edges: `vec` for storing edges which reading from the edge file with separate,
///     4. vertex_key: the key name set for vertex attributes,
///     5. edge_key: the key name set for edge attributes.
///     6. vertex_index: `HashMap` for indexing nodes by id and return pos in vector `vertices`,
///     7. neighbour_index: `HashMap` for indexing node's neighbours start and end index by
///                         source's id, and return pos in vector `edges`,
///
#[derive(Clone)]
pub struct Graph<T> {
    graph_type: GraphType,
    vertices: Vec<Vertex<T>>,
    edges: Vec<Edge<T>>,
    vertex_key: Vec<String>,
    edge_key: Vec<String>,
    vertex_index: HashMap<T, usize>,
    neighbour_index: HashMap<T, (usize, usize)>,
}

/// Structure `GraphReader` used to read a graph csv files in parallel to construct a graph structure
/// (edges,vertices,etc) and store some basic infomations about .
///
/// The constructor designed by `Builder Pattern`,so you need to call function continuously for
/// supplying required informations when constructing a `Graph` instance.
///
/// Here are the descriptions for the fields in structure `GraphReader`:
///     1. graph_type: identifing the graph is directed or not,
///     2. schema(Unimplemented): pre-defined the attributes' name for nodes and edges,
///     3. node_file_path: optional csv node file path,
///     4. edge_file_path: csv edge file path,
///     5. separate: csv files separate where each term is separated with sep,
///     6. thread_pool_size: thread pool size for graph reader(default value is 4),
///     7. thread_pool: maintain a thread pool for reading huge file in parallel,
///
/// Here are the descriptions for the functions:
///     1. default(): create a default `GraphReader`,
///     2. with_dir(GraphType::Directed(or Undirected)): the graph reading(constructing) now is directed or not;
///     3. with_schema(): identifing the schema of the graph,
///     4. from_file(edge_file:String,node_file:Option<String>,sep:String): node_file path,
///             and optional edge file path, both are csv files where each term is separated with sep,
///     5. with_thread(): reading graph files in parallel with how much threads,
///     6. build(): return a graph instance according to the iniformations gave above.
///
/// # Example
///
/// ```
///  use graph_rust::{GraphReader, GraphType};
///  let g = GraphReader::default()
///    .with_dir(GraphType::Directed)
///    .with_schema(String::from("..."))
///    .from_file(String::from("data_edges.csv"), None, b',')
///    .with_thread(4)
///    .build::<i32>();
///
/// ```
pub struct GraphReader {
    schema: String,
    separate: u8,
    graph_type: GraphType,
    edge_file_path: String,
    node_file_path: Option<String>,
    thread_pool_size: u64,
    thread_pool: ThreadPool,
}

impl GraphReader {
    pub fn default() -> GraphReader {
        return GraphReader {
            schema: "".to_string(),
            node_file_path: None,
            edge_file_path: "".to_string(),
            separate: b',',
            graph_type: GraphType::Directed,
            thread_pool_size: 4,
            thread_pool: ThreadPoolBuilder::new().num_threads(4).build().unwrap(),
        };
    }

    pub fn with_dir(&mut self, dir: GraphType) -> &mut GraphReader {
        self.graph_type = dir;
        self
    }

    pub fn with_thread(&mut self, thread_size: u64) -> &mut GraphReader {
        self.thread_pool_size = thread_size;
        self
    }

    pub fn with_schema(&mut self, schema: String) -> &mut GraphReader {
        self.schema = schema;
        self
    }

    pub fn from_file(
        &mut self,
        edge_file: String,
        node_file: Option<String>,
        sep: u8,
    ) -> &mut GraphReader {
        self.node_file_path = node_file;
        self.edge_file_path = edge_file;
        self.separate = sep;
        self
    }

    /// Call function `build` to generate a instance of `Graph` in parallel.
    /// In this function use `thread_pool_size` threads for reading edge file and node file.
    /// You can modify the `thread_pool_size` by method `with_thread(thread_size: u64)`,while
    /// the param `thread_size` need to be larger than 1.
    ///
    /// The basic process in this method as follows:
    /// 1. Set up thread pool with the configurations provided by user,
    /// 2. Build a empty instance of `Graph` for filling later,
    /// 3. Create index for edge file used to separate file for threads,
    ///      and obtains edge key names from file,
    /// 4. Construct channel for threads' communication,
    /// 5. Distribute reading task for threads and run them,
    /// 6. Collecting thread's result on last thread in pool,
    /// 7. (Optional) Reading vertices file by parallel if there exist,
    /// 8. Generating index for vertices vector and edges vector.
    pub fn build<T>(&mut self) -> Graph<T>
    where
        T: Eq + Hash + FromStr + Display + Clone + Copy + Sync + Send + Ord,
        <T as FromStr>::Err: Debug,
        for<'de> T: Deserialize<'de>,
    {
        self.thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.thread_pool_size as usize)
            .build()
            .unwrap();

        let mut g = Graph {
            graph_type: self.graph_type.clone(),
            vertices: vec![],
            edges: vec![],
            vertex_key: vec![],
            edge_key: vec![],
            neighbour_index: HashMap::new(),
            vertex_index: HashMap::new(),
        };

        let directed = match self.graph_type {
            GraphType::Directed => true,
            GraphType::Undirected => false,
        };
        let has_node_file = self.node_file_path.is_some();

        let sep = self.separate;
        let edge_file = &self.edge_file_path;
        // preserve a thread for collecting
        let worker_count = self.thread_pool_size - 1;

        // 0.Create index for separating file
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .delimiter(sep)
            .from_path(edge_file)
            .expect("Please check `edge_file` path");
        let mut wtr = Cursor::new(vec![]);
        RandomAccessSimple::create(&mut rdr, &mut wtr);
        let mut idx = RandomAccessSimple::open(wtr).unwrap();
        // Obtains key names from file
        for key in rdr.headers().unwrap().iter() {
            g.edge_key.push(key.to_string());
        }

        // 1. Construct channel for threads' communication,
        //    Transmit Box pointer here to avoid large memory copy.
        let (tx, rx) = channel();

        // 2. Distribute reading task for threads.
        let job_size = idx.len() / worker_count + 1;
        let mut real_worker_count = worker_count;
        for i in 0..worker_count {
            let tx = tx.clone();
            let start_index = job_size * i + 1;
            // Break distribute when no more jobs
            if start_index >= idx.len() {
                real_worker_count = i;
                break;
            }
            let start_pos = idx.get(start_index as u64).unwrap();
            self.thread_pool.install(move || {
                read_edge_file_task(
                    sep,
                    &edge_file,
                    start_pos,
                    job_size,
                    directed,
                    has_node_file,
                    tx,
                );
            });
        }

        // 3. Collecting thread's result.
        let mut vertices: HashMap<T, Vertex<T>> = HashMap::new();
        let mut edges: HashMap<T, Vec<Edge<T>>> = HashMap::new();
        let (tx_collect, rx_collect) = channel();
        self.thread_pool
            .install(|| collect_edge_task(has_node_file, real_worker_count, rx, tx_collect));
        let (vertices_rec, edges_rec) = rx_collect.recv().unwrap();
        vertices = *vertices_rec;
        edges = *edges_rec;

        if has_node_file {
            let node_file_path = self.node_file_path.clone().unwrap();
            // 0.Create index for node file
            let mut rdr = ReaderBuilder::new()
                .has_headers(true)
                .delimiter(sep)
                .from_path(&node_file_path)
                .expect("Please check `node_file` path");
            let mut wtr = Cursor::new(vec![]);
            RandomAccessSimple::create(&mut rdr, &mut wtr);
            let mut idx = RandomAccessSimple::open(wtr).unwrap();
            // Obtains key names from file
            for key in rdr.headers().unwrap().iter() {
                g.vertex_key.push(key.to_string());
            }

            // 1. Construct channel for threads' communication.
            let (tx, rx) = channel();

            // 2. Distribute reading task for threads.
            let job_size = idx.len() / worker_count + 1;
            for i in 0..self.thread_pool_size {
                let tx = tx.clone();
                let start_index = job_size * i + 1;
                if start_index >= idx.len() {
                    real_worker_count = i;
                    break;
                }
                let start_pos = idx.get(start_index).unwrap();
                self.thread_pool
                    .install(|| read_node_file_task(sep, &node_file_path, start_pos, job_size, tx));
            }

            // 3. Collecting thread's result.
            let (tx_collect, rx_collect) = channel();
            self.thread_pool
                .install(|| collect_node_task(real_worker_count, rx, tx_collect));
            vertices = *rx_collect.recv().unwrap();
        }

        // Generating index
        let mut cur_index = 0;
        for (id, v) in vertices {
            let edge_opt = edges.get_mut(&id);
            g.vertices.push(v);
            if edge_opt.is_none() {
                continue;
            }
            let edge_vec: &mut Vec<Edge<T>> = edge_opt.unwrap();
            g.neighbour_index
                .insert(id, (cur_index, cur_index + edge_vec.len()));
            cur_index = cur_index + edge_vec.len();
            g.edges.append(edge_vec);
        }
        g
    }
}

fn read_edge_file_task<T>(
    sep: u8,
    edge_file: &String,
    start_pos: Position,
    job_size: u64,
    directed: bool,
    node_file: bool,
    tx: Sender<(Box<HashMap<T, Vertex<T>>>, Box<HashMap<T, Vec<Edge<T>>>>)>,
) where
    T: Eq + Hash + FromStr + Display + Clone + Copy,
    for<'de> T: Deserialize<'de>,
    <T as FromStr>::Err: Debug,
{
    let mut vertices = Box::new(HashMap::new());
    let mut edges = Box::new(HashMap::new());

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(sep)
        .from_path(edge_file)
        .expect("Please check `edge_file` path");
    rdr.seek(start_pos).expect("File split error!");

    let mut iter = rdr.deserialize();
    for _ in 0..job_size {
        let origin_line = iter.next().unwrap();
        if origin_line.is_err() {
            println!("{}", origin_line.err().unwrap());
            continue;
        }

        let edge_info: Edge<T> = origin_line.unwrap();
        let source = edge_info.source;
        let target = edge_info.target;
        let attrs = edge_info.attrs;

        if !directed {
            add_edge(&mut edges, target, source, attrs.clone());
        }
        add_edge(&mut edges, source, target, attrs.clone());

        if !node_file {
            if !vertices.contains_key(&edge_info.source) {
                vertices.insert(edge_info.source, Vertex::new(edge_info.source));
            }
            if !vertices.contains_key(&edge_info.target) {
                vertices.insert(edge_info.target, Vertex::new(edge_info.target));
            }
        }
    }
    tx.send((vertices, edges));
}

fn read_node_file_task<T>(
    sep: u8,
    node_file: &String,
    start_index: Position,
    job_size: u64,
    tx: Sender<Box<HashMap<T, Vertex<T>>>>,
) where
    T: Eq + Hash + FromStr + Display + Copy,
    for<'de> T: Deserialize<'de>,
    <T as FromStr>::Err: Debug,
{
    let mut vertices = Box::new(HashMap::new());

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(sep)
        .from_path(node_file)
        .expect("Please check `node_file` path");
    rdr.seek(start_index).expect("File split error!");

    let mut iter = rdr.deserialize();
    for _ in 0..job_size {
        let origin_line = iter.next().unwrap();
        if origin_line.is_err() {
            println!("{}", origin_line.err().unwrap());
            continue;
        }

        let v: Vertex<T> = origin_line.unwrap();
        vertices.insert(v.id, v);
    }
    tx.send(vertices);
}

fn collect_node_task<T>(
    job_count: u64,
    rx: Receiver<Box<HashMap<T, Vertex<T>>>>,
    tx: Sender<Box<HashMap<T, Vertex<T>>>>,
) where
    T: Eq + Hash,
{
    let mut vertices = Box::new(HashMap::new());
    for _ in 0..job_count {
        let mut vertices_rec = rx.recv().unwrap();
        for (id, v) in *vertices_rec {
            vertices.insert(id, v);
        }
    }
    tx.send(vertices);
}

fn collect_edge_task<T>(
    has_node_file: bool,
    job_count: u64,
    rx: Receiver<(Box<HashMap<T, Vertex<T>>>, Box<HashMap<T, Vec<Edge<T>>>>)>,
    tx: Sender<(Box<HashMap<T, Vertex<T>>>, Box<HashMap<T, Vec<Edge<T>>>>)>,
) where
    T: Eq + Hash,
{
    let mut vertices = Box::new(HashMap::new());
    let mut edges = Box::new(HashMap::new());
    for _ in 0..job_count {
        let (vertices_rec, edges_rec) = rx.recv().unwrap();
        if !has_node_file {
            for (id, v) in *vertices_rec {
                vertices.insert(id, v);
            }
        }
        for (id, vec) in *edges_rec {
            let edge_vec = edges.entry(id).or_insert(Vec::new());
            for e in vec {
                edge_vec.push(e);
            }
        }
    }
    tx.send((vertices, edges));
}

/// Assistance function for initial edge set
fn add_edge<T>(edges: &mut HashMap<T, Vec<Edge<T>>>, from: T, target: T, attrs: Vec<Attr>)
where
    T: Eq + Hash + Clone + Copy,
{
    let neighbors = edges.entry(from).or_insert_with(Vec::new);
    neighbors.push(Edge {
        source: from,
        target,
        attrs,
    });
}

/// Structure `Bfs` used to bind a graph and visit nodes of the bind graph in a breadth-first-seBoxh (BFScc).
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
///  use graph_rust::{Bfs, GraphReader, GraphType};
///  let graph = GraphReader::default()
///        .with_dir(GraphType::Directed)
///        .with_schema(String::from("..."))
///        .from_file(String::from("data_edges.csv"), None, b',')
///        .build();
///  let mut bfs = Bfs::new(&graph, 1);
///  while let Some(node) = bfs.next() {
///        print!("{}->", node);
///  }
/// ```
pub trait Reset {
    fn reset(&mut self);
}

pub struct Bfs<'a, T> {
    pub bind_graph: &'a Graph<T>,
    pub stack: VecDeque<T>,
    pub visited: HashSet<T>,
    pub start: T,
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
            start,
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
        if let Some(node) = self.stack.pop_front() {
            let neighbour_index_opt = self.bind_graph.neighbour_index.get(&node);
            if neighbour_index_opt.is_none() {
                return Some(node);
            }
            let (neighbour_index, neighbour_end) = neighbour_index_opt.unwrap();
            let mut i = neighbour_index.clone();
            let end_index = neighbour_end.clone();
            while i < end_index {
                let cur = self.bind_graph.edges.get(i).unwrap();
                if !self.visited.contains(&cur.target) {
                    self.visited.insert(cur.target);
                    self.stack.push_back(cur.target);
                }
                i += 1;
            }
            Some(node)
        } else {
            None
        }
    }
}

impl<'a, T> Reset for Bfs<'a, T>
where
    T: Copy + Eq + Hash,
{
    fn reset(&mut self) {
        self.visited.clear();
        self.visited.insert(self.start);
        self.stack.clear();
        self.stack.push_front(self.start);
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
///  use graph_rust::{Dfs, GraphReader, GraphType};
///  let graph = GraphReader::default()
///        .with_dir(GraphType::Directed)
///        .with_schema(String::from("..."))
///        .from_file(String::from("data_edges.csv"), None, b',')
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
    pub start: T,
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
            start,
        };
        dfs.visited.insert(start);
        dfs.stack.clear();
        dfs.stack.push(start);
        dfs
    }
}

impl<'a, T> Iterator for Dfs<'a, T>
where
    T: Copy + PartialEq + Eq + Hash,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.stack.pop() {
            let neighbour_index_opt = self.bind_graph.neighbour_index.get(&node);
            if neighbour_index_opt.is_none() {
                return Some(node);
            }
            let (neighbour_index, neighbour_end) = neighbour_index_opt.unwrap();
            let mut i = neighbour_index.clone();
            let end_index = neighbour_end.clone();
            while i < end_index {
                let cur = self.bind_graph.edges.get(i).unwrap();
                if !self.visited.contains(&cur.target) {
                    self.visited.insert(cur.target);
                    self.stack.push(cur.target);
                }
                i += 1;
            }
            return Some(node);
        } else {
            None
        }
    }
}

impl<'a, T> Reset for Dfs<'a, T>
where
    T: Copy + Eq + Hash,
{
    fn reset(&mut self) {
        self.visited.clear();
        self.visited.insert(self.start);
        self.stack.clear();
        self.stack.push(self.start);
    }
}

///TODO(Yu Chen):Other traversal algorithms here
fn other_traversal_algorithms() {}

/// Test case here
#[cfg(test)]
mod tests {
    use crate::{Bfs, Dfs, GraphReader, GraphType};
    use std::cmp::Ordering;

    #[test]
    fn file_reading_test() {
        let g = GraphReader::default()
            .with_dir(GraphType::Directed)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges.csv"),
                Option::from(String::from("data_vertices.csv")),
                b',',
            )
            .with_thread(10)
            .build::<i32>();
        let mut vec = vec![];
        for v in g.vertices {
            vec.push(v.id);
        }
        vec.sort();
        assert_eq!(vec, vec![1, 2, 3, 4]);

        let mut edges = vec![];
        for e in g.edges {
            edges.push((e.source, e.target));
        }
        edges.sort_by(|(s1, t1), (s2, t2)| {
            if s1 == s2 {
                if t1 == t2 {
                    return Ordering::Equal;
                }
                return t1.cmp(t2);
            }
            return s1.cmp(s2);
        });
        assert_eq!(edges, vec![(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4)]);
    }

    #[test]
    fn dfs_iter_test() {
        let g = GraphReader::default()
            .with_dir(GraphType::Directed)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges.csv"),
                Option::from(String::from("data_vertices.csv")),
                b',',
            )
            .with_thread(10)
            .build();
        let dfs = Dfs::new(&g, 1);
        let res: Vec<i32> = dfs.collect();
        assert_eq!(res, vec![1, 4, 3, 2]);
    }

    #[test]
    fn bfs_iter_test() {
        let g = GraphReader::default()
            .with_dir(GraphType::Directed)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges.csv"),
                Option::from(String::from("data_vertices.csv")),
                b',',
            )
            .with_thread(10)
            .build();
        let bfs = Bfs::new(&g, 1);
        let res: Vec<i32> = bfs.collect();
        assert_eq!(res, vec![1, 2, 3, 4]);
    }

    //TODO(Yu Chen): allowing build graph in memory(from_reader()) and testing different graph data.
    #[test]
    fn build_graph_in_memory_test() {}
}
