#[macro_use]
extern crate serde_derive;

use csv::{Error, Position, ReaderBuilder};
use csv_index::RandomAccessSimple;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::io::Cursor;
use std::io::Write;
use std::str::FromStr;

mod errors;

use crate::errors::graph_build_error::BuildGraphError;
use errors::parse_attr_error::ParseAttrError;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use uuid::Uuid;

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
    type Err = ParseAttrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.parse::<i64>().is_ok() {
            return Ok(Attr::Int(s.parse()?));
        }
        if s.parse::<f64>().is_ok() {
            return Ok(Attr::Int(s.parse()?));
        }
        Ok(Attr::Text(s.to_string()))
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
pub struct Vertex<T> {
    id: T,
    attrs: Vec<Attr>,
}

impl<T> Vertex<T> {
    fn new(id: T) -> Vertex<T> {
        Vertex { id, attrs: vec![] }
    }
}
impl<T> PartialEq for Vertex<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Vertex<T>) -> bool {
        self.id == other.id
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

impl<T> Graph<T> {
    fn get_vertex(&self, id: &T) -> Option<Vertex<T>>
    where
        T: Eq + Hash + Clone,
    {
        let v_index = self.vertex_index.get(id).unwrap();
        if let Some(v) = self.vertices.get(*v_index) {
            return Some(v.clone());
        }
        None
    }

    fn get_neighbours(&self, id: &T) -> Vec<Vertex<T>>
    where
        T: Eq + Hash + Clone,
    {
        if let Some((start_pos, end_pos)) = self.neighbour_index.get(id) {
            let mut neighbours = vec![];
            let start_pos = start_pos.clone();
            let end_pos = end_pos.clone();
            for edge in start_pos..end_pos {
                let edge = self.edges[edge].clone();
                neighbours.push(self.get_vertex(&edge.target).unwrap());
            }
            return neighbours;
        }
        vec![]
    }
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
///  let g = GraphReader::default()?
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
    thread_pool: Vec<JoinHandle<()>>,
}

impl GraphReader {
    pub fn default() -> GraphReader {
        let g = GraphReader {
            schema: "".to_string(),
            node_file_path: None,
            edge_file_path: "".to_string(),
            separate: b',',
            graph_type: GraphType::Directed,
            thread_pool_size: 4,
            thread_pool: vec![],
        };
        g
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

    /// Construct graph from reader.(Only for testing.)
    /// For reusing the code, we write the data into a temp file,and reuse the origin algorithm.
    pub fn from_reader(
        &mut self,
        edge_buf: String,
        node_buf: Option<String>,
        sep: u8,
    ) -> &mut GraphReader {
        let edge_path_addr = format!("temp_edge_data_{}.txt", Uuid::new_v4());
        let edge_path = Path::new(&edge_path_addr);
        let mut edge_file = match File::create(&edge_path) {
            Err(_) => panic!("couldn't create {}", edge_path_addr),
            Ok(file) => file,
        };
        write!(edge_file, "{}", edge_buf);
        self.edge_file_path = edge_path_addr;
        if node_buf.is_some() {
            let node_path_addr = format!("temp_node_data_{}.txt", Uuid::new_v4());
            let node_path = Path::new(&node_path_addr);
            let mut node_file = match File::create(&node_path) {
                Err(_) => panic!("couldn't create {}", node_path_addr),
                Ok(file) => file,
            };
            write!(node_file, "{}", node_buf.unwrap());
            self.node_file_path = Some(node_path_addr);
        }
        self.separate = sep;
        self
    }

    /// Call function `build` to generate a instance of `Graph` in parallel.
    /// In this function use `thread_pool_size` threads for reading edge file and node file.
    /// You can modify the `thread_pool_size` by method `with_thread(thread_size: u64)`,while
    /// the param `thread_size` need to be larger than 1.
    ///
    /// The basic process in this method as follows:
    ///     1. Build a empty instance of `Graph` for filling later,
    ///     2. Read edge file in parallel by calling function `read_file_in_parallel`,
    ///     3. (Optional) Read node file in parallel if there exist a node file path,
    ///     4. Generating index for vertices vector and edges vector.
    ///
    /// # Feature Explanation
    /// 1. Storing layout of graph: in structure graph, we stores the edges and vertices in a
    ///     `Vec<Edge>` and `Vec<Vertex>`, so what will happen when we need to get a vertex neighbours
    ///     which id is 1? We need to search from the first position of Vec<Vertex> in graph, and
    ///     then we need to search from the first position to end of Vec<Edge> in graph. That's
    ///     cost a lot. So we generating two index for vertices and neighbours.
    /// 2. Vertex index used for indexing nodes by id and return the pos of vertex in vector `vertices`.
    ///     And neighbour_index used for indexing node's neighbours start and end index by source's
    ///     id, and return pos in vector `edges`,
    pub fn build<T>(&mut self) -> Result<Graph<T>, BuildGraphError>
    where
        T: Eq + Hash + FromStr + Display + Clone + Copy + Debug + Send,
        <T as FromStr>::Err: Debug,
        for<'de> T: Deserialize<'de>,
        T: 'static,
    {
        let mut g = Graph {
            graph_type: self.graph_type.clone(),
            vertices: vec![],
            edges: vec![],
            vertex_key: vec![],
            edge_key: vec![],
            neighbour_index: HashMap::new(),
            vertex_index: HashMap::new(),
        };

        let sep = self.separate;
        let has_node_file = self.node_file_path.is_some();

        // Read file in parallel
        let (mut vertices_arc, edges_arc) = read_file_in_parallel(
            &self.edge_file_path,
            sep,
            true,
            has_node_file,
            self.thread_pool_size,
            &mut g,
        )?;
        if has_node_file {
            let (vertices_rec, _edges) = read_file_in_parallel(
                &self.node_file_path.clone().unwrap(),
                sep,
                false,
                has_node_file,
                self.thread_pool_size,
                &mut g,
            )?;
            vertices_arc = vertices_rec;
        }

        // Generating index
        let mut vertices = Arc::try_unwrap(vertices_arc)?.into_inner()?;
        let mut edges = Arc::try_unwrap(edges_arc)?.into_inner()?;

        let mut cur_edge_index = 0;
        let mut cur_node_index = 0;
        for (id, v) in vertices {
            let edge_opt = edges.get_mut(&id);
            g.vertex_index.insert(v.id, cur_node_index);
            cur_node_index += 1;
            g.vertices.push(v);
            if edge_opt.is_none() {
                continue;
            }
            let edge_vec: &mut Vec<Edge<T>> = edge_opt.unwrap();
            g.neighbour_index
                .insert(id, (cur_edge_index, cur_edge_index + edge_vec.len()));
            cur_edge_index = cur_edge_index + edge_vec.len();
            g.edges.append(edge_vec);
        }
        Ok(g)
    }
}

/// Call function `read_file_in_parallel` to read vertex file or edge file and transform them
/// into `HashMap` by parallel.
///
/// # Parameters
/// Here are the descriptions for the parameters:
///     1. file_path: the csv file path which we need to read,
///     2. sep: the separator of the file to be read,
///     3. is_edge_file: the file to be read is for edge(true) or for node(false),
///     4. has_node_file: user configure `GraphReader` with a node file path(true) or not(false),
///     5. worker_count: the thread pool size which user configure in `GraphReader`,
///     6. graph: the graph for storing the vertices and edges key attributes.
///
/// # Responsibility
/// 1. Build csv file index for separating file reading task to threads.
/// 2. Calculate task size and start index for reading in threads.
/// 3. Waiting all thread finish jobs.
/// 4. Return vertices or edges set (HashMap) to caller.
///
/// # Feature Explanation
/// 1. In this function, we use shared-memory to process the communication between main thread with
///     sub-threads.
/// 2. We construct the type `Arc<Mutex<T>> for storing the reading result in different thread.
///     We use this structure for two reasons:
///     - `Arc` allow us to sharing memory with Thread Safety, while the container data in `Arc` can
///         not be modified.
///     - At the same time, the data stores in `Arc` need to be update with the reading process in
///         different thread. So, there exist race condition here.
///     So we use a `Mutex` to wrap the content data. Finally, we got a structure `Arc<Mutex<T>>`
/// 3. In this function, we use `join` to waiting for all sub-threads finished jobs on main thread.
fn read_file_in_parallel<T>(
    file_path: &String,
    sep: u8,
    is_edge_file: bool,
    has_node_file: bool,
    worker_count: u64,
    graph: &mut Graph<T>,
) -> Result<
    (
        Arc<Mutex<HashMap<T, Vertex<T>>>>,
        Arc<Mutex<HashMap<T, Vec<Edge<T>>>>>,
    ),
    Error,
>
where
    T: Eq + Hash + FromStr + Display + Copy + Debug + Send,
    for<'de> T: Deserialize<'de>,
    T: 'static,
    <T as FromStr>::Err: Debug,
{
    let vertices = Arc::new(Mutex::new(HashMap::new()));
    let edges = Arc::new(Mutex::new(HashMap::new()));

    let directed = match graph.graph_type {
        GraphType::Directed => true,
        GraphType::Undirected => false,
    };

    // Create index for separating file
    let mut idx = build_csv_index(file_path, sep, true, graph)?;

    // Calculate and distribute jobs index for threads.
    let job_size = idx.len() / worker_count + 1;
    let mut real_worker_count = worker_count;
    real_worker_count = (idx.len() - 1) / job_size;
    if (idx.len() - 1) % job_size != 0 {
        real_worker_count += 1;
    }
    let mut thread_pool = vec![];
    for i in 0..real_worker_count {
        let start_index = job_size * i + 1;
        let edge_file = file_path.clone();
        let start_pos = idx.get(start_index as u64)?;
        let vertices_arc = vertices.clone();
        let edges_arc = edges.clone();
        thread_pool.push(thread::spawn(move || {
            read_file_task(
                sep,
                edge_file,
                start_pos,
                job_size,
                directed,
                is_edge_file,
                has_node_file,
                vertices_arc,
                edges_arc,
            );
        }));
    }
    for t in thread_pool {
        t.join();
    }
    Ok((vertices, edges))
}

/// Call function `build_csv_index` to generate the index for the file in the parameter,
/// and reading the csv header stored in `Graph`. The function return the file index at last.
///
/// # Parameters
/// Here are the descriptions for the parameters:
///     1. file_path: the file which need to building index.
///     2. sep: the separator of the file to be read.
///     3. is_edge_file: the file for generating index is edge file(true) or node file(false).
///     4. graph: the `Graph` for storing csv header.
fn build_csv_index<T>(
    file_path: &String,
    sep: u8,
    is_edge_file: bool,
    graph: &mut Graph<T>,
) -> Result<RandomAccessSimple<Cursor<Vec<u8>>>, Error> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(sep)
        .from_path(file_path)
        .expect("Please check file path");
    let mut wtr = Cursor::new(vec![]);
    RandomAccessSimple::create(&mut rdr, &mut wtr);
    let idx = RandomAccessSimple::open(wtr)?;
    if is_edge_file {
        for key in rdr.headers()?.iter() {
            graph.edge_key.push(key.to_string());
        }
    } else {
        for key in rdr.headers()?.iter() {
            graph.vertex_key.push(key.to_string());
        }
    }
    Ok(idx)
}

///
/// In this function, reads file content in parallel and formats to specific structure of vertex
/// or edge with the configuration in `Graph` g.
///
/// # Parameters
/// Here are the descriptions for the parameters:
///     1. sep: the separator of the file to be read.
///     2. file_path: the file which need to building index.
///     3. start_pos: the task for reading file need to read start at which position.
///     4. job_size: the records size need to be read.
///     5. is_directed: the file reading now is directed or not (Only for edge file).
///     6. is_edge_file: the file for reading is edge file(true) or node file(false).
///     7. has_node_file: user configure `GraphReader` with a node file path(true) or not(false).
///         The function will generate a default vertex only with id if the parameter is false.
///     8. vertices: the point to a memory for storing the reading result of vertices.
///     9. edges: the point to a memory for storing the reading result of edges.
fn read_file_task<T>(
    sep: u8,
    file_path: String,
    start_pos: Position,
    job_size: u64,
    is_directed: bool,
    is_edge_file: bool,
    has_node_file: bool,
    vertices: Arc<Mutex<HashMap<T, Vertex<T>>>>,
    edges: Arc<Mutex<HashMap<T, Vec<Edge<T>>>>>,
) where
    T: Eq + Hash + FromStr + Display + Clone + Copy,
    for<'de> T: Deserialize<'de>,
    <T as FromStr>::Err: Debug,
{
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(sep)
        .from_path(file_path)
        .expect("Please check file path");
    rdr.seek(start_pos).expect("File split error!");

    if !is_edge_file {
        let mut iter = rdr.deserialize();
        for _ in 0..job_size {
            let vertex_info = iter.next().unwrap();
            if vertex_info.is_err() {
                println!("{}", vertex_info.err().unwrap());
                continue;
            }
            let v: Vertex<T> = vertex_info.unwrap();
            let mut data = vertices.lock().unwrap();
            data.insert(v.id, v);
        }
        return;
    }
    let mut iter = rdr.deserialize();
    for _ in 0..job_size {
        let origin_line = iter.next().unwrap();
        if origin_line.is_err() {
            println!("{}", origin_line.err().unwrap());
            continue;
        }
        let line_info: Edge<T> = origin_line.unwrap();
        let source = line_info.source;
        let target = line_info.target;
        let attrs = line_info.attrs;

        let mut data = edges.lock().unwrap();
        // Generating a reverse edge for undirected graph.
        if !is_directed {
            add_edge(&mut data, target, source, attrs.clone());
        }
        add_edge(&mut data, source, target, attrs.clone());

        // Generating a default node if there is no node file in configurations.
        if !has_node_file {
            let mut data = vertices.lock().unwrap();
            if !data.contains_key(&line_info.source) {
                data.insert(line_info.source, Vertex::new(line_info.source));
            }
            if !data.contains_key(&line_info.target) {
                data.insert(line_info.target, Vertex::new(line_info.target));
            }
        }
    }
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
    pub stack: VecDeque<Vertex<T>>,
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
        bfs.stack.push_front(graph.get_vertex(&start).unwrap());
        bfs
    }
}

impl<'a, T> Iterator for Bfs<'a, T>
where
    T: Copy + PartialEq + Eq + Hash,
{
    type Item = Vertex<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.stack.pop_front() {
            let neighbour_index_opt = self.bind_graph.neighbour_index.get(&node.id);
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
                    self.stack
                        .push_back(self.bind_graph.get_vertex(&cur.target).unwrap());
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
        self.stack
            .push_front(self.bind_graph.get_vertex(&self.start).unwrap());
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
    pub stack: Vec<Vertex<T>>,
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
        dfs.stack.push(graph.get_vertex(&start).unwrap());
        dfs
    }
}

impl<'a, T> Iterator for Dfs<'a, T>
where
    T: Copy + PartialEq + Eq + Hash,
{
    type Item = Vertex<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.stack.pop() {
            let neighbour_index_opt = self.bind_graph.neighbour_index.get(&node.id);
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
                    self.stack
                        .push(self.bind_graph.get_vertex(&cur.target).unwrap());
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
        self.stack
            .push(self.bind_graph.get_vertex(&self.start).unwrap());
    }
}

///TODO(Yu Chen):Other traversal algorithms here
fn other_traversal_algorithms() {}

/// Test case here
#[cfg(test)]
mod tests {
    use crate::errors::graph_build_error::BuildGraphError;
    use crate::{Bfs, Dfs, Graph, GraphReader, GraphType, Vertex};
    use std::cmp::Ordering;
    use std::collections::{HashMap, HashSet};
    use std::hash::Hash;

    #[test]
    fn file_reading_test() -> Result<(), BuildGraphError> {
        let g = GraphReader::default()
            .with_dir(GraphType::Directed)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges.csv"),
                Option::from(String::from("data_vertices.csv")),
                b',',
            )
            .with_thread(10)
            .build::<i32>()?;
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
        Ok(())
    }

    #[test]
    fn build_graph_in_memory_test() -> Result<(), BuildGraphError> {
        let edge_data = "source,target,a,b,c\
                         \n1,2,1,2,3\n1,3,1,2,3\n1,4,1,2,3\
                         \n2,1,1,2,3\n2,3,1,2,3\n2,4,1,2,3";
        let node_data = "id,a\n1,1\n2,1\n3,1\n4,1";
        let g = GraphReader::default()
            .with_dir(GraphType::Directed)
            .with_schema(String::from("..."))
            .from_reader(edge_data.to_string(), Some(node_data.to_string()), b',')
            .with_thread(10)
            .build::<i32>()?;
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
        Ok(())
    }

    #[test]
    fn dfs_iter_test_directed() -> Result<(), BuildGraphError> {
        let g = GraphReader::default()
            .with_dir(GraphType::Directed)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges.csv"),
                Option::from(String::from("data_vertices.csv")),
                b',',
            )
            .with_thread(10)
            .build()?;
        let dfs = Dfs::new(&g, 1);
        let mut res: Vec<Vertex<i32>> = dfs.collect();
        let mut parent = res.pop().unwrap();
        let mut stack = vec![];
        let mut map = HashMap::new();
        map.insert(parent.id, 1);
        stack.push(map);
        for v in res {
            let neighbours: Vec<i32> = g.get_neighbours(&parent.id).iter().map(|v| v.id).collect();
            if !neighbours.contains(&v.id) {}
            assert_eq!(neighbours.contains(&v.id), true);
            parent = v;
        }
        Ok(())
    }

    #[test]
    fn bfs_iter_test_directed() -> Result<(), BuildGraphError> {
        let g = GraphReader::default()
            .with_dir(GraphType::Directed)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges.csv"),
                Option::from(String::from("data_vertices.csv")),
                b',',
            )
            .with_thread(10)
            .build()?;
        let bfs = Bfs::new(&g, 1);
        let res: Vec<Vertex<i32>> = bfs.collect();
        Ok(())
    }

    #[test]
    fn dfs_iter_test_undirected() -> Result<(), BuildGraphError> {
        let g = GraphReader::default()
            .with_dir(GraphType::Undirected)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges_undirected.csv"),
                Option::from(String::from("data_vertices_undirected.csv")),
                b',',
            )
            .with_thread(10)
            .build()?;
        let dfs = Dfs::new(&g, 1);
        let res: Vec<Vertex<i32>> = dfs.collect();
        Ok(())
    }

    #[test]
    fn bfs_iter_test_undirected() -> Result<(), BuildGraphError> {
        let g = GraphReader::default()
            .with_dir(GraphType::Undirected)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges_undirected.csv"),
                Option::from(String::from("data_vertices_undirected.csv")),
                b',',
            )
            .with_thread(10)
            .build()?;
        let bfs = Bfs::new(&g, 1);
        let res: Vec<Vertex<i32>> = bfs.collect();
        let bfs_perm: Vec<i32> = res.iter().map(|v| v.id).collect();
        check_bfs_valid(bfs_perm, g);
        Ok(())
    }

    /// Check the given bfs permutation is valid for the given graph or not.
    fn check_bfs_valid<T>(mut bfs: Vec<T>, g: Graph<T>) -> bool
    where
        T: Clone + Eq + Hash,
    {
        if bfs.len() == 0 {
            return g.vertices.len() == 0;
        }
        // Initial
        let mut match_queue = HashSet::new();
        let mut parent_index = 1;
        let neighbours: Vec<T> = g
            .get_neighbours(&bfs[parent_index])
            .iter()
            .map(|v| v.id)
            .collect();
        match_queue.extend(neighbours.iter());

        for id in bfs {
            if match_queue.remove(&id) {
                continue;
            }
            if match_queue.is_empty() {
                parent_index += 1;
                fetch_next_level_and_assert_target(&mut match_queue, &bfs[parent_index], &g, &id);
                continue;
            }
            // Clear visited node
            for i in 0..parent_index {
                match_queue.remove(&bfs[i]);
            }
            assert_eq!(match_queue.len(), 0);
            parent_index += 1;
            fetch_next_level_and_assert_target(&mut match_queue, &bfs[parent_index], &g, &id);
        }
        g.vertices.len() == bfs.len()
    }

    fn fetch_next_level_and_assert_target<T>(
        match_queue: &mut HashSet<T>,
        root: T,
        g: &Graph<T>,
        target: T,
    ) where
        T: Clone + Eq + Hash,
    {
        let neighbours: Vec<T> = g.get_neighbours(&root).iter().map(|v| v.id).collect();
        match_queue.extend(neighbours.iter());
        assert!(match_queue.contains(&target), true);
        match_queue.remove(&target);
    }

    ///TODO(YuChen): Check the given dfs permutation is valid for the given graph or not.
    fn check_dfs_valid<T>(dfs: Vec<T>, g: Graph<T>) -> bool {
        false
    }
}
