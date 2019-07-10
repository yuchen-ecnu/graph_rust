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
use json::JsonValue;
use rocksdb::{Options, DB};
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

impl From<Attr> for JsonValue {
    fn from(attr: Attr) -> Self {
        match attr {
            Attr::Int(i) => json::from(i),
            Attr::Float(f) => json::from(f),
            Attr::Text(t) => json::from(t),
        }
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
    ///     5. Storing graph properties into rocksDB.
    ///
    /// # Feature Explanation
    /// 1. Storing layout of graph: in structure graph, we stores the edges and vertices in a
    ///     `Vec<Edge>` and `Vec<Vertex>`, so what will happen when we need to get a vertex neighbours
    ///     which id is 1? We need to search from the first position of Vec<Vertex> in graph, and
    ///     then we need to search from the first position to end of Vec<Edge> in graph. That's
    ///     cost a lot. So we generating two index for vertices and neighbours.
    /// 2. Vertex index used for indexing nodes by id and return the pos of vertex in vector `vertices`.
    ///     And neighbour_index used for indexing node's neighbours start and end index by source's
    ///     id, and return pos in vector `edges`.
    /// 3. In the process of storing graph properties into rocksDB, we try to open two ColumnFamily
    ///     which are named `meta_data` and `data`. In `meta_data`, we storing the basic graph
    ///     structure information,such as, the neighbours' id of vertices. And in `data`, we storing
    ///     the properties of vertices and edges.
    /// 4. In the process of storing properties, we aggregate the properties on vertices or edges
    ///     into a same value with `id` or `source-target` as key , which is formatted as a `json`.
    /// 5. For multi-graph, we aggregate the properties on the edges sharing the same source node
    ///     and target node into a same value with `source-target` as key.
    pub fn build<T>(&mut self) -> Result<Graph<T>, BuildGraphError>
    where
        T: Eq + Hash + FromStr + Display + Clone + Copy + Debug + Send,
        JsonValue: std::convert::From<T>,
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

        // Generating index && Storing graph and properties into rocksdb
        let vertices = Arc::try_unwrap(vertices_arc)?.into_inner()?;
        let mut edges = Arc::try_unwrap(edges_arc)?.into_inner()?;

        let path = "graph_storage";
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = DB::open_cf(&opts, path, &["meta_data", "data"]).unwrap();
        db.drop_cf("meta_data");
        db.drop_cf("data");
        db.create_cf("meta_data", &opts);
        db.create_cf("data", &opts);
        let cf_meta = db.cf_handle("meta_data").unwrap();
        let cf_data = db.cf_handle("data").unwrap();

        let mut cur_edge_index = 0;
        let mut cur_node_index = 0;
        for (id, v) in vertices {
            db.put_cf(
                cf_data,
                json::stringify(v.id),
                json::stringify(v.attrs.clone()),
            );
            let mut default_vec = vec![];
            let edge_vec = edges.get_mut(&id).unwrap_or_else(|| &mut default_vec);
            let neighbours: Vec<T> = edge_vec.iter().map(|e| e.target).collect();
            db.put_cf(cf_meta, json::stringify(id), json::stringify(neighbours));

            g.vertex_index.insert(v.id, cur_node_index);
            cur_node_index += 1;
            g.vertices.push(v);
            g.neighbour_index
                .insert(id, (cur_edge_index, cur_edge_index + edge_vec.len()));
            cur_edge_index = cur_edge_index + edge_vec.len();
            g.edges.append(edge_vec);
        }
        //storing data of graph
        for (source, edge_vec) in edges {
            // filter the parallel edge in the multi-graph and aggregate properties of them.
            let mut filter_map: HashMap<T, Vec<Attr>> = HashMap::new();
            for mut e in edge_vec {
                let mut default_vec = vec![];
                filter_map
                    .get_mut(&e.target)
                    .unwrap_or_else(|| &mut default_vec)
                    .append(&mut e.attrs);
            }
            for (target,attr_vec) in filter_map{
                let key = source.to_string() + "-" + String::as_str(&target.to_string());
                db.put_cf(cf_data, key, json::stringify(attr_vec));
            }
        }
        DB::destroy(&opts, path);

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
    let mut real_worker_count = (idx.len() - 1) / job_size;
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
    use crate::{Bfs, Dfs, Edge, Graph, GraphReader, GraphType, Vertex};
    use json::JsonValue;
    use rocksdb::{IteratorMode, Options, DB};
    use std::cmp::Ordering;
    use std::collections::HashMap;
    use std::rc::Rc;
    use std::cell::RefCell;
    use hdfs::{HdfsFsCache,HdfsFs};

    #[test]
    fn file_reading_and_rocksdb_storing_test() -> Result<(), BuildGraphError> {
        let g = generate_directed_graph(10)?;
        assert_eq!(check_graph_validation(g), true);

        // testing data in rocksdb valid
        let db = DB::open_cf(&Options::default(), "graph_storage", &["meta_data", "data"])?;
        let vertex_iter =
            db.iterator_cf(db.cf_handle("meta_data").unwrap(), IteratorMode::Start)?;
        let vertices: Vec<Vertex<i32>> = vertex_iter
            .map(|(id, _value)| {
                let id = parse_json_value_from_pointer(id).as_i32().unwrap();
                Vertex { id, attrs: vec![] }
            })
            .collect();
        let edge_iter = db.iterator_cf(db.cf_handle("meta_data").unwrap(), IteratorMode::Start)?;
        let edges: Vec<Edge<i32>> = edge_iter
            .flat_map(|(source, edge)| {
                let source = parse_json_value_from_pointer(source).as_i32().unwrap();
                let mut vec = vec![];

                let json = parse_json_value_from_pointer(edge);
                if let JsonValue::Array(neighbour_vec) = json {
                    for target in neighbour_vec {
                        vec.push(Edge {
                            source,
                            target: target.as_i32().unwrap(),
                            attrs: vec![],
                        });
                    }
                }
                vec
            })
            .collect();
        let graph_in_rocksdb = Graph {
            graph_type: GraphType::Directed,
            vertices,
            edges,
            vertex_key: vec![],
            edge_key: vec![],
            vertex_index: HashMap::new(),
            neighbour_index: HashMap::new(),
        };
        assert_eq!(check_graph_validation(graph_in_rocksdb), true);

        Ok(())
    }

    /// Testing the case of reading graph from string in memory.
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
        assert_eq!(check_graph_validation(g), true);
        Ok(())
    }

    #[test]
    fn dfs_iter_test_directed() -> Result<(), BuildGraphError> {
        let g = generate_directed_graph(10)?;
        let dfs = Dfs::new(&g, 1);
        let res: Vec<Vertex<i32>> = dfs.collect();
        let dfs_perm: Vec<i32> = res.iter().map(|v| v.id).collect();
        assert_eq!(check_dfs_permutation_valid(&dfs_perm), true);
        Ok(())
    }

    #[test]
    fn bfs_iter_test_directed() -> Result<(), BuildGraphError> {
        let g = generate_directed_graph(10)?;
        let bfs = Bfs::new(&g, 1);
        let res: Vec<Vertex<i32>> = bfs.collect();
        let bfs_perm: Vec<i32> = res.iter().map(|v| v.id).collect();
        assert_eq!(check_bfs_permutation_valid(&bfs_perm), true);
        Ok(())
    }

    #[test]
    fn dfs_iter_test_undirected() -> Result<(), BuildGraphError> {
        let g = generate_undirected_graph(10)?;
        let dfs = Dfs::new(&g, 1);
        let res: Vec<Vertex<i32>> = dfs.collect();
        let dfs_perm: Vec<i32> = res.iter().map(|v| v.id).collect();
        assert_eq!(check_dfs_permutation_valid(&dfs_perm), true);
        Ok(())
    }

    #[test]
    fn bfs_iter_test_undirected() -> Result<(), BuildGraphError> {
        let g = generate_undirected_graph(10)?;
        let bfs = Bfs::new(&g, 1);
        let res: Vec<Vertex<i32>> = bfs.collect();
        let bfs_perm: Vec<i32> = res.iter().map(|v| v.id).collect();
        assert_eq!(check_bfs_permutation_valid(&bfs_perm), true);
        Ok(())
    }

    #[test]
    fn hdfs_reading_test(){
        //TODO(Yu Chen):File system load failed
        let cache = Rc::new(RefCell::new(HdfsFsCache::new()));  
        let fs: HdfsFs = cache.borrow_mut().get("hdfs://localhost:9000/").ok().unwrap();
        match fs.mkdir("/data") {
            Ok(_) => { println!("/data has been created") },
            Err(_)  => { panic!("/data creation has failed") }
        }; 
    }

    fn parse_json_value_from_pointer(source: Box<[u8]>) -> JsonValue {
        json::parse(
            String::from_utf8((*source.to_vec()).to_owned())
                .unwrap()
                .as_str(),
        )
        .unwrap()
    }

    /// Generate a undirected `Graph` instance from test files
    fn generate_undirected_graph(thread_num: u64) -> Result<Graph<i32>, BuildGraphError> {
        let g = GraphReader::default()
            .with_dir(GraphType::Undirected)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges_undirected.csv"),
                Option::from(String::from("data_vertices_undirected.csv")),
                b',',
            )
            .with_thread(thread_num)
            .build()?;
        Ok(g)
    }

    /// Generate a directed `Graph` instance from test files
    fn generate_directed_graph(thread_num: u64) -> Result<Graph<i32>, BuildGraphError> {
        let g = GraphReader::default()
            .with_dir(GraphType::Directed)
            .with_schema(String::from("..."))
            .from_file(
                String::from("data_edges.csv"),
                Option::from(String::from("data_vertices.csv")),
                b',',
            )
            .with_thread(thread_num)
            .build::<i32>()?;
        Ok(g)
    }

    /// Check the given bfs permutation is valid for the test graph or not.
    fn check_bfs_permutation_valid(bfs_perm: &Vec<i32>) -> bool {
        return bfs_perm.eq(&vec![1, 2, 3, 4])
            || bfs_perm.eq(&vec![1, 2, 4, 3])
            || bfs_perm.eq(&vec![1, 3, 2, 4])
            || bfs_perm.eq(&vec![1, 3, 4, 2])
            || bfs_perm.eq(&vec![1, 4, 2, 3])
            || bfs_perm.eq(&vec![1, 4, 3, 2]);
    }

    /// Check the given dfs permutation is valid for the test graph or not.
    fn check_dfs_permutation_valid(dfs_perm: &Vec<i32>) -> bool {
        return dfs_perm.eq(&vec![1, 2, 3, 4])
            || dfs_perm.eq(&vec![1, 2, 4, 3])
            || dfs_perm.eq(&vec![1, 3, 2, 4])
            || dfs_perm.eq(&vec![1, 3, 4, 2])
            || dfs_perm.eq(&vec![1, 4, 2, 3])
            || dfs_perm.eq(&vec![1, 4, 3, 2]);
    }

    /// Check the given graph is valid for the test graph data file.
    fn check_graph_validation(g: Graph<i32>) -> bool {
        let mut vec = vec![];
        for v in g.vertices {
            vec.push(v.id);
        }
        vec.sort();

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

        return vec.eq(&vec![1, 2, 3, 4])
            && edges.eq(&vec![(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4)]);
    }
}
