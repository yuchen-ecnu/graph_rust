use std::collections::{HashMap, HashSet, LinkedList};
use std::fs::File;
use std::io::Read;
use std::str::FromStr;
use std::string::ParseError;

#[derive(Clone)]
enum Attr {
    Int(i64),
    Float(f64),
    Text(String),
    //...other types
}

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

enum GraphType {
    UNDIRECTED,
    DIRECTED,
}

struct Edge {
    target: u64,
    attrs: HashMap<String, Attr>,
}

struct Vertex {
    id: u64,
    attrs: HashMap<String, Attr>,
}

impl Vertex {
    fn new(id: u64) -> Vertex {
        Vertex {
            id,
            attrs: HashMap::new(),
        }
    }
}

struct Graph {
    graph_type: GraphType,
    vertex_set: HashMap<u64, Vertex>,
    edge_set: HashMap<u64, Vec<Edge>>,
}

impl Graph {
    fn bfs(&self, source: u64) {
        let mut queue = LinkedList::new();
        let mut visited = HashSet::new();

        let source_v = self.vertex_set.get(&source);
        if let None = source_v {
            return;
        }
        let mut id = source_v.unwrap().id;
        queue.push_back(id);
        visited.insert(id);
        while !queue.is_empty() {
            let v = queue.pop_front().unwrap();
            print!("{}->", v);
            let neighbour_opt = self.edge_set.get(&v);
            if let None = neighbour_opt {
                continue;
            }
            let neighbour_vec = neighbour_opt.unwrap();
            for edge in neighbour_vec {
                if visited.contains(&edge.target) {
                    continue;
                }
                queue.push_back(edge.target);
                visited.insert(edge.target);
            }
        }
    }

    fn dfs_rec(&self, visited: &mut HashSet<u64>, source: u64) {
        //base case
        if visited.contains(&source) {
            return;
        }
        let neighbour_opt = self.edge_set.get(&source);
        if let None = neighbour_opt {
            print!("{}->", source);
            visited.insert(source);
            return;
        }
        visited.insert(source);
        let neighbour_vec = neighbour_opt.unwrap();
        for edge in neighbour_vec {
            self.dfs_rec(visited, edge.target);
        }
        print!("{}->", source);
    }

    fn dfs(&self, source: u64) {
        self.dfs_rec(&mut HashSet::new(), source);
    }

    fn load_graph(file_path: &String, graph_type: GraphType) -> Graph {
        //init graph data sets
        let mut vertex_set: HashMap<u64, Vertex> = HashMap::new();
        let mut edge_set: HashMap<u64, Vec<Edge>> = HashMap::new();

        //load file content to memory
        let mut content = String::new();
        let mut file = File::open(file_path).expect("Please check the file path");
        file.read_to_string(&mut content)
            .expect("something went wrong reading the file");

        //split file data into structure
        let directed = match graph_type {
            GraphType::DIRECTED => true,
            GraphType::UNDIRECTED => false,
        };
        for line in content.split("\n") {
            let edge_info: Vec<&str> = line.split(";").collect();
            let from = edge_info[0].parse().unwrap();
            let target = edge_info[1].parse().unwrap();

            //generate edge
            let mut attrs: HashMap<String, Attr> = HashMap::new();
            for i in 2..edge_info.len() {
                let attr: Vec<&str> = edge_info[i].split("=").collect();
                //invalidate line data: SKIP
                if attr.len() != 2 {
                    continue;
                }
                attrs.insert(attr[0].to_string(), attr[1].parse().unwrap());
            }

            //update edge set
            if !directed {
                update_edge_set(&mut edge_set, target, from, attrs.clone())
            }
            update_edge_set(&mut edge_set, from, target, attrs);

            //update vertex set
            if !vertex_set.contains_key(&from) {
                vertex_set.insert(from, Vertex::new(from));
            }
            if !vertex_set.contains_key(&target) {
                vertex_set.insert(target, Vertex::new(target));
            }
        }
        return Graph {
            graph_type,
            vertex_set,
            edge_set,
        };
    }
}

//assistance function for initial edge set
fn update_edge_set(
    edge_set: &mut HashMap<u64, Vec<Edge>>,
    from: u64,
    target: u64,
    attrs: HashMap<String, Attr>,
) {
    match edge_set.get(&from) {
        Some(v) => {
            let mut edge_vec = edge_set.remove(&from).unwrap();
            edge_vec.push(Edge { target, attrs });
            edge_set.insert(from, edge_vec);
        }
        None => {
            let mut vec = Vec::new();
            vec.push(Edge { target, attrs });
            edge_set.insert(from, vec);
        }
    }
}

fn main() {
    println!("Loading Graph...");
    let file_path = String::from("data.txt");
    let g = Graph::load_graph(&file_path, GraphType::DIRECTED);

    println!("Testing graph structure...");
    for (id, vertex) in &g.edge_set {
        print!("{}:", id);
        for edge in vertex {
            print!("{},", edge.target)
        }
        println!();
    }

    println!("Testing BFS...");
    g.bfs(2);
    println!();

    println!("Testing DFS...");
    g.dfs(1);
}
