use graph_rust::{Bfs, Dfs, Graph, GraphType};

#[test]
fn bfs_non_rec_test() {
    let g = Graph::default()
        .with_dir(GraphType::Directed)
        .with_schema(String::from("..."))
        .from_file(String::from("data_edges.csv"), None, ",")
        .build();

    g.bfs(2);
    println!()
}

#[test]
fn dfs_non_rec_test() {
    let g = Graph::default()
        .with_dir(GraphType::Directed)
        .with_schema(String::from("..."))
        .from_file(String::from("data_edges.csv"), None, ",")
        .build();

    g.dfs(1);
    println!()
}

#[test]
fn dfs_iter_test() {
    let g = Graph::default()
        .with_dir(GraphType::Directed)
        .with_schema(String::from("..."))
        .from_file(String::from("data_edges.csv"), None, ",")
        .build();
    let mut dfs = Dfs::new(&g, 1);
    while let Some(node) = dfs.next() {
        print!("{}->", node);
    }
    println!()
}

#[test]
fn bfs_iter_test() {
    let g = Graph::default()
        .with_dir(GraphType::Directed)
        .with_schema(String::from("..."))
        .from_file(String::from("data_edges.csv"), None, ",")
        .build();
    let mut bfs = Bfs::new(&g, 1);
    while let Some(node) = bfs.next() {
        print!("{}->", node);
    }
    println!()
}
