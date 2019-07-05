use crate::{Edge, Vertex};
use std::collections::HashMap;
use std::error;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

#[derive(Debug, Clone)]
pub struct BuildGraphError {
    source: String,
}

impl fmt::Display for BuildGraphError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parse attr from string error: {}", self.source)
    }
}

impl error::Error for BuildGraphError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

impl<'a, T> From<PoisonError<MutexGuard<'a, HashMap<T, Vertex<T>>>>> for BuildGraphError {
    fn from(e: PoisonError<MutexGuard<'a, HashMap<T, Vertex<T>>>>) -> Self {
        BuildGraphError {
            source: e.description().to_string(),
        }
    }
}
impl<'a, T> From<PoisonError<MutexGuard<'a, HashMap<T, Vec<Edge<T>>>>>> for BuildGraphError {
    fn from(e: PoisonError<MutexGuard<'a, HashMap<T, Vec<Edge<T>>>>>) -> Self {
        BuildGraphError {
            source: e.description().to_string(),
        }
    }
}

impl From<csv::Error> for BuildGraphError {
    fn from(e: csv::Error) -> Self {
        BuildGraphError {
            source: e.description().to_string(),
        }
    }
}

impl<T> From<Arc<Mutex<HashMap<T, Vec<Edge<T>>>>>> for BuildGraphError {
    fn from(_e: Arc<Mutex<HashMap<T, Vec<Edge<T>>>>>) -> Self {
        BuildGraphError {
            source: "Consuming `mutex` of edges failed.".to_string(),
        }
    }
}

impl<T> From<Arc<Mutex<HashMap<T, Vertex<T>>>>> for BuildGraphError {
    fn from(_e: Arc<Mutex<HashMap<T, Vertex<T>>>>) -> Self {
        BuildGraphError {
            source: "Consuming `mutex` of vertices failed.".to_string(),
        }
    }
}

impl<T> From<PoisonError<HashMap<T, Vertex<T>>>> for BuildGraphError {
    fn from(e: PoisonError<HashMap<T, Vertex<T>>>) -> Self {
        BuildGraphError {
            source: e.description().to_string(),
        }
    }
}

impl<T> From<PoisonError<HashMap<T, Vec<Edge<T>>>>> for BuildGraphError {
    fn from(e: PoisonError<HashMap<T, Vec<Edge<T>>>>) -> Self {
        BuildGraphError {
            source: e.description().to_string(),
        }
    }
}
