use std::error;
use std::error::Error;
use std::fmt;
use std::num::{ParseFloatError, ParseIntError};

#[derive(Debug, Clone)]
pub struct ParseAttrError {
    source: String,
}

impl fmt::Display for ParseAttrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parse attr from string error: {}", self.source)
    }
}

// This is important for other errors to wrap this one.
impl error::Error for ParseAttrError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

impl From<ParseIntError> for ParseAttrError {
    fn from(e: ParseIntError) -> Self {
        ParseAttrError {
            source: e.description().to_string(),
        }
    }
}

impl From<ParseFloatError> for ParseAttrError {
    fn from(e: ParseFloatError) -> Self {
        ParseAttrError {
            source: e.description().to_string(),
        }
    }
}
