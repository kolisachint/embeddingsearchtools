use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors surfaced by the embedding search engine.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// A query/insert vector did not match the index dimension.
    #[error("dimension mismatch: index is {expected}-d, got {got}-d")]
    DimensionMismatch { expected: usize, got: usize },

    /// Referenced an id that is not present in the index.
    #[error("unknown id: {0}")]
    UnknownId(String),

    /// Tried to add an id that already exists.
    #[error("duplicate id: {0}")]
    DuplicateId(String),

    /// On-disk data could not be interpreted.
    #[error("corrupt store: {0}")]
    Corrupt(String),

    /// The embedding backend failed.
    #[error("embedding backend error: {0}")]
    Embed(String),

    /// A configuration value was invalid.
    #[error("invalid config: {0}")]
    Config(String),
}

impl Error {
    pub(crate) fn corrupt(msg: impl fmt::Display) -> Self {
        Error::Corrupt(msg.to_string())
    }

    #[cfg(feature = "onnx")]
    pub(crate) fn embed(msg: impl fmt::Display) -> Self {
        Error::Embed(msg.to_string())
    }
}
