//! This module provides a way to load weights from NPZ files and compile
//! them directly into Rust modules. This provides an easy way to build and use models,
//! as the dependency on the correct weights is resolved at compile time.
use crate::WeightPrecision;
use ndarray::{Array, Array1, ArrayBase, Dimension, ShapeError, StrideShape};
use ndarray_npy::{NpzReader, ReadNpzError, ReadableElement};
use serde_json::{self, Map, Value};
use std::io::{Cursor, Read, Seek};
use std::{fs, path::Path};
use thiserror::Error;

type WeightResult<T> = Result<T, WeightError>;

/// Error type for the weight loader.
#[derive(Error, Debug)]
pub enum WeightError {
    #[error("No weights with name {0} found")]
    WeightKeyError(String),
    #[error("Weight file didn't have the correct format (required: JSON dict of pairs (key, flattened array of weights))")]
    WeightFormatError,
    #[error("Weight file not found. Filesystem reported error\n {0}.")]
    WeightFileNotFoundError(#[from] std::io::Error),
    #[error("Weight file not readable. Filesystem reported error\n {0}.")]
    WeightFileNpzError(#[from] ReadNpzError),
    #[error("Wrong shape for weight:\n {0}.")]
    WeightShapeError(#[from] ShapeError),
}

pub trait WeightLoader {
    /// Gets the weights out from the weight loader.
    ///
    /// We assume that the shapes in the weight loader have
    /// the given shape.
    fn get_weight<D, Sh, P: ReadableElement + Copy>(
        &mut self,
        param_name: &str,
        shape: Sh,
    ) -> WeightResult<Array<P, D>>
    where
        D: Dimension,
        Sh: Into<StrideShape<D>>;
}

/// Object to load weights that are in JSON format.
/// As an aside, the npz loader is a lot faster and more elegant,
/// as the weights can be loaded in the correct shape already, so they should be used
/// where possible.
pub struct JsonWeightLoader {
    content: Map<String, Value>,
}

impl JsonWeightLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> WeightResult<JsonWeightLoader> {
        let raw_file = fs::read_to_string(path)?;
        let parsed: Value =
            serde_json::from_str(&raw_file).map_err(|_| WeightError::WeightFormatError)?;
        let content = parsed.as_object().unwrap().clone();
        Ok(JsonWeightLoader { content })
    }
}

/// Object to load weights that are in NPZ format.
/// It can read from any readable, seekable object that contains npz data,
/// this might be files, temp files, byte arrays, ...
pub struct NpzWeightLoader<R>
where
    R: Seek + Read,
{
    handle: R,
}

impl NpzWeightLoader<std::fs::File> {
    /// Returns a weight loader from a given path
    pub fn from_path<P: AsRef<Path>>(path: P) -> WeightResult<NpzWeightLoader<std::fs::File>> {
        let handle = std::fs::File::open(path)?;
        Ok(NpzWeightLoader { handle })
    }
}

impl NpzWeightLoader<Cursor<&[u8]>> {
    /// Returns a weight loader from a byte array
    pub fn from_buffer(bytes_array: &[u8]) -> WeightResult<NpzWeightLoader<Cursor<&[u8]>>> {
        Ok(NpzWeightLoader {
            handle: Cursor::new(bytes_array),
        })
    }

    /// Returns a weight loader that has full access to all weights.
    /// The weights are compiled into the struct, so no file access is needed.
    /// This requires valid weights to be saved under <project_root>/model_weights/weights.npz
    /// at compile time.
    pub fn full_loader() -> NpzWeightLoader<Cursor<&'static [u8]>> {
        let bytes = include_bytes!("../../model_weights/weights.npz");
        NpzWeightLoader::from_buffer(bytes).unwrap()
    }
}

impl<R> WeightLoader for NpzWeightLoader<R>
where
    R: Seek + Read,
{
    /// Returns weights from the npz loader.
    ///
    /// First tries to load array with the shape given.
    /// If this is not possible, we assume that weights were saved flat
    /// and try to retrieve them flat and reshape.
    fn get_weight<D, Sh, P: Copy + ReadableElement>(
        &mut self,
        param_name: &str,
        shape: Sh,
    ) -> WeightResult<Array<P, D>>
    where
        D: Dimension,
        Sh: Into<StrideShape<D>>,
    {
        // The reader in the npy package has to be mut, so we recreate.
        // Else get_weight would have to be mutable (or we have to put it
        // into a RefCell). I dislike both solutions
        // We hope that this doesn't hurt perforrmance, we'll have to see.
        let mut reader = NpzReader::new(&mut self.handle)?;

        // checking for flat weights and reshaping
        let arr: Result<ArrayBase<_, D>, _> = reader.by_name(param_name);
        Ok(match arr {
            Ok(a) => {
                debug_assert_eq!(&a.raw_dim(), shape.into().raw_dim());
                a
            }
            Err(_) => {
                let arr_flat: Array1<P> = reader.by_name(param_name)?;
                let arr_reshaped =
                    Array::from_shape_vec(shape, arr_flat.iter().copied().collect())?;
                arr_reshaped
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use super::*;
    use ndarray::{array, Array2};
    use tempfile::tempdir;

    #[test]
    fn test_npz_weight_loader() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("temp-weights.npz");
        let file = File::create(&file_path).unwrap();
        let mut npz = ndarray_npy::NpzWriter::new(file);
        let a: Array2<f32> = array![[1., 2., 3.], [4., 5., 6.]];
        let b: Array1<f32> = array![7., 8., 9.];
        npz.add_array("a", &a).unwrap();
        npz.add_array("b", &b).unwrap();
        npz.finish().unwrap();

        let mut loader = NpzWeightLoader::from_path(file_path).unwrap();

        assert_eq!(loader.get_weight::<_, _, f32>("a", (2, 3)).unwrap(), a);
        assert_eq!(loader.get_weight::<_, _, f32>("b", 3).unwrap(), b);

        dir.close().unwrap();
    }
}
