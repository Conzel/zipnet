use crate::WeightPrecision;
use ndarray::{Array, ArrayBase, Dimension, ShapeError, StrideShape};
use ndarray_npy::{NpzReader, ReadNpzError};
use serde_json::{self, Map, Value};
use std::io::{Cursor, Read, Seek};
use std::{fs, path::Path};
use thiserror::Error;

type WeightResult<T> = Result<T, WeightError>;

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
    fn get_weight<D, Sh>(
        &mut self,
        param_name: &str,
        shape: Sh,
    ) -> WeightResult<Array<WeightPrecision, D>>
    where
        D: Dimension,
        Sh: Into<StrideShape<D>>;
}

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

impl WeightLoader for JsonWeightLoader {
    /// Returns weights with the given name from the weight loader. Weights are returned in a FLATTENED form
    /// (to facilitate working with JSON, as then all arrays have the same length.)
    fn get_weight<D, Sh>(
        &mut self,
        param_name: &str,
        shape: Sh,
    ) -> WeightResult<Array<WeightPrecision, D>>
    where
        D: Dimension,
        Sh: Into<StrideShape<D>>,
    {
        let raw_arr = self
            .content
            .get(param_name)
            .ok_or(WeightError::WeightKeyError(param_name.to_string()))?;

        let raw_value_vector = match raw_arr {
            Value::Array(v) => v,
            _ => return Err(WeightError::WeightFormatError),
        };

        // We might want to disable this check on release?
        let weight_vector: Result<Vec<_>, _> = raw_value_vector
            .iter()
            .map(|j| {
                j.as_f64()
                    .map(|v| v as f32)
                    .ok_or(WeightError::WeightFormatError)
            })
            .collect();

        let weights = Array::from_shape_vec(shape, weight_vector?)?;

        Ok(weights)
    }
}

pub struct NpzWeightLoader<R>
where
    R: Seek + Read,
{
    handle: R,
}

impl NpzWeightLoader<std::fs::File> {
    pub fn from_path<P: AsRef<Path>>(path: P) -> WeightResult<NpzWeightLoader<std::fs::File>> {
        let handle = std::fs::File::open(path)?;
        Ok(NpzWeightLoader { handle })
    }
}

impl NpzWeightLoader<Cursor<&[u8]>> {
    pub fn from_buffer(bytes_array: &[u8]) -> WeightResult<NpzWeightLoader<Cursor<&[u8]>>> {
        Ok(NpzWeightLoader {
            handle: Cursor::new(bytes_array),
        })
    }

    /// Returns a weight loader that has full access to all weight.
    /// The weights are compiled into the struct, so no file access is needed.
    pub fn full_loader() -> NpzWeightLoader<Cursor<&'static [u8]>> {
        todo!()
    }
}

impl<R> WeightLoader for NpzWeightLoader<R>
where
    R: Seek + Read,
{
    fn get_weight<D, Sh>(
        &mut self,
        param_name: &str,
        _shape: Sh,
    ) -> WeightResult<Array<WeightPrecision, D>>
    where
        D: Dimension,
        Sh: Into<StrideShape<D>>,
    {
        // The reader in the npy package has to be mut, so we recreate.
        // Else get_weight would have to be mutable (or we have to put it
        // into a RefCell). I dislike both solutions
        // We hope that this doesn't hurt perforrmance, we'll have to see.
        let mut reader = NpzReader::new(&mut self.handle)?;

        let arr: ArrayBase<_, D> = reader.by_name(param_name)?;

        debug_assert_eq!(&arr.raw_dim(), _shape.into().raw_dim());
        Ok(arr)
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
    fn test_json_weight_loader() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("temp-weights.txt");
        let mut file = File::create(&file_path).unwrap();
        writeln!(
            file,
            // Rust escapes curly braces by doubling them
            "{{ \"arr1\": [0.0, 1e-3, 1.0], \"arr2\": [0.0, 1.0, 2.0, 3.0]}}"
        )
        .unwrap();

        let mut loader = JsonWeightLoader::new(file_path).unwrap();

        assert_eq!(
            loader.get_weight("arr1", 3).unwrap(),
            array![0.0, 1e-3, 1.0]
        );
        assert_eq!(
            loader.get_weight("arr2", (2, 2)).unwrap(),
            array![[0.0, 1.0], [2.0, 3.0]]
        );

        drop(file);
        dir.close().unwrap();
    }

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

        assert_eq!(loader.get_weight("a", (2, 3)).unwrap(), a);
        assert_eq!(loader.get_weight("b", 3).unwrap(), b);

        dir.close().unwrap();
    }
}
