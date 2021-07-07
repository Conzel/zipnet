use crate::WeightPrecision;
use ndarray::{Array, Array1, Dimension, Shape, ShapeError, StrideShape};
use serde_json::{self, Map, Value};
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
    #[error("Wrong shape for weight:\n {0}.")]
    WeightShapeError(#[from] ShapeError),
}

pub struct WeightLoader {
    content: Map<String, Value>,
}

impl WeightLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> WeightResult<WeightLoader> {
        let raw_file = fs::read_to_string(path)?;
        let parsed: Value =
            serde_json::from_str(&raw_file).map_err(|_| WeightError::WeightFormatError)?;
        let content = parsed.as_object().unwrap().clone();
        Ok(WeightLoader { content })
    }

    /// Returns weights with the given name from the weight loader. Weights are returned in a FLATTENED form
    /// (to facilitate working with JSON, as then all arrays have the same length.)
    pub fn get_weight<D, Sh>(
        &self,
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

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use super::*;
    use ndarray::array;
    use tempfile::tempdir;

    #[test]
    fn test_weight_loader() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("temp-weights.txt");
        let mut file = File::create(&file_path).unwrap();
        writeln!(
            file,
            // Rust escapes curly braces by doubling them
            "{{ \"arr1\": [0.0, 1e-3, 1.0], \"arr2\": [0.0, 1.0, 2.0, 3.0]}}"
        )
        .unwrap();

        let loader = WeightLoader::new(file_path).unwrap();

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
}
