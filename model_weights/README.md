# Weights
The weights shall be shared between Rust and Python either as a 

- JSON (.json) file that contains a single object, which itself consists of (_key_, _value_) pairs, where _key_ is the name of the parameter, and _value_ is a flattened array containing the weight values.
- NPZ file file, where each array is saved (non-flattened) under its parameter name. This can be achieved by generating a Python dict `d` consisting of (_key_, _value_) pairs, where _key_ is the name of the parameter, _value_ is the non-flattened array and then saving it via `npz.save(file, **d)`

HD5 is another good idea, but the corresponding Rust crate (hd5-rust) currently does not support WASM.

In the weights file, we are following Pytorch layout. Formats of the weights shall be as specified in the Pytorch documentation, as well as the naming convention (in general: <module_name>.<layer_name>.weight). 
