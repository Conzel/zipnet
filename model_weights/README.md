# Models

We are collecting some useful scripts for extraction of model weights here, as well as the specifcations of the 
variable names that the Rust model is using. The scripts are targeted to be used with tensorflow checkpoints (directly with the directory).

If you work with the scripts, please take care not to push the weights or other result files.

## Weights
The weights shall be shared between Rust and Python either as a 

- JSON (.json) file that contains a single object, which itself consists of (_key_, _value_) pairs, where _key_ is the name of the parameter, and _value_ is a flattened array containing the weight values.
- NPZ file file, where each array is saved (non-flattened) under its parameter name. This can be achieved by generating a Python dict `d` consisting of (_key_, _value_) pairs, where _key_ is the name of the parameter, _value_ is the non-flattened array and then saving it via `npz.save(file, **d)`

HD5 is another good idea, but the corresponding Rust crate (hd5-rust) currently does not support WASM.
## Scripts
- `get_checkpoint_vars.py` can be used to extract a dictionary of variable names and shapes that are present in a model. First argument
is the checkpoint, second is the text file to save to.
Example usage: `python3 get_checkpoint_vars.py checkpoints/my_model/ weight_keys.txt`
- `get_weights_from_tf.py` can be used to extract the weights of a checkpoint to a weights file. The file extension determines the type (json, npz, hd5). 
The argument `--valid-keys` can be used to extract only a part of the checkpoint weights (since in the checkpoint, 
the training information is also contained, this is not useful for the Rust model). To this argument, you can pass a file that contains a python list or a dictionary, and only variables that appear in the list (or in the dict as key) are written to the weights.json file.
This works well with the `get_checkpoint_vars.py` script.
Example usage: `python3 get_weights_from_tf.py --valid-keys rust_model_keys.txt checkpoints/my_model/ weights.npz`

A usual workflow might look like this:
1. Obtain a checkpoints folder from tensorflow
2. Use `get_checkpoint_vars.py` to write out a `keys.txt` file
3. Remove the unwanted keys from `keys.txt` (this is manual labor, but shouldn't take too long)
4. Use `get_vars_fromtf.py` to get the weights into npz format