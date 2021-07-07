# Models

We are collecting some useful scripts for extraction of model weights here, as well as the specifcations of the 
variable names that the Rust model is using. The scripts are targeted to be used with tensorflow checkpoints (directly with the directory).

If you work with the scripts, please take care not to push the weights or other result files.

## Scripts
- `get_checkpoint_vars.py` can be used to extract a dictionary of variable names and shapes that are present in a model. First argument
is the checkpoint, second is the text file to save to.
- `get_weights_from_tf.py` can be used to extract the weights of a checkpoint to a JSON file. 
The argument `--valid-keys` can be used to extract only a part of the checkpoint weights (since in the checkpoint, 
the training information is also contained, this is not useful for the Rust model). To this argument, you can pass a file that contains a python list or a dictionary, and only variables that appear in the list (or in the dict as key) are written to the weights.json file.
