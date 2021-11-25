# python

This folder contains various experiments we did with Python. It already contains a working Python implementation of the hierarchical coding procedure.

This folder has been forked from https://github.com/mandt-lab/improving-inference-for-neural-image-compression and slightly altered to suit our needs.

NOTE: The code is a bit messy because the original code was written such that it would take "args" as input in a lot of functions, but I didn't want to completely break that functionality. 

# Setup
In any case, we recommend using a virtual environment, as the package dependencies here are a bit delicate. You can install the packages either with `conda` (recommended) or with `pip`. Additionally, to run the code, you might have to enter `EXPORT CUDA_VISIBLE_DEVICES="-1"` in the terminal (if TF doesn't recognize your GPU drivers).
## conda
Follow the installation instructions for conda. Switch into this folder and run `conda create --file environment.yml`. You can then activate the environment using `conda activate zipnet`.

## pip
This method works with multiple types of virtual environments (either `venv` oder `conda`). Simply activate your venv and run `pip install -r requirements.txt`.

# Save each layer's output
First activate the virtual environment and make sure that weights are downloaded and are available in `checkpoints` folder
`mkdir layers`

To run without activation function, run
`python run.py False`

to run with activation function, run
`python run.py True`

The script for now uses the image in `images` folder (more changes are soon to come), and saves each layer's output in the `layers` folder individually as `.npz` files. 

# Useage 

@Johan: 
In the main file (run.py) there are 3 different functionalities supported:

- Train the model, and save the checkpoints;
- Compress and decompress an image;
- Encode the latents into a grid;

The training of the model is straightforward. There are some hyperparameters that can be changed without changing the model itself. From "args" this is "lmbda", and from "train_args" these are "patch_size", "batch_size", and "n_steps". In additions there are a number of default parameters, i.e. the checkpoints will be saved in the "checkpoints" folder, and the training data should be in "images/*.png". During training some logs are also saved. To save weights at different epochs, just "save_checkpoint_secs" to the required epoch.

Compressing and decompressing an image does not have any parameters, except the name of the input image. By default, compressing an image also saves the latents, which are important if you want to encode the latents into a grid. Also, the intermediate output for each layer is saved to disk for compression and decompression, for debugging purposes. There is also code for printing model weights, or other important variables in the compress method. 

Encoding the latents into a grid requires a saved latents file obtained by compression an image. Then by calling this function with the correct input saves a rasterized grid of the latents as a Rust file.
