# zipnet
ZipNet is a Work-In-Progress for the course project of [Data Compression With Deep Probabilistic Models](https://robamler.github.io/teaching/compress21/) offered by Robert Bamler at the University of Tübingen.

We want to provide a fully functional Neural Image-Encoder and Decoder on the Web. The goal is to achieve a superior compression rate over classical codes (JPEG, …) while retaining acceptable performance for a Web Application (compression in a few seconds). This results in a maximally portable application that could help Neural Compression Codecs achieve a higher adoption rate. For the implementation, we plan to leverage the new, performant WebASM standard along with a model architecture that allows for various performance optimizations such as quantization and parallelization.

If you are interested in what all of the crates and modules do, 
the folders either have READMEs or you can refer to the compiled cargo doc. 
You can compile the documentation for all crates you are interested in by 
switching to the corresponding folder (e.g. `cd coders`) and running `cargo doc --open`.

## Using the CLI
- Install the Rust toolchain (https://www.rust-lang.org/tools/install)
- Run the CLI via `cargo run -p zipnet -- <all the other commands you want to add>`. You can get an overview over the
possibilites by running `cargo run -p zipnet -- --help`. For subcommands, just call the subcommand, and then `--help`
to get information on them. Alternatively, you can build the program via `cargo build --release` 
(debug is likely too slow) and access the binary under `target/release/zipnet`.

In the following, we present some example usages.

### Compression
`./zipnet compress /path/to/image`
For compression, we accept .jpg, .png and .npy as inputs 
(.npy inputs are not preprocessed and are mainly intended for debugging). The compressed file is written out 
to /path/to/image.bin, overwriting the output file if it is already present.


### Decompression
`./zipnet decompress /path/to/image.bin /path/to/output_image.png`
Decompression takes a previous output from the compression step and outputs a recovered version of the
image under the given path. The output is always a .png.

## Building WASM locally
### Setup (Ubuntu)
Make sure you have installed the Rust-toolchain, wasm-pack and npm.
- _Rust_: Follow the instructions at https://www.rust-lang.org/tools/install
- _wasm-pack_: Run `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh `
- _npm_: If you don't have npm installed, run `sudo apt install nodejs npm`. If you have it installed, make you have the latest version by running `npm install npm@latest -g`

### Build
You can build each target specified in the Cargo.toml under workspace-members by running `cargo build -p <target-name>`. 

To setup the website locally, follow these steps from the root directory:
- `cd wasm`
- `wasm-pack build`
- `cd www`
- `npm install` 
- `npm run start`
- Open your browser and navigate to http://localhost:8080/

You should see the website running! As long as you have the server running, you can make changes by just re-running `wasm-pack build`.

You can get more detailed info on why we are doing everything like that at https://rustwasm.github.io/book/game-of-life/hello-world.html.

## Testing
As always, tests can be run via `cargo test`. There exist automated, randomized tests for the Rust modules that implement Python functionality to closely match their behaviour. They can be generated through the script by simply running the script in `scripts/generate_tests.py`.

For this, we recommend `tensorflow 2.5.0`, `numpy 1.19.5` and `Jinja2 3.0.0`.

### List of crates tested for WASM-compatibility
- ndarray v0.15.3 (no feature flags)

## Contributing for maintainers
### Setting up commit hooks
Please ensure you have git hooks installed when you push changes.
We use [rusty-hook](https://github.com/swellaby/rusty-hook) to manage git-hooks. To install, 
run `cargo install rusty-hook` and then `rusty-hook init` in this directory.
We run `cargo fmt` and `cargo check` on commits.

### General note
If you make changes and push them to the main branch, they are automatically reflected on the website via a Github Action. Thus please ensure beforehand that everything works locally as you want it. Please also make sure, that the GitHub Action correctly runs and fix errors that can occur.
