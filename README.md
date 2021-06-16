# zipnet
ZipNet is a Work-In-Progress for the course project of [Data Compression With Deep Probabilistic Models](https://robamler.github.io/teaching/compress21/) offered by Robert Bamler at the University of Tübingen.

We want to provide a fully functional Neural Image-Encoder and Decoder on the Web. The goal is to achieve a superior compression rate over classical codes (JPEG, …) while retaining acceptable performance for a Web Application (compression in a few seconds). This results in a maximally portable application that could help Neural Compression Codecs achieve a higher adoption rate. For the implementation, we plan to leverage the new, performant WebASM standard along with a model architecture that allows for various performance optimizations such as quantization and parallelization.

## How to build locally
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

### List of crates tested for WASM-compatibility
- ndarray v0.15.3 (no feature flags)

## Contributing for maintainers
If you make changes and push them to the main branch, they are automatically reflected on the website via a Github Action. Thus please ensure beforehand that everything works locally as you want it. Please also make sure, that the GitHub Action correctly runs and fix errors that can occur.
