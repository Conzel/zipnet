const CopyWebpackPlugin = require("copy-webpack-plugin");
const webpack = require("webpack");
const path = require('path');

module.exports = {
  entry: "./bootstrap.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bootstrap.js",
  },
  mode: "development",
  plugins: [
    new CopyWebpackPlugin(['index.html'])
  ],
  // refer to https://github.com/webpack/webpack/issues/7352
  // Hack to make it work with emscripten-wasm
  node: { 
    "fs": "empty" // ← !!
  },
  module: {
    rules: [
      /* ... */
      {
        test: /fibonacci\.wasm$/,
        type: "javascript/auto", // ← !!
        loader: "file-loader",
        options: {
          publicPath: "dist/"
        }
      }
    ]
  },
};
