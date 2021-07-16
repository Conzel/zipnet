import os
import time
from compress import _compress, _decompress
from train import _train

args = {
    "model": "mbt2018",
    "checkpoint_dir": "checkpoints",
    "results_dir": "results",
    "input_file": "dog.jpg",
    "lmbda": 0.001,
    "num_filters": 192,
}
args["model_file"] = "{}-num_filters={}-lmbda={}".format(
    args["model"], args["num_filters"], args["lmbda"]
)

train_args = {
    "patch_size": 256,
    "batch_size": 8,
}


def run(mode, input_file, verbose=False):
    flags = "--num_filters {num_filters} {verbose} --checkpoint_dir {checkpoint_dir}".format(
        num_filters=args["num_filters"],
        checkpoint_dir=args["checkpoint_dir"],
        verbose="--verbose" if verbose else "",
    )
    results_flag = (
        "--results_dir {}".format(args["results_dir"]) if mode == "compress" else ""
    )
    command = "python {model}.py {flags} {mode} {model_file} {input_file} {results_flag}".format(
        model=args["model"],
        flags=flags,
        model_file=args["model_file"],
        mode=mode,
        input_file=input_file,
        results_flag=results_flag,
    )
    os.system(command)


def decompress(input_file, verbose=False):
    # run("decompress", input_file, verbose)
    runname = args["model_file"]
    checkpoint_dir = args["checkpoint_dir"]
    num_filters = args["num_filters"]
    output_file = input_file + ".png"
    _decompress(runname, input_file, output_file, checkpoint_dir, num_filters)


def compress(input_file, verbose=False):
    """

    :param input_file: singe input image or np array of batch of images with shape (num_imgs, H, W, 3) and type(uint8)
    :param verbose:
    :return:
    """
    # run("compress", input_file, verbose)

    runname = args["model_file"]
    checkpoint_dir = args["checkpoint_dir"]
    results_dir = args["results_dir"]
    num_filters = args["num_filters"]
    output_file = input_file + ".tfci"

    _compress(
        runname, input_file, output_file, checkpoint_dir, results_dir, num_filters
    )



    compressed_file = input_file + ".tfci"
    results_file = "rd-{model_file}-file={input_file}.npz".format(
        model_file=args["model_file"], input_file=input_file
    )
    results_file = os.path.join(args["results_dir"], results_file)

    return compressed_file, results_file


def main(input_file):
    start_time = time.time()
    print(f">>> compressing {input_file} ...")
    compressed_file, results_file = compress(input_file, verbose=True)
    intermediate_time = time.time()
    compress_time = intermediate_time - start_time
    print(f">>> compressing {input_file} done in {compress_time} seconds")
    compressed_file = "dog.jpg.tfci"
    print(f"<<< decompressing {compressed_file} ...")
    decompress(compressed_file, verbose=True)
    stop_time = time.time()
    decompress_time = stop_time - intermediate_time
    print(f"<<< decompressing {compressed_file} done in {decompress_time} seconds")
    total_time = stop_time - start_time
    print(f"compressing and decompressing took {total_time} seconds")
    print(f"compressing took {(compress_time / total_time) * 100}% of the total time")
    print(
        f"decompressing took {(decompress_time / total_time) * 100}% of the total time"
    )


def train():
    _train(patch_size=train_args['patch_size'],
           batch_size=train_args['batch_size'],
           num_filters=args['num_filters'],
           lmbda=args['lmbda'],
           last_step=10000)


if __name__ == "__main__":
    my_picture = "dog.jpg"
    main(my_picture)
