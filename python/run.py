import os
import time
from compress import _compress, _decompress, encode_latents
from train import _train
import sys
##################
#  Hyperparems   #
##################
args = {
    "model": "mbt2018",
    "checkpoint_dir": "checkpoints",
    "results_dir": "results",
    "input_file": "images/0001.png",
    "lmbda": 0.01,
    "num_filters": 160,
    "model_file": "my_model",
}

train_args = {
    "patch_size": 256,
    "batch_size": 8,
    "n_steps": 10000,
}


# def run(mode, input_file, verbose=False):
#     flags = "--num_filters {num_filters} {verbose} --checkpoint_dir {checkpoint_dir}".format(
#         num_filters=args["num_filters"],
#         checkpoint_dir=args["checkpoint_dir"],
#         verbose="--verbose" if verbose else "",
#     )
#     results_flag = (
#         "--results_dir {}".format(args["results_dir"]) if mode == "compress" else ""
#     )
#     command = "python {model}.py {flags} {mode} {model_file} {input_file} {results_flag}".format(
#         model=args["model"],
#         flags=flags,
#         model_file=args["model_file"],
#         mode=mode,
#         input_file=input_file,
#         results_flag=results_flag,
#     )
#     os.system(command)


def decompress(input_file, activation):
    """
    :param input_file: name.tfci file
    """
    runname = args["model_file"]
    checkpoint_dir = args["checkpoint_dir"]
    num_filters = args["num_filters"]
    output_file = input_file + ".png"
    _decompress(runname, input_file, output_file, checkpoint_dir, num_filters, activation)


def compress(input_file, activation):
    """

    :param input_file: singe input image or np array of batch of images with shape (num_imgs, H, W, 3) and type(uint8)
    :param verbose:
    :return:
    """
    print("CHECK - COMPRESS")
    runname = args["model_file"]
    checkpoint_dir = args["checkpoint_dir"]
    results_dir = args["results_dir"]
    num_filters = args["num_filters"]
    output_file = input_file + ".tfci"

    _compress(
        runname, input_file, output_file, checkpoint_dir, results_dir, num_filters, activation
    )

    compressed_file = input_file + ".tfci"
    results_file = "rd-{model_file}-file={input_file}.npz".format(
        model_file=args["model_file"], input_file=input_file
    )
    results_file = os.path.join(args["results_dir"], results_file)

    return compressed_file, results_file


def train():
    _train(
        patch_size=train_args["patch_size"],
        batch_size=train_args["batch_size"],
        num_filters=args["num_filters"],
        lmbda=args["lmbda"],
        last_step=train_args["n_steps"],
    )


def main(input_file, activation):
    start_time = time.time()
    print(f">>> compressing {input_file} ...")
    compressed_file, results_file = compress(input_file, activation)
    intermediate_time = time.time()
    compress_time = intermediate_time - start_time
    print(f">>> compressing {input_file} done in {compress_time} seconds")
    compressed_file = "{}.tfci".format(input_file)
    print(f"<<< decompressing {compressed_file} ...")
    decompress(compressed_file, activation)
    stop_time = time.time()
    decompress_time = stop_time - intermediate_time
    print(f"<<< decompressing {compressed_file} done in {decompress_time} seconds")
    total_time = stop_time - start_time
    print(f"compressing and decompressing took {total_time} seconds")
    print(f"compressing took {(compress_time / total_time) * 100}% of the total time")
    print(
        f"decompressing took {(decompress_time / total_time) * 100}% of the total time"
    )


if __name__ == "__main__":
    if sys.argv[1] == 'True':
        activation = True
    else:
        activation = False

    ##################
    #  Compresssion  #
    ##################
    my_picture = "images/0001.png"
    main(my_picture, activation)

    ##################
    #    Latents     #
    ##################
    # latent_loc = 'results/latents-my_model-input=chess.png.npz'
    # encode_latents(latent_loc,
    #                args['num_filters'],
    #                args['checkpoint_dir'],
    #                args['model_file'],
    #                seperate=True)

    ##################
    #    Training    #
    ##################
    # train()
