import glob
import os
import sys

import tensorflow.compat.v1 as tf

from utils import read_png, read_npy_file_helper, get_runname

from mbt2018_graph import _build_train_graph


def train(args):

    patch_size = args.patchsize
    batch_size = args.batchsize
    last_step = args.last_step
    num_filters = args.num_filters
    lmbda = args.lmbda

    _train(patch_size, batch_size, num_filters, lmbda, last_step)


def _train(
    patch_size,
    batch_size,
    num_filters,
    lmbda,
    last_step,
    checkpoint_dir="checkpoints",
    logdir="tmp/tf_logs",
    verbose=False,
    save_summary_secs=60,
    save_checkpoint_secs=300,
    train_glob="images/*.png",
    preprocess_threads=16,
):
    """Trains the model."""

    if verbose:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)

    def input_data_pipeline(train_glob):

        # Create input data pipeline.
        with tf.device("/cpu:0"):
            train_files = glob.glob(train_glob)
            if not train_files:
                raise RuntimeError(
                    "No training images found with glob '{}'.".format(train_glob)
                )
            train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
            train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
            if (
                "npy" in train_glob
            ):  # reading numpy arrays directly instead of from images
                train_dataset = (
                    train_dataset.map(  # https://stackoverflow.com/a/49459838
                        lambda item: tuple(
                            tf.numpy_function(
                                read_npy_file_helper,
                                [item],
                                [
                                    tf.float32,
                                ],
                            )
                        ),
                        num_parallel_calls=preprocess_threads,
                    )
                )
            else:
                train_dataset = train_dataset.map(
                    read_png, num_parallel_calls=preprocess_threads
                )
            train_dataset = train_dataset.map(
                lambda x: tf.random_crop(x, (patch_size, patch_size, 3))
            )
            return train_dataset

    train_dataset = input_data_pipeline(train_glob)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(32)

    # num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()
    print("Shape of training data:", x.shape)
    res = _build_train_graph(x, num_filters, batch_size, patch_size, lmbda)
    train_loss = res["train_loss"]
    train_op = res["train_op"]
    model_name = res["model_name"]

    # boiler plate code for logging
    # runname = get_runname(
    #     vars(args),
    #     record_keys=("num_filters", "num_hfilters", "lmbda"),
    #     prefix=model_name,
    # )
    runname = "my_model"
    save_dir = os.path.join(checkpoint_dir, runname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    import json
    import datetime

    # with open(
    #     os.path.join(save_dir, "record.txt"), "a"
    # ) as f:  # keep more detailed record in text file
    #     f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    #     f.write(json.dumps(vars(args), indent=4, sort_keys=True) + "\n")
    #     f.write("\n")
    # with open(os.path.join(save_dir, "args.json"), "w") as f:  # will overwrite existing
    #     json.dump(vars(args), f, indent=4, sort_keys=True)

    # save a copy of the script that defined the model
    from shutil import copy

    copied_path = copy(model_name + ".py", save_dir)
    print("Saved a copy of %s.py to %s" % (model_name, copied_path))

    hooks = [
        tf.train.StopAtStepHook(last_step=last_step),
        tf.train.NanTensorHook(train_loss),
    ]

    if logdir != "":
        for key in res:
            if "bpp" in key or "loss" in key or key in ("mse", "psnr"):
                tf.summary.scalar(key, res[key])
            elif key in ("original", "reconstruction"):
                tf.summary.image(key, res[key], max_outputs=2)

        summary_op = tf.summary.merge_all()
        tf_log_dir = os.path.join(logdir, runname)
        summary_hook = tf.train.SummarySaverHook(
            save_secs=save_summary_secs, output_dir=tf_log_dir, summary_op=summary_op
        )
        hooks.append(summary_hook)

    with tf.train.MonitoredTrainingSession(
        hooks=hooks,
        checkpoint_dir=save_dir,
        save_checkpoint_secs=save_checkpoint_secs,
        save_summaries_secs=save_summary_secs,
    ) as sess:
        while not sess.should_stop():
            sess.run(train_op)


if __name__ == '__main__':
    _train(patch_size=256, batch_size=8, num_filters=192, lmbda=0.05, last_step=10000)
    