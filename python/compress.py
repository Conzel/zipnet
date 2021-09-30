import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

# import constriction

from utils import write_png

from mbt2018_graph import _build_graph, SCALES_MIN, SCALES_MAX, SCALES_LEVELS
from nn_models import SynthesisTransform
from nn_models import MBT2018HyperSynthesisTransform as HyperSynthesisTransform


def load_input(input_file):
    if input_file.endswith(".npy"):
        # .npy file should contain N images of the same shapes, in the form of an array of shape [N, H, W, 3]
        X = np.load(input_file)
    else:
        # Load input image and add batch dimension.
        from PIL import Image

        x = np.asarray(Image.open(input_file).convert("RGB"))
        X = x[None, ...]
    # cast and normalize
    X = X.astype("float32")
    X /= 255.0
    return X


def get_eval_batch_size(num_pixels_per_image):
    # num pixels in the batch; corresponding to 10 1000x1000 images,
    # using 0.03GB memory (conversion from number of pixels to bytes: #bytes = #pixels * 24 / 8)
    eval_batch_num_pixels = 1e7
    return round(eval_batch_num_pixels / num_pixels_per_image)


def compress(args):
    input_file = args.input_file
    output_file = args.output_file
    checkpoint_dir = args.checkpoint_dir
    results_dir = args.results_dir
    num_filters = args.num_filters
    runname = args.runname

    _compress(
        runname, input_file, output_file, checkpoint_dir, results_dir, num_filters
    )


def decompress(args):
    input_file = args.input_file
    output_file = args.output_file
    checkpoint_dir = args.checkpoint_dir
    num_filters = args.num_filters
    runname = args.runname

    _decompress(runname, input_file, output_file, checkpoint_dir, num_filters)


def encode_latents(input_file, num_filters, checkpoint_dir, runname, seperate):
    # https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/ans-coder/bb_sga.py#L381-L386

    Z_DENSITY = 1 << 2

    # Load the latents
    latents_file = np.load(input_file)
    y = latents_file['y_tilde_cur']
    # z_mean = Z_DENSITY * latents_file['z_mean_cur']
    # z_std = Z_DENSITY * np.exp(0.5 * latents_file['z_logvar_cur'])
    batch_size = y.shape[0]
    width, height = latents_file['img_dimensions']

    # # Generate random side information and decode into z using q(z|y)
    # rng = np.random.RandomState(1234)
    # num_z = len(z_mean[0].ravel()) if seperate else len(z_mean.ravel())
    # # 32 bits per parameter should do; if not, just increase the size
    # side_information = rng.randint(0, 1 << 32, dtype=np.uint32, size=(10 + num_z,))
    # # Highest bit must be set (and therefore does not count towards the size of the side information
    # side_information[-1] |= 1 << 31
    # side_information_bits = 32 * len(side_information) - 1

    # # find maximum range for grid in z-space:
    # encoder_ranges_z = np.array([7, 15, 31, 63])
    # max_z_abs = (np.abs(z_mean) + 3.0 * z_std)
    # z_grid_range_index = min(len(encoder_ranges_z) - 1, np.sum(Z_DENSITY * encoder_ranges_z < max_z_abs))
    # z_grid_range = Z_DENSITY * encoder_ranges_z[z_grid_range_index]
    #
    # instantiate the model
    fake_X = np.zeros((batch_size, width, height, 3), dtype=np.float32)
    graph = _build_graph(fake_X, num_filters, training=False)
    #
    # z_mean = graph['mu']
    # _, z_width, z_height, _ = z_mean.shape.as_list()

    # Rasterize the hyperprior
    z_grid_range = 30
    z_grid = tf.tile(tf.reshape(tf.range(-z_grid_range, z_grid_range + 1), (-1, 1, 1, 1)),
                     (1, 1, 1, num_filters))
    z_grid = (1.0 / Z_DENSITY) * tf.cast(z_grid, tf.float32)
    entropy_bottleneck = graph['entropy_bottleneck']
    # z_grid_likelihood = tf.reshape(model.hyper_prior.pdf(z_grid), (2 * z_grid_range + 1, num_filters))
    z_tilde, z_likelihoods = entropy_bottleneck(z_grid, training=False)
    z_grid_likelihood = tf.reshape(z_likelihoods, (2 * z_grid_range + 1, num_filters))

    with tf.Session() as sess:
        # Load latest model checkpoint
        save_dir = os.path.join(checkpoint_dir, runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        tf.train.Saver().restore(sess, save_path=latest)

        # Replace tensorflow op with its corresponding value
        z_grid_likelihood = sess.run(z_grid_likelihood).astype(np.single)

    prefix = 'pub const MINNEN_HYPERPRIOR: [[f32; {}]; {}] ='.format(z_grid_range*2+1, num_filters)
    suffix = np.array2string(z_grid_likelihood,
                             threshold=sys.maxsize,
                             suppress_small=True,
                             separator=',')
    suffix = suffix.replace('\n', '')
    suffix = suffix.replace(' ', '')
    result = prefix + suffix + ';'

    support = 'pub const MINNEN_SUPPORT: (i32, i32) = (-{}, {});'.format(z_grid_range, z_grid_range+1)

    with open("file.rs", 'w') as file:
        with np.printoptions(suppress=True):
            print(support + '\n' + result, file=file)

    print("done")


def _compress(
        runname, input_file, output_file, checkpoint_dir, results_dir, num_filters, save_latents=True
):
    """Compresses an image, or a batch of images of the same shape in npy format."""
    tf.reset_default_graph()

    write_tfci_for_eval = True

    X = load_input(input_file)
    print(X.shape)

    num_images = int(X.shape[0])
    num_pixels = int(np.prod(X.shape[1:-1]))

    eval_batch_size = get_eval_batch_size(num_pixels)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(batch_size=eval_batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/data/Iterator
    # Importantly, each sess.run(op) call will consume a new batch, where op is any operation that depends on
    # x. Therefore if multiple ops need to be evaluated on the same batch of data, they have to be grouped like
    # sess.run([op1, op2, ...]).
    x_next = dataset.make_one_shot_iterator().get_next()

    x_ph = x = tf.placeholder(
        "float32", (None, *X.shape[1:])
    )  # keep a reference around for feed_dict
    graph = _build_graph(x, num_filters, training=False)

    print(graph["my_x_shape"])

    y_likelihoods, z_likelihoods, x_tilde = (
        graph["y_likelihoods"],
        graph["z_likelihoods"],
        graph["x_tilde"],
    )
    string, side_string = graph["string"], graph["side_string"]

    # Total number of bits divided by number of pixels.
    axes_except_batch = list(range(1, len(x.shape)))  # should be [1,2,3]
    y_bpp = tf.reduce_sum(-tf.log(y_likelihoods), axis=axes_except_batch) / (
            np.log(2) * num_pixels
    )
    z_bpp = tf.reduce_sum(-tf.log(z_likelihoods), axis=axes_except_batch) / (
            np.log(2) * num_pixels
    )
    eval_bpp = y_bpp + z_bpp  # shape (N,)

    # Bring both images back to 0..255 range.
    x *= 255
    x_tilde = tf.clip_by_value(x_tilde, 0, 1)
    x_tilde = tf.round(x_tilde * 255)

    mse = tf.reduce_mean(
        tf.squared_difference(x, x_tilde), axis=axes_except_batch
    )  # shape (N,)
    psnr = tf.image.psnr(x_tilde, x, 255)  # shape (N,)
    msssim = tf.image.ssim_multiscale(x_tilde, x, 255, filter_size=4)  # shape (N,)
    msssim_db = -10 * tf.log(1 - msssim) / np.log(10)  # shape (N,)
    x_shape = graph["x_shape"]
    y_shape = graph["y_shape"]
    z_shape = tf.shape(graph["z"])

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        save_dir = os.path.join(checkpoint_dir, runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        eval_fields = [
            "mse",
            "psnr",
            "msssim",
            "msssim_db",
            "est_bpp",
            "est_y_bpp",
            "est_z_bpp",
        ]
        eval_tensors = [mse,
                        psnr,
                        msssim,
                        msssim_db,
                        eval_bpp,
                        y_bpp,
                        z_bpp]
        all_results_arrs = {key: [] for key in eval_fields}  # append across all batches

        compression_tensors = [
            string,
            side_string,
            x_shape[1:-1],
            y_shape[1:-1],
            z_shape[1:-1],
        ]
        batch_actual_bpp = []
        batch_sizes = []

        batch_idx = 0
        while True:
            try:
                x_val = sess.run(x_next)
                x_feed_dict = {x_ph: x_val}

                # If requested, transform the quantized image back and measure performance.
                eval_arrs = sess.run(eval_tensors, feed_dict=x_feed_dict)
                for field, arr in zip(eval_fields, eval_arrs):
                    all_results_arrs[field] += arr.tolist()

                # Write a binary file with the shape information and the compressed string.
                packed = tfc.PackedTensors()
                compression_arrs = sess.run(compression_tensors, feed_dict=x_feed_dict)

                # test_string = sess.run(graph['string'], feed_dict=x_feed_dict)
                # test_side_string = sess.run(graph['side_string'], feed_dict=x_feed_dict)
                y = sess.run(graph['y'], feed_dict=x_feed_dict)
                z = sess.run(graph['z'], feed_dict=x_feed_dict)
                mu = sess.run(graph['mu'], feed_dict=x_feed_dict)
                sigma = sess.run(graph['sigma'], feed_dict=x_feed_dict)

                packed.pack(compression_tensors, compression_arrs)
                if write_tfci_for_eval:
                    print('writing tfci to {}'.format(output_file))
                    with open(output_file, "wb") as f:
                        f.write(packed.string)

                # The actual bits per pixel including overhead.
                batch_actual_bpp.append(
                    len(packed.string) * 8 / num_pixels
                )  # packed.string is the encoding for the entire batch
                batch_sizes.append(len(eval_arrs[0]))

                batch_idx += 1

            except tf.errors.OutOfRangeError:
                break

        for field in eval_fields:
            all_results_arrs[field] = np.asarray(all_results_arrs[field])

        all_results_arrs["batch_actual_bpp"] = np.asarray(batch_actual_bpp)
        all_results_arrs["batch_sizes"] = np.asarray(batch_sizes)

        avg_batch_actual_bpp = np.sum(batch_actual_bpp) / np.sum(batch_sizes)
        eval_fields.append("avg_batch_actual_bpp")
        all_results_arrs["avg_batch_actual_bpp"] = avg_batch_actual_bpp

        input_file = os.path.basename(input_file)
        results_dict = all_results_arrs
        np.savez(
            os.path.join(results_dir, "rd-%s-file=%s.npz" % (runname, input_file)),
            **results_dict
        )
        for field in eval_fields:
            arr = all_results_arrs[field]
            print("Avg {}: {:0.4f}".format(field, arr.mean()))

        if save_latents:
            prefix = 'latents'
            save_file = '{}-{}-input={}.npz'.format(prefix, runname, input_file)
            print('writing latents to {}'.format(os.path.join(results_dir, save_file)))
            np.savez(
                os.path.join(results_dir, save_file),
                y_tilde_cur=np.round(y).astype(np.int32),
                z_mean_cur=mu,
                z_logvar_cur=sigma,
                img_dimensions=np.array(X.shape[1:-1], dtype=np.int32)
            )


def _decompress(runname, input_file, output_file, checkpoint_dir, num_filters):
    """Decompresses an image."""
    # Adapted from https://github.com/tensorflow/compression/blob/master/examples/bmshj2018.py
    # Read the shape information and compressed string from the binary file.
    tf.reset_default_graph()
    string = tf.placeholder(tf.string, [1])
    side_string = tf.placeholder(tf.string, [1])
    x_shape = tf.placeholder(tf.int32, [2])
    y_shape = tf.placeholder(tf.int32, [2])
    z_shape = tf.placeholder(tf.int32, [2])
    with open(input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = [string, side_string, x_shape, y_shape, z_shape]
    arrays = packed.unpack(tensors)

    # Instantiate model. TODO: automate this with build_graph
    synthesis_transform = SynthesisTransform(num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(
        num_filters, num_output_filters=2 * num_filters
    )
    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)

    # Decompress and transform the image back.
    z_shape = tf.concat([z_shape, [num_filters]], axis=0)
    z_hat = entropy_bottleneck.decompress(side_string, z_shape, channels=num_filters)

    mu, sigma = tf.split(
        hyper_synthesis_transform(z_hat), num_or_size_splits=2, axis=-1
    )
    sigma = tf.exp(sigma)  # make positive
    training = False
    if (
            not training
    ):  # need to handle images with non-standard sizes during compression; mu/sigma must have the same shape as y
        mu = mu[:, : y_shape[0], : y_shape[1], :]
        sigma = sigma[:, : y_shape[0], : y_shape[1], :]
    scale_table = np.exp(
        np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)
    )
    conditional_bottleneck = tfc.GaussianConditional(
        sigma, scale_table, mean=mu, dtype=tf.float32
    )
    y_hat = conditional_bottleneck.decompress(string)
    x_hat = synthesis_transform(y_hat)

    # Remove batch dimension, and crop away any extraneous padding on the bottom
    # or right boundaries.
    x_hat = x_hat[0, : x_shape[0], : x_shape[1], :]

    # Write reconstructed image out as a PNG file.
    op = write_png(output_file, x_hat)

    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
        save_dir = os.path.join(checkpoint_dir, runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        sess.run(op, feed_dict=dict(zip(tensors, arrays)))
