"""Mean-scale hyperprior model (no context model), as described in "Joint Autoregressive and Hierarchical Priors for
Learned Image Compression", NeurIPS2018, by Minnen, Ballé, and Toderici (https://arxiv.org/abs/1809.02736

Also see
Yibo Yang, Robert Bamler, Stephan Mandt:
"Improving Inference for Neural Image Compression", NeurIPS 2020
https://arxiv.org/pdf/2006.04240.pdf
where this is the "base" hyperprior model (M3 in Table 1 of paper).

We have a generative model of images:
z_tilde -> y_tilde -> x
where
p(z_tilde) = flexible_cdf_dist,
p(y_tilde | z_tilde) = N(y_tilde | hyper_synthesis_transform(z_tilde)) convolved with U(-0.5, 0.5),
p(x | y_tilde) = N(x | synthesis_transform(y_tilde)

and the following inference model:
x -> y_tilde  z_tilde
   \_________/^
where
q(y_tilde | x) = U(y-0.5, y+0.5), where y = analysis_transform(x)
q(z_tilde | x) = U(z-0.5, z+0.5), where z = hyper_analysis_transform(y)
"""

import os

import numpy as np
import tensorflow.compat.v1 as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

import tensorflow_compression as tfc

# Johnston = True  # use optimized num_filters
# if Johnston:
#     from nn_models import AnalysisTransform, HyperAnalysisTransform
#     from nn_models import SynthesisTransform_Johnston as SynthesisTransform
#     from nn_models import MBT2018HyperSynthesisTransform_Johnston as HyperSynthesisTransform
# else:  # use default num_filters
#     from nn_models import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform
#     from nn_models import MBT2018HyperSynthesisTransform as HyperSynthesisTransform
from utils import quantize_image
# from nn_models import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform
# from nn_models import MBT2018HyperSynthesisTransform as HyperSynthesisTransform

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def build_graph(args, x, training=True):
    num_filters = args.num_filters
    return _build_graph(x, num_filters, training)


def _build_graph(x, num_filters, training=True, activation=True):
    """
    Build the computational graph of the model. x should be a float tensor of shape [batch, H, W, 3].
    Given original image x, the model computes a lossy reconstruction x_tilde and various other quantities of interest.
    During training we sample from box-shaped posteriors; during compression this is approximated by rounding.
    """

    # Instantiate model.
    if activation == True:
        print("--"*50)
        print("Activation function is in use for compression")
        from nn_models import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform
        from nn_models import MBT2018HyperSynthesisTransform as HyperSynthesisTransform
        print("--"*50)
    else:
        print("--"*50)
        print("Activation function is not in use for compression")
        from nn_models_without_activation import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform
        from nn_models_without_activation import MBT2018HyperSynthesisTransform as HyperSynthesisTransform
        print("--"*50)
        
    analysis_transform = AnalysisTransform(num_filters)
    synthesis_transform = SynthesisTransform(num_filters)
    hyper_analysis_transform = HyperAnalysisTransform(num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(
        num_filters, num_output_filters=2 * num_filters
    )
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Build autoencoder and hyperprior.
    y, analysis_layers_output = analysis_transform(x)  # y = g_a(x)
    # import pdb;pdb.set_trace()
    assert y == analysis_layers_output[-1]
    # y = analysis_transform(x)  # y = g_a(x)
    z, hyp_analysis_layers_output = hyper_analysis_transform(y)  # z = h_a(y)
    assert z == hyp_analysis_layers_output[-1]
    # z = hyper_analysis_transform(y)  # z = h_a(y)

    # sample z_tilde from q(z_tilde|x) = q(z_tilde|h_a(g_a(x))), and compute the pdf of z_tilde under the flexible prior
    # p(z_tilde) ("z_likelihoods")
    z_tilde, z_likelihoods = entropy_bottleneck(z, training=training)
    # temp, hyp_synth_layers_output = hyper_synthesis_transform(z_tilde)
    temp, hyp_synth_layers_output = hyper_synthesis_transform(z_tilde)
    assert temp == hyp_synth_layers_output[-1]
    mu, sigma = tf.split(temp, num_or_size_splits=2, axis=-1)
    sigma = tf.exp(sigma)  # make positive

    if (
        not training
    ):  # need to handle images with non-standard sizes during compression; mu/sigma must have the same
        # shape as y
        y_shape = tf.shape(y)
        mu = mu[:, : y_shape[1], : y_shape[2], :]
        sigma = sigma[:, : y_shape[1], : y_shape[2], :]

    scale_table = np.exp(
        np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)
    )
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mu)

    # sample y_tilde from q(y_tilde|x) = U(y-0.5, y+0.5) = U(g_a(x)-0.5, g_a(x)+0.5), and then compute the pdf of
    # y_tilde under the conditional prior/entropy model p(y_tilde|z_tilde) = N(y_tilde|mu, sigma^2) conv U(-0.5,  0.5)
    print(y.shape)
    y_tilde, y_likelihoods = conditional_bottleneck(y, training=training)
    x_tilde, synth_layers_output = synthesis_transform(y_tilde)
    assert x_tilde == synth_layers_output[-1]
    # x_tilde = synthesis_transform(y_tilde)


    if not training:
        side_string = entropy_bottleneck.compress(z)
        string = conditional_bottleneck.compress(y)
        x_shape = tf.shape(x)
        x_tilde = x_tilde[
            :, : x_shape[1], : x_shape[2], :
        ]  # crop reconstruction to have the same shape as input

        my_x_shape = x.shape
        my_y_shape = y.shape
        my_z_shape = z.shape

    return locals()


def build_train_graph(args, x):
    num_filters = args.num_filters
    batch_size = args.batch_size
    patch_size = args.patch_size
    lmbda = args.lmbda

    return _build_train_graph(x, num_filters, batch_size, patch_size, lmbda)


def _build_train_graph(x, num_filters, batch_size, patch_size, lmbda):
    graph = _build_graph(x, num_filters, training=True, activation=False)
    y_likelihoods, z_likelihoods, x_tilde, = (
        graph["y_likelihoods"],
        graph["z_likelihoods"],
        graph["x_tilde"],
    )
    entropy_bottleneck = graph["entropy_bottleneck"]
    # Total number of bits divided by number of pixels.
    # - log p(\tilde y | \tilde z) - log p(\tilde z)
    num_pixels = batch_size * patch_size ** 2
    y_bpp = -tf.reduce_sum(tf.log(y_likelihoods)) / (np.log(2) * num_pixels)
    z_bpp = -tf.reduce_sum(tf.log(z_likelihoods)) / (np.log(2) * num_pixels)
    # train_bpp = (-tf.reduce_sum(tf.log(y_likelihoods)) -
    #              tf.reduce_sum(tf.log(z_likelihoods))) / (np.log(2) * num_pixels)
    train_bpp = y_bpp + z_bpp

    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    # Multiply by 255^2 to correct for rescaling.
    float_train_mse = train_mse
    psnr = -10 * (
        tf.log(float_train_mse) / np.log(10)
    )  # float MSE computed on float images
    train_mse *= 255 ** 2

    # The rate-distortion cost.
    train_loss = lmbda * train_mse + train_bpp

    # Minimize loss and auxiliary loss, and execute update op.
    step = tf.train.create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    model_name = os.path.splitext(os.path.basename(__file__))[0]
    original = quantize_image(x)
    reconstruction = quantize_image(x_tilde)
    return locals()
