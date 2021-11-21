import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import tensorflow as tf


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        names = [
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
        ]
        self._layers = [
            tf.layers.Conv2D(
                self.num_filters,
                (5, 5),
                name=names[0],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tfc.GDN(name="gnd_0"),
            ),
            tf.layers.Conv2D(
                self.num_filters,
                (5, 5),
                name=names[1],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tfc.GDN(name="gnd_1"),
            ),
            tf.layers.Conv2D(
                self.num_filters,
                (5, 5),
                name=names[2],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tfc.GDN(name="gnd_2"),
            ),
            tf.layers.Conv2D(
                self.num_filters,
                (5, 5),
                name=names[3],
                strides=2,
                padding="same",
                use_bias=False,
                activation=None,
            ),
        ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        layers_output = []
        for layer in self._layers:
            tensor = layer(tensor)
            layers_output.append(tensor)
        return tensor, layers_output


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        names = [
            "decoder_layer_0",
            "decoder_layer_1",
            "decoder_layer_2",
            "decoder_layer_3",
        ]
        filters = [79, 22, 43, 3]
        self._layers = [
            tf.layers.Conv2DTranspose(
                filters[0],
                (5, 5),
                name=names[0],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tfc.GDN(name="igdn_0", inverse=True),
            ),
            tf.layers.Conv2DTranspose(
                filters[1],
                (5, 5),
                name=names[1],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tfc.GDN(name="igdn_1", inverse=True),
            ),
            tf.layers.Conv2DTranspose(
                filters[2],
                (5, 5),
                name=names[2],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            tf.layers.Conv2DTranspose(
                filters[3],
                (5, 5),
                name=names[3],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tfc.GDN(name="igdn_3", inverse=True),
            ),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        layers_output = []
        for layer in self._layers:
            tensor = layer(tensor)
            layers_output.append(tensor)
        print("check")
        return tensor, layers_output


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, num_output_filters=None, *args, **kwargs):
        self.num_filters = num_filters
        if num_output_filters is None:  # default to the same
            num_output_filters = num_filters
        self.num_output_filters = num_output_filters
        super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        names = [
            "hyperencoder_layer_0",
            "hyperencoder_layer_1",
            "hyperencoder_layer_2",
            "hyperencoder_layer_3",
        ]
        self._layers = [
            tf.layers.Conv2D(
                self.num_filters,
                (3, 3),
                name=names[0],
                strides=1,
                padding="same",
                use_bias=False,
                activation=tf.nn.relu,
            ),
            tf.layers.Conv2D(
                self.num_filters,
                (5, 5),
                name=names[1],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tf.nn.relu,
            ),
            tf.layers.Conv2D(
                self.num_filters,
                (5, 5),
                name=names[2],
                strides=2,
                padding="same",
                use_bias=False,
                activation=None,
            ),
        ]
        super(HyperAnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        layers_output = []
        for layer in self._layers:
            tensor = layer(tensor)
            layers_output.append(tensor)
        return tensor, layers_output


# Architecture (mean-scale, no context model) based on Table 1 of https://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
class MBT2018HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, num_output_filters=320, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        names = [
            "hyperdecoder_layer_0",
            "hyperdecoder_layer_1",
            "hyperdecoder_layer_2",
            "hyperdecoder_layer_3",
        ]
        self._layers = [
            tf.layers.Conv2DTranspose(
                76,
                (5, 5),
                name=names[0],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tf.nn.relu,
            ),
            tf.layers.Conv2DTranspose(
                107,
                (5, 5),
                name=names[1],
                strides=2,
                padding="same",
                use_bias=False,
                activation=tf.nn.relu,
            ),
            tf.layers.Conv2DTranspose(
                320,
                (3, 3),
                name=names[2],
                strides=1,
                padding="same",
                use_bias=False,
                activation=None,
            ),
        ]
        super().build(input_shape)

    def call(self, tensor):
        layers_output = []
        for layer in self._layers:
            tensor = layer(tensor)
            layers_output.append(tensor)
        return tensor, layers_output
