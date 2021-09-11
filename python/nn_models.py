import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import tensorflow as tf

class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        names = ['ana_layer_0', 'ana_layer_1', 'ana_layer_2', 'ana_layer_3']
        self._layers = [
            tf.layers.Conv2D(self.num_filters, (5, 5), name=names[0], strides=2,
                             padding="same", use_bias=False, activation=tfc.GDN(name="gnd_0")),
            tf.layers.Conv2D(self.num_filters, (5, 5), name=names[1], strides=2,
                             padding="same", use_bias=False, activation=tfc.GDN(name="gnd_1")),
            tf.layers.Conv2D(self.num_filters, (5, 5), name=names[2], strides=2,
                             padding="same", use_bias=False, activation=tfc.GDN(name="gnd_2")),
            tf.layers.Conv2D(self.num_filters, (5, 5), name=names[3], strides=2,
                             padding="same", use_bias=False, activation=None),
            # tfc.SignalConv2D(
            #     self.num_filters, (5, 5), name=names[0], corr=True, strides_down=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=tfc.GDN(name="gdn_0")),
            # tfc.SignalConv2D(
            #     self.num_filters, (5, 5), name=names[1], corr=True, strides_down=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=tfc.GDN(name="gdn_1")),
            # tfc.SignalConv2D(
            #     self.num_filters, (5, 5), name=names[2], corr=True, strides_down=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=tfc.GDN(name="gdn_2")),
            # tfc.SignalConv2D(
            #     self.num_filters, (5, 5), name=names[3], corr=True, strides_down=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=None),
        ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


# class SynthesisTransform(tf.keras.layers.Layer):
#     """The synthesis transform."""
#
#     def __init__(self, num_filters, *args, **kwargs):
#         self.num_filters = num_filters
#         super(SynthesisTransform, self).__init__(*args, **kwargs)
#
#     def build(self, input_shape):
#         self._layers = [
#             tfc.SignalConv2D(
#                 self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
#                 padding="same_zeros", use_bias=True,
#                 activation=tfc.GDN(name="igdn_0", inverse=True)),
#             tfc.SignalConv2D(
#                 self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
#                 padding="same_zeros", use_bias=True,
#                 activation=tfc.GDN(name="igdn_1", inverse=True)),
#             tfc.SignalConv2D(
#                 self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
#                 padding="same_zeros", use_bias=True,
#                 activation=tfc.GDN(name="igdn_2", inverse=True)),
#             tfc.SignalConv2D(
#                 3, (5, 5), name="layer_3", corr=False, strides_up=2,
#                 padding="same_zeros", use_bias=True,
#                 activation=None),
#         ]
#         super(SynthesisTransform, self).build(input_shape)
#
#     def call(self, tensor):
#         for layer in self._layers:
#             tensor = layer(tensor)
#         return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        names = ['layer_0', 'layer_1', 'layer_2', 'layer_3']
        filters = [79, 22, 43, 3]
        self._layers = [

            tf.layers.Conv2DTranspose(filters[0], (5, 5), name=names[0], strides=2,
                                      padding='same', use_bias=False, activation=tfc.GDN(name='igdn_0', inverse=True)),
            tf.layers.Conv2DTranspose(filters[1], (5, 5), name=names[1], strides=2,
                                      padding='same', use_bias=False, activation=tfc.GDN(name='igdn_1', inverse=True)),
            tf.layers.Conv2DTranspose(filters[2], (5, 5), name=names[2], strides=2,
                                      padding='same', use_bias=False, activation=tfc.GDN(name='igdn_2', inverse=True)),
            tf.layers.Conv2DTranspose(filters[3], (5, 5), name=names[3], strides=2,
                                      padding='same', use_bias=False, activation=tfc.GDN(name='igdn_3', inverse=True)),

            # tfc.SignalConv2D(
            #     filters[0], (5, 5), name=names[0], corr=False, strides_up=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=tfc.GDN(name="igdn_0", inverse=True)),
            # tfc.SignalConv2D(
            #     filters[1], (5, 5), name=names[1], corr=False, strides_up=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=tfc.GDN(name="igdn_1", inverse=True)),
            # tfc.SignalConv2D(
            #     filters[2], (5, 5), name=names[2], corr=False, strides_up=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=tfc.GDN(name="igdn_2", inverse=True)),
            # tfc.SignalConv2D(
            #     filters[3], (5, 5), name=names[3], corr=False, strides_up=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=None),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, num_output_filters=None, *args, **kwargs):
        self.num_filters = num_filters
        if num_output_filters is None:  # default to the same
            num_output_filters = num_filters
        self.num_output_filters = num_output_filters
        super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        names = ['layer_0', 'layer_1', 'layer_2', 'layer_3']
        self._layers = [
            tf.layers.Conv2D(self.num_filters, (3, 3), name=names[0], strides=1,
                             padding="same", use_bias=False, activation=tf.nn.relu),
            tf.layers.Conv2D(self.num_filters, (5, 5), name=names[1], strides=2,
                             padding="same", use_bias=False, activation=tf.nn.relu),
            tf.layers.Conv2D(self.num_filters, (5, 5), name=names[2], strides=2,
                             padding="same", use_bias=False, activation=None),

            # tfc.SignalConv2D(
            #     self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
            #     padding="same_zeros", use_bias=True,
            #     activation=tf.nn.relu),
            # tfc.SignalConv2D(
            #     self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            #     padding="same_zeros", use_bias=True,
            #     activation=tf.nn.relu),
            # tfc.SignalConv2D(
            #     self.num_output_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            #     padding="same_zeros", use_bias=False,
            #     activation=None)
        ]
        super(HyperAnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


# class HyperSynthesisTransform(tf.keras.layers.Layer):
#     """The synthesis transform for the entropy model parameters."""
#
#     def __init__(self, num_filters, num_output_filters=None, *args, **kwargs):
#         self.num_filters = num_filters
#         if num_output_filters is None:  # default to the same
#             num_output_filters = num_filters
#         self.num_output_filters = num_output_filters
#         super(HyperSynthesisTransform, self).__init__(*args, **kwargs)
#
#     def build(self, input_shape):
#         self._layers = [
#             tfc.SignalConv2D(
#                 self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
#                 padding="same_zeros", use_bias=True, kernel_parameterizer=None,
#                 activation=tf.nn.relu),
#             tfc.SignalConv2D(
#                 self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
#                 padding="same_zeros", use_bias=True, kernel_parameterizer=None,
#                 activation=tf.nn.relu),
#             tfc.SignalConv2D(
#                 self.num_output_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
#                 padding="same_zeros", use_bias=True, kernel_parameterizer=None,
#                 activation=None),
#         ]
#         super(HyperSynthesisTransform, self).build(input_shape)
#
#     def call(self, tensor):
#         for layer in self._layers:
#             tensor = layer(tensor)
#         return tensor


# # Architecture (mean-scale, no context model) based on Table 1 of https://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
# class MBT2018HyperSynthesisTransform(tf.keras.layers.Layer):
#     """The synthesis transform for the entropy model parameters."""
#
#     def __init__(self, num_filters, num_output_filters=None, *args, **kwargs):
#         self.num_filters = num_filters
#         if num_output_filters is None:  # default to the same
#             num_output_filters = num_filters
#         self.num_output_filters = num_output_filters
#         super().__init__(*args, **kwargs)
#
#     def build(self, input_shape):
#         self._layers = [
#             tfc.SignalConv2D(
#                 self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
#                 padding="same_zeros", use_bias=True, kernel_parameterizer=None,
#                 activation=tf.nn.relu),
#             tfc.SignalConv2D(
#                 int(self.num_filters * 1.5), (5, 5), name="layer_1", corr=False, strides_up=2,
#                 padding="same_zeros", use_bias=True, kernel_parameterizer=None,
#                 activation=tf.nn.relu),
#             tfc.SignalConv2D(
#                 self.num_output_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
#                 padding="same_zeros", use_bias=True, kernel_parameterizer=None,
#                 activation=None),
#         ]
#         super().build(input_shape)
#
#     def call(self, tensor):
#         for layer in self._layers:
#             tensor = layer(tensor)
#         return tensor
#
# if __name__ == '__main__':
#     num_filters = 192
#     encoder = AnalysisTransform(num_filters)
#     img = tf.random.uniform(shape=[1, 3, 4, 4])
#     y = encoder(img)



# Architecture (mean-scale, no context model) based on Table 1 of https://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
class MBT2018HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, num_output_filters=320, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        names = ['layer_0', 'layer_1', 'layer_2', 'layer_3']
        self._layers = [
            tf.layers.Conv2DTranspose(76, (3, 3), name=names[0], strides=2,
                             padding="same", use_bias=False, activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(107, (5, 5), name=names[1], strides=2,
                             padding="same", use_bias=False, activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(320, (5, 5), name=names[2], strides=1,
                             padding="same", use_bias=False, activation=None),

            # tfc.SignalConv2D(
            #     76, (5, 5), name="layer_0", corr=False, strides_up=2,
            #     padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            #     activation=tf.nn.relu),
            # tfc.SignalConv2D(
            #     107, (5, 5), name="layer_1", corr=False, strides_up=2,
            #     padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            #     activation=tf.nn.relu),
            # tfc.SignalConv2D(
            #     320, (3, 3), name="layer_2", corr=False, strides_up=1,
            #     padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            #     activation=None),
        ]
        super().build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor
