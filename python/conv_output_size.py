

def conv_output_size(input_shape, strides_down, strides_up=1):
    """
    using 
    https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/signal_conv.py#L945-L974
    """

    def f(s, strides_up, strides_down):
        s = (s - 1) * strides_up + 1
        s = (s - 1) // strides_down + 1
        return s

    W = f(input_shape[0], strides_up, strides_down)
    H = f(input_shape[1], strides_up, strides_down)

    return W, H


def mbt2018_output_sizes(img_shape):

    img_w, img_h = img_shape

    sizes = [(img_w, img_h)]

    strides_down = 2

    for i in range(4):
        w, h = conv_output_size(sizes[i], strides_down)
        sizes.append((w, h))

    for i, s in enumerate(sizes):
        str = "input" if i == 0 else "layer {}".format(i)
        print(str, s)


def main():
    mbt2018_output_sizes((533, 800))


if __name__ == "__main__":
    main()