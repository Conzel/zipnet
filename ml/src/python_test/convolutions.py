import numpy as np

def col2im(mul,h_prime,w_prime,C):
    """
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out

def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])
    # print(new_h, new_w, c, hh, ww)
    # print("img matrix:", x)
    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           print(patch.shape)
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad_num = conv_param['pad']
  stride = conv_param['stride']
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_prime = (H+2*pad_num-HH) // stride + 1
  W_prime = (W+2*pad_num-WW) // stride + 1
  out = np.zeros([N,F,H_prime,W_prime])
  #im2col
  for im_num in range(N):
      im = x[im_num,:,:,:]
      im_pad = np.pad(im,((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
      im_col = im2col(im_pad,HH,WW,stride)
      filter_col = np.reshape(w,(F,-1))
      mul = im_col.dot(filter_col.T) + b
      print(mul)
      out[im_num,:,:,:] = col2im(mul,H_prime,W_prime,1)
  cache = (x, w, b, conv_param)
  return out, cache


input = np.array([[[[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 9.0], [7.0, 8.0, 9.0, 9.0]], [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 9.0], [7.0, 8.0, 9.0, 9.0]], [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 9.0], [7.0, 8.0, 9.0, 9.0]]]]) # N, C, H, W
weights = np.array([[[[1.0, 2.0],[1.0, 2.0]],[[1.0, 2.0],[1.0, 2.0]], [[1.0, 2.0],[1.0, 2.0]]]]) # F, C, HH, WW
print(input.shape)
print(weights.shape)
bias = np.zeros((1, 1))# null
conv_param = {"pad":0,"stride":1}

output, _ = conv_forward_naive(input, weights, bias, conv_param)

print("output:", output)

test_im = np.random.rand(3, 4, 4)
cols = im2col(test_im, 2, 2, 2)
print(cols.shape, test_im.shape)
H_prime = (4+2*pad_num-HH) // stride + 1
W_prime = (4+2*-WW) // stride + 1
im = col2im(cols, 4, 4, 3)
assert(im==test_im)