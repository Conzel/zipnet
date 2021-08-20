import numpy as np
from math import ceil

def get_same_padding(input_h, input_w, stride, filter_h, filter_w):
	if input_h % stride == 0:
		pad_along_height = max((filter_h - stride), 0)
	else:
		pad_along_height = max(filter_h - (input_h % stride), 0)
	if input_w % stride == 0:
		pad_along_width = max((filter_w - stride), 0)
	else:
		pad_along_width = max(filter_w - (input_w % stride), 0)
	
	pad_top = pad_along_height // 2
	pad_bottom = pad_along_height - pad_top
	pad_left = pad_along_width // 2
	pad_right = pad_along_width - pad_left

	return pad_along_height, pad_along_width, pad_top, pad_bottom, pad_left, pad_right            
        
      
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
        #    print(patch.shape)
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
	N,CC,H,W = x.shape
	F,C,HH,WW = w.shape
	pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right = get_same_padding(H, W, stride, HH, WW)
	# H_prime = (H+2*pad_num_h-HH) // stride + 1
	# W_prime = (W+2*pad_num_w-WW) // stride + 1
	H_prime = int(ceil(float(H) / float(stride)))
	W_prime = int(ceil(float(W) / float(stride)))
	out = np.zeros([N,1,H_prime,W_prime]) #SET FILTERS AS 1
	#im2col
	for im_num in range(N):
		im = x[im_num,:,:,:]
		#   im_pad = np.pad(im,((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
		im_pad = np.zeros((C, H+pad_num_h, W+pad_num_w))
		im_pad[:, pad_top:-pad_bottom, pad_left:-pad_right] = im
		# import pdb;pdb.set_trace()
		im_col = im2col(im_pad,HH,WW,stride)
		filter_col = np.reshape(w,(F,-1))
		mul = im_col.dot(filter_col.T) #+ b
		# print(mul)
		out[im_num,:,:,:] = col2im(mul,H_prime,W_prime,1)
		cache = (x, w, b, conv_param)
		return out, cache


# input = np.array([[[[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 9.0], [7.0, 8.0, 9.0, 9.0]], [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 9.0], [7.0, 8.0, 9.0, 9.0]], [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 9.0], [7.0, 8.0, 9.0, 9.0]]]]) # N, C, H, W
# weights = np.array([[[[1.0, 2.0],[1.0, 2.0]],[[1.0, 2.0],[1.0, 2.0]], [[1.0, 2.0],[1.0, 2.0]]]]) # F, C, HH, WW

# input = np.array(
# [[[[0.17014849, 0.43056882, 0.5715329 , 0.06520256, 0.12669588],
# [0.75015649, 0.98379819, 0.55574155, 0.04181346, 0.23677547],
# [0.51154924, 0.02844254, 0.60484786, 0.72306337, 0.22177844],
# [0.16487044, 0.46672951, 0.54035134, 0.69223571, 0.27845532],
# [0.66966338, 0.41083884, 0.45831479, 0.70402897, 0.61773261]]]]
# )
# weights = np.array(
# [0.92697753, 0.91485179, 0.85028299, 0.26970649, 0.55898563,
# 0.84558665, 0.75231163, 0.90343251, 0.07658575, 0.56033562,
# 0.33565241, 0.96145765, 0.24242379, 0.5888119 , 0.04742411,
# 0.96925828, 0.2795916 , 0.71978642, 0.90980128, 0.37189406,
# 0.55666793, 0.79448488, 0.44166553, 0.23985275, 0.12854726,
# 0.75676637, 0.13313323]	
# ).reshape(1, 3, 3, 3)
# weights = np.random.rand(1, 5, 3, 3)

input = np.array([[[[3, 5, 2, 7], [4, 1, 3, 8], [6, 3, 8, 2], [9, 6, 1, 5]]]])
weights = np.array([[[[1, 2, 1], [2, 1, 2], [1, 1, 2]]]])
print(input.shape)
print(weights.shape)
bias = np.zeros((1, 1))# null
conv_param = {"pad":0,"stride":2}

output, _ = conv_forward_naive(input, weights, bias, conv_param)
print(output.shape)
print("output:", output)