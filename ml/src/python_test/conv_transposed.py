import numpy as np
from math import floor
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

def col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c):
	"""
	Args:
	dim_col: gradients for im_col,(h_prime*w_prime,hh*ww*c)
	h_prime,w_prime: height and width for the feature map
	strid: stride
	hh,ww,c: size of the filters
	Returns:
	dx: Gradients for x, (C,H,W)
	"""
	H = (h_prime - 1) * stride + hh
	W = (w_prime - 1) * stride + ww
	dx = np.zeros([c,H,W])
	for i in range(h_prime*w_prime):
		row = dim_col[i,:]
		h_start = int((i / w_prime) * stride)
		w_start = (i % w_prime) * stride
		# print(h_start, h_start+hh, w_start, w_start+ww)
		dx[:,h_start:h_start+hh,w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
	return dx

def conv_trans_naive(x, w, conv_param):
	pad_type = conv_param['pad']
	stride = conv_param['stride']
	N,C,H,W = x.shape
	F,C,HH,WW = w.shape
	# pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right = get_same_padding(H, W, stride, HH, WW)
	# assert((W + 2 * pad_num - WW) % stride == 0)
	# assert((H + 2 * pad_num - HH) % stride == 0)
	# H_prime = (H+2*pad_num-HH) // stride + 1
	# W_prime = (W+2*pad_num-WW) // stride + 1
	# H_prime = int(ceil(float(H) / float(stride)))
	# W_prime = int(ceil(float(W) / float(stride)))
	H_prime = (H - 1)*stride + HH
	W_prime = (W - 1)*stride + WW
	# out = np.zeros([1,F,H_prime,W_prime]).astype('float64') #SET FILTERS AS 1
	#im2col
	pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right = get_same_padding(H, W, stride, HH, WW)
	pad_num = HH - 1 # kernel - 1
	for im_num in range(N):
		im = x[im_num,:,:,:]
		im_pad = np.pad(im,((0,0), (pad_num,pad_num),(pad_num,pad_num)),'constant').astype('float64')
		im_col = im2col(im_pad,HH,WW,stride)
		# print(im_col.shape)
		filter_col = np.reshape(np.flip(w, (2,3)),(F,-1))
		mul = im_col.dot(filter_col.T)#+ b
		output = mul.reshape(1, F, H_prime, W_prime)
		# out = output.transpose(1, 0, 2, 3)
		cache = (x, w, conv_param)
		output = output[:, :, pad_left:-pad_right, pad_top:-pad_bottom]
		return output, cache


# input = np.array([[[2,1,4,4]]]).reshape(1,1,2,2)
# weights = np.array([[[[1, 3, 3], [3, 4, 1], [1, 4, 1]]]]) # FLIPPED
# weights = np.array([[[[1,4,1],[1,4,3],[3,3,1]]]])
# input = np.array([[[[55, 52], [57,50]]]])
# weights = np.array([[[[1, 2], [2, 1]]]])
# input = np.array([[[[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 9.0], [7.0, 8.0, 9.0, 9.0]]]])
# weights = np.array([[[[80, 20], [-13, 12]]]])

input = np.array(
[[[[0.17014849, 0.43056882, 0.5715329 , 0.06520256, 0.12669588],
[0.75015649, 0.98379819, 0.55574155, 0.04181346, 0.23677547],
[0.51154924, 0.02844254, 0.60484786, 0.72306337, 0.22177844],
[0.16487044, 0.46672951, 0.54035134, 0.69223571, 0.27845532],
[0.66966338, 0.41083884, 0.45831479, 0.70402897, 0.61773261]]]]
)
weights = np.array(
[
0.92697753, 0.91485179, 0.85028299, 0.26970649, 0.55898563, 0.84558665, 0.75231163,
0.90343251, 0.07658575
]).reshape(1,1,3,3)
print("input", input.shape)
print("weights", weights.shape)
bias = np.zeros((1, 1))# null
conv_param = {"pad":"same","stride":1}

out, _ = conv_trans_naive(input, weights, conv_param)
# out = Conv2DTranspose(input.reshape(1, 4, 4, 1), weights, padding="valid", strides=(1,1))
print(out.astype('float32'))