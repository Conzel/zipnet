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
	C,F,HH,WW = w.shape
	if stride != 1:
		stride_h = stride*(x.shape[2])
		stride_w  = stride*(x.shape[3])
		x_stride = np.zeros((*x.shape[:2], stride_h, stride_w))
		for i in range(x.shape[2]):
			for j in range(x.shape[3]):
				x_stride[:, :, i*stride, j*stride] = x[:, :, i, j]
		# import pdb;pdb.set_trace()
		x = x_stride[:,:,:-stride+1,:-stride+1]
	# pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right = get_same_padding(H, W, stride, HH, WW)
	# assert((W + 2 * pad_num - WW) % stride == 0)
	# assert((H + 2 * pad_num - HH) % stride == 0)
	# H_prime = (H+2*pad_num-HH) // stride + 1
	# W_prime = (W+2*pad_num-WW) // stride + 1
	# H_prime = int(ceil(float(H) / float(stride)))
	# W_prime = int(ceil(float(W) / float(stride)))
	H_prime = (H - 1)*stride + HH # THIS IS CORRECT
	W_prime = (W - 1)*stride + WW # THIS IS CORRECT
	# out = np.zeros([1,F,H_prime,W_prime]).astype('float32') #SET FILTERS AS 1
	#im2col
	pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right = get_same_padding(x.shape[2], x.shape[3], stride, HH, WW)
	pad_num = HH - 1 # kernel - 1
	for im_num in range(N):
		im = x[im_num,:,:,:]
		# im_pad_ = np.pad(im,((0,0), (1,1),(1,1)),'constant').astype('float32')
		im_pad = np.pad(im,((0,0), (pad_num,pad_num),(pad_num,pad_num)),'constant').astype('float32')
		im_col = im2col(im_pad,HH,WW,1)
		print(im_pad.shape)
		print(im_col.shape)
		# import pdb;pdb.set_trace()
		filter_col = np.reshape(np.flip(w, (2,3)),(F,-1))
		mul = im_col.dot(filter_col.T)#+ b
		# print(mul)
		output = mul.reshape(1, F, H_prime, W_prime)
		# out = output.transpose(1, 0, 2, 3)
		cache = (x, w, conv_param)
		if pad_type == "same":
			output = output[:, :, pad_left:-pad_right, pad_top:-pad_bottom]
		return output, cache

input = np.array([
            0.17014849, 0.4305688, 0.5715329, 0.06520256, 0.12669589, 0.7501565, 0.9837982,
            0.55574155, 0.04181346, 0.23677547, 0.51154923, 0.02844254, 0.60484785, 0.72306335,
            0.22177844, 0.16487044, 0.46672952, 0.54035133, 0.6922357, 0.27845532, 0.66966337,
            0.41083884, 0.4583148, 0.70402896, 0.6177326, 0.9269775, 0.56033564, 0.9098013,
            0.2697065, 0.24242379, 0.7944849, 0.75231165, 0.9692583, 0.12854727, 0.9148518,
            0.3356524, 0.37189406, 0.55898565, 0.5888119, 0.44166553,
        ]).reshape(1, 2, 5, 4)

weights = np.array([
            0.9034325, 0.2795916, 0.7567664, 0.85028297, 0.96145767, 0.5566679, 0.84558666,
            0.0474241, 0.23985276, 0.07658575, 0.7197864, 0.13313323, 0.69580543, 0.12692,
            0.38484824, 0.775336, 0.52113837, 0.4364637, 0.14352316, 0.8997107, 0.64410555,
            0.04471071, 0.767672, 0.43464628, 0.16569944, 0.18875164, 0.12285258, 0.2781115,
            0.5390728, 0.5066572, 0.97435564, 0.39133722,
        ]).reshape(2,1,4,4)

print("input", input.shape)
print("weights", weights.shape)
bias = np.zeros((1, 1))# null
conv_param = {"pad":"same","stride":2}

out, _ = conv_trans_naive(input, weights, conv_param)
# out = Conv2DTranspose(input.reshape(1, 4, 4, 1), weights, padding="valid", strides=(1,1))
print(out.astype('float32'))
print(out.shape)