import numpy as np
from math import ceil

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
		dx[:,h_start:h_start+hh,w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
	return dx


def conv_backward_naive(grad, cache):
	dx, dw, db = None, None, None

	x, w, conv_param = cache
	pad_num = conv_param['pad']
	stride = conv_param['stride']
	N,C,H,W = x.shape
	F,C,HH,WW = w.shape
	H_prime = (H+2*pad_num-HH) // stride + 1
	W_prime = (W+2*pad_num-WW) // stride + 1

	dw = np.zeros(w.shape)
	dx = np.zeros(x.shape)
	# db = np.zeros(b.shape)

	# We could calculate the bias by just summing over the right dimensions
	# Bias gradient (Sum on dout dimensions (batch, rows, cols)
	#db = np.sum(dout, axis=(0, 2, 3))

	for i in range(N):
		im = x[i,:,:,:]
		dout = np.pad(im,((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
		# im_col = im2col(im_pad,HH,WW,stride)
		filter_col = np.reshape(w,(F,-1)).T

		# print(dout)

		# dout_i = dout[:,:,:]
		dbias_sum = np.reshape(dout,(F,-1))
		dbias_sum = dbias_sum.T

		#bias_sum = mul + b
		# db += np.sum(dbias_sum,axis=0)
		dmul = dbias_sum

		#mul = im_col * filter_col
		# dfilter_col = (im_col.T).dot(dmul)
		dim_col = dmul.dot(filter_col.T)
		
		dx_padded = col2im_back(dim_col,H_prime,W_prime,stride,HH,WW,C)
		print(dx_padded)
		dx[i,:,:,:] = dx_padded[:,pad_num:H+pad_num,pad_num:W+pad_num]
		# dw += np.reshape(dfilter_col.T,(F,C,HH,WW))
	return dx, dw, db

input = np.array([[[[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 9.0], [7.0, 8.0, 9.0, 9.0]]]])
weights = np.array([[[[80, 20], [-13, 12]]]])
conv_param = {"pad":1,"stride":1}
dx, dw, db = conv_backward_naive(input, (input, weights, conv_param))
print(dx)