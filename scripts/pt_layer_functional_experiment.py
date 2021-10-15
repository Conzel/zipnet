# coding: utf-8

im = np.random.rand(*im_shape)
import numpy as np
import torch
im = np.random.rand(*im_shape)
im = np.random.rand(*(2,5,4))
ker = np.random.rand(2,,1,4,4)
ker = np.random.rand(2,1,4,4)
im_tf = torch.Tensor(np.expand_dims(im, axis=0))
im_tf
conv = torch.nn.ConvTranspose2d(im.shape[0], ker.shape[0], ker.shape[2])
with torch.no_grad():
    conv.weight=torch.nn.Parameter(torch.from_numpy(ker).float())
    
out_tf = conv(im_tf)
conv = torch.nn.ConvTranspose2d(im.shape[0], ker.shape[0], ker.shape[2], bias=False, padding=0)
with torch.no_grad():
    conv.weight=torch.nn.Parameter(torch.from_numpy(ker).float())
    
out_tf = conv(im_tf)
out = np.squeeze(out_tf.detach().numpy(), axis=0)
out
out_tf
out
out_tf
torch.nn.functional.conv_transpose2d(im_tf, ker_tf)
ker
torch.nn.functional.conv_transpose2d(im_tf, ker)
ker_tf = torch.Tensor(ker)
torch.nn.functional.conv_transpose2d(im_tf, ker)
torch.nn.functional.conv_transpose2d(im_tf, ker_tf)
out2 = torch.nn.functional.conv_transpose2d(im_tf, ker_tf)
out
out2
out2
out
out2 == out
out[0]
out[0,0,0]
out2[0,0,0]
out2
out2.shape
out.shape
get_ipython().run_line_magic('save', 'pt_layer_functional_experiment.py 0-39')
