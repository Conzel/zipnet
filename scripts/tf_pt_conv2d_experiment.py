# coding: utf-8
import torch
import numpy as np
import torch.nn.functional as f
import tensorflow as tf

k = np.random.rand(3, 2, 5, 5).astype(np.float32)
x = np.random.rand(1, 2, 10, 10).astype(np.float32)
x_ = np.moveaxis(x, 1,3)
k_ = np.moveaxis(k, [0, 1], [3, 2])
tf.nn.conv2d(x_, k_, strides=1, padding="VALID")
o_ = tf.nn.conv2d(x_, k_, strides=1, padding="VALID")
o = f.conv2d(torch.FloatTensor(x), torch.FloatTensor(k), stride=1)
o = o.numpy()
o = np.moveaxis(o, 1, 3)
o_ = o_.numpy()
(o - o_).sum()
