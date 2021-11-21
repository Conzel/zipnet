import numpy as np 
import sys
import os

def print_layer(weights_folder, n):
    npy_files = os.listdir(weights_folder)
    npy_files.sort()
    for npy in npy_files:
        print("Layer name:", npy)
        arr = np.load(os.path.join(weights_folder, npy))
        print("Layer shape:", arr.shape)
        print(arr[0,0,0,:n])

def print_weights(npz_path, n):
    npz = np.load(npz_path, mmap_mode='r')
    file_names = [k for k in npz.files]
    file_names.sort()
    for name in file_names:
        if 'kernel' in name.split("/") and 'encoder_layer_0' in name.split("/"):
            npy = npz[str(name)]
            print(name)
            print(npy.shape)
            import pdb;pdb.set_trace()

if __name__ == '__main__':
    weights_folder = sys.argv[1]
    n = int(sys.argv[2])
    if weights_folder.split(".")[-1] == "npz":
        print_weights(weights_folder, n)
    else: 
        print_layer(weights_folder, n)
    
    