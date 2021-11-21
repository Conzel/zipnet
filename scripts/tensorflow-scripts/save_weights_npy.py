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

def save_weights(npz_path):
    npz = np.load(npz_path, mmap_mode='r')
    file_names = [k for k in npz.files]
    file_names.sort()
    for name in file_names:
        print(name)    

if __name__ == '__main__':
    weights_file = sys.argv[1]
    assert(os.path.exists(weights_file))
    if not weights_file.split(".")[-1] == "npz":
        print("Not a valid weights folder, please provide an npz file")
    
    save_weights(weights_file)
    
    