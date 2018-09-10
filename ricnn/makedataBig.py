import h5py
import numpy as np
import math
from scipy.ndimage.interpolation import rotate


def _transform(padded, number_of_transformations):
    tiled = np.tile(np.expand_dims(padded, 4), [number_of_transformations])
    for transformation_index in xrange(number_of_transformations):
        angle = 360.0 * transformation_index / float(number_of_transformations)
        tiled[:, :, :, :, transformation_index] = rotate(
            tiled[:, :, :, :, transformation_index],
            angle,
            axes = [1, 2],
            reshape = False)
    return tiled

file_load = h5py.File('/home/liuqi/Desktop/mnist_rotation_new/trainRICNN.h5','r')
file_save = h5py.File('/media/liuqi/Files/dataset/train_mnist_ricnn.h5','w')

data = file_save.create_dataset('data', (12000,227,227,1,8))
#file = h5py.File('')
print file_load['data'][1,:,:,:].shape
for number in range(0,120,1):
    print number
    load = file_load['data'][number*100:100*(number+1),:,:,:]
    load = _transform(load, 8)
    data[number*100:100*(number+1),:,:,:,:] = load
