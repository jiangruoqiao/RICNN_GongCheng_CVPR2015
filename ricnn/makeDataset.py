import tensorflow as tf
import h5py
import numpy as np
from scipy.ndimage.interpolation import rotate


def _pad(loaded_x, loaded_size, desired_size):
    padding_size = (desired_size - loaded_size) / 2
    padding_list = [[0, 0],
                    [padding_size, padding_size + 1],
                    [padding_size, padding_size + 1],
                    [0, 0]]
    return np.pad(np.reshape(loaded_x, [-1, loaded_size, loaded_size, 1]),
                  padding_list,
                  'constant',
                  constant_values = 0)


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

file_name = '/home/liuqi/Desktop/mnist_rotation_new/mnist_all_rotation_normalized_float_test.h5'
file = h5py.File(file_name,'r')

pad_x = _pad(file['data'], 28, 227)
labels = file['labels']
#dataset = _transform(pad_x, 36)
print pad_x.shape
print labels.shape
file_save = h5py.File('/home/liuqi/Desktop/mnist_rotation_new/test.h5','w')
#file_save['data'] = pad_x
#file_save['label'] = labels
file_save.create_dataset ('data',data = pad_x)
file_save.create_dataset('labels', data = labels)
file_save.close()