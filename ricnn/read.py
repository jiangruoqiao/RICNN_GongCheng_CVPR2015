import h5py
file_name = '/home/liuqi/Desktop/mnist_rotation_new/test.h5'
file = h5py.File(file_name,'r')


print file.keys()
print file['data'].shape