

from keras.datasets import mnist

(XTrain, yTrain), (XTest, yTest) = mnist.load_data()


from skimage.feature import hog
import numpy as np




XTrain_hog_fd = []
for feature in XTrain:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    XTrain_hog_fd.append(fd)
XTrain_features = np.array(XTrain_hog_fd, 'float64')

print(XTrain_features.shape)



XTest_hog_fd = []
for feature in XTest:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    XTest_hog_fd.append(fd)
XTest_features = np.array(XTest_hog_fd, 'float64')




filename = 'MNIST.npy'
with open(filename, 'wb') as f:
    np.save(f, XTrain_features)
    np.save(f, yTrain)
    np.save(f, XTest_features)



