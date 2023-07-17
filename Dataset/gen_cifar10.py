from keras.datasets import cifar10

(XTrain, yTrain), (XTest, yTest) = cifar10.load_data()
yTrain = yTrain.ravel()
yTest = yTest.ravel()


from skimage.feature import hog
import numpy as np




XTrain_hog_fd = []
for feature in XTrain:
    fd = hog(feature.reshape((32, 32, 3)), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    XTrain_hog_fd.append(fd)
XTrain_features = np.array(XTrain_hog_fd, 'float64')

print(XTrain_features.shape)



XTest_hog_fd = []
for feature in XTest:
    fd = hog(feature.reshape((32, 32, 3)), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    XTest_hog_fd.append(fd)
XTest_features = np.array(XTest_hog_fd, 'float64')




filename = 'cifar10.npy'
with open(filename, 'wb') as f:
    np.save(f, XTrain_features)
    np.save(f, yTrain)
    np.save(f, XTest_features)



