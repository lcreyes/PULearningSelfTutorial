import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap

#define constants

n_l = 500 #number of labeled positive points
n_p = 500 #number of positive points
n_n = 1000 # number of negative points

#define Gaussians mean and covariance matrices

mean1 = [0,0]
mean2 = [4,3]

cov1 = [[1, 0.5], [0.5, 1]]
cov2 = [[1, -0.5], [-0.5, 1]]


#random generated data
positive = np.transpose(np.random.multivariate_normal(mean1,cov1,n_p).T)
negative = np.transpose(np.random.multivariate_normal(mean2,cov2,n_n).T)

data = np.vstack((positive,negative))
label = [1]*n_l + [-1]*(n_p+n_n-n_l)


# run classification
clf = neighbors.KNeighborsClassifier(15, 'uniform')
clf.fit(data, label)

#xy grid of points to be evaluated with classifier
xx, yy = np.meshgrid(np.linspace(-10, 10, 200),  np.linspace(-10, 10, 200))

#calculate "distances" to hyperplane
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


#plot 
plt.figure()
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(data[:, 0], data[:, 1], s=30, c=label, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-10, 10, -10, 10])
plt.show()
