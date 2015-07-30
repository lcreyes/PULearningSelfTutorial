print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#define constants

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
label = [1]*n_p + [-1]*n_n


# run classification
clf = svm.NuSVC()
clf.fit(data, label)

#xy grid of points to be evaluated with classifier
xx, yy = np.meshgrid(np.linspace(-10, 10, 200),  np.linspace(-10, 10, 200))

#calculate "distances" to hyperplane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


#plot 
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[-0.5, 0, 0.5], linewidths=2,
                       linetypes='--')

plt.scatter(data[:, 0], data[:, 1], s=30, c=label, cmap=plt.cm.Paired)

plt.clabel(contours, inline=1, fontsize=10)

plt.xticks(())
plt.yticks(())
plt.axis([-10, 10, -10, 10])
plt.show()
