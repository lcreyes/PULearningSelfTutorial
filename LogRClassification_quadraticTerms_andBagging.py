print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import linear_model
from scipy import interpolate
from scipy import integrate

#define constants

n_l = 30 #number of labeled positives
n_p = 100 #number of total positives
n_n = 200 + n_l - n_p # number of negative points, first term is number of unlabeled points
n_T = 200 # number of bootstraps
n_K = 100 # size of bootstrap sample

#define Gaussians mean and covariance matrices

mean1 = [0,0]
mean2 = [1,1]

cov1 = [[1, 0.5], [0.5, 1]]
cov2 = [[1, -0.5], [-0.5, 1]]


#random generated data
positive = np.transpose(np.random.multivariate_normal(mean1,cov1,n_l).T)
unlabeled_positive = np.transpose(np.random.multivariate_normal(mean1,cov1,n_p-n_l).T)
negative = np.transpose(np.random.multivariate_normal(mean2,cov2,n_n).T)

positive_x = positive[:,0]
positive_y = positive[:,1]

positive_x2 = positive_x*positive_x
positive_y2 = positive_y*positive_y

unlabeled_positive_x = unlabeled_positive[:,0]
unlabeled_positive_y = unlabeled_positive[:,1]

unlabeled_positive_x2 = unlabeled_positive_x*unlabeled_positive_x
unlabeled_positive_y2 = unlabeled_positive_y*unlabeled_positive_y


negative_x = negative[:,0]
negative_y = negative[:,1]

negative_x2 = negative_x*negative_x
negative_y2 = negative_y*negative_y

positive=np.column_stack((positive_x, positive_y, positive_x2, positive_y2))
unlabeled_positive = np.column_stack((unlabeled_positive_x, unlabeled_positive_y, 
                                      unlabeled_positive_x2, unlabeled_positive_y2))
negative=np.column_stack((negative_x, negative_y, negative_x2, negative_y2))

#shuffle unlabeled data
unlabeled_data_matrix = np.vstack((unlabeled_positive, negative))

unlabeled_data_slices = np.vsplit(unlabeled_data_matrix, unlabeled_data_matrix.shape[0])

grid_points = 500
grid_size = 7.5
 #xy grid of points to be evaluated with classifier
xx, yy = np.meshgrid(np.linspace(-grid_size, grid_size, grid_points),  
                     np.linspace(-grid_size, grid_size, grid_points))
xx2 = xx*xx
yy2 = yy*yy

Z= np.zeros((grid_points, grid_points))

for ith_bootstrap in range(n_T):
    random.shuffle(unlabeled_data_slices)
    unlabeled_data = np.vstack(unlabeled_data_slices[0:n_K])
    data = np.vstack((positive, unlabeled_data))
    label = [1]*n_l + [0]*(n_K)
    # run classification
    logreg = linear_model.LogisticRegression()
    logreg.fit(data, label)
    #evaluate probabilities
    prob = logreg.predict_proba(np.c_[xx.ravel(), yy.ravel(), xx2.ravel(), yy2.ravel()])[:,1]
    Z+= prob.reshape(xx.shape)
   
   
Z/=n_T

all_data = np.vstack((positive, unlabeled_positive, negative))
all_data_label = [1]*n_l + [0]*(n_p-n_l+n_n)

#plot 
map = plt.figure()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0.25, 0.5, 0.75], linewidths=2,
                       linetypes='--')
 
plt.scatter(all_data[:, 0], all_data[:, 1], s=30, c=all_data_label, cmap=plt.cm.Paired)

plt.clabel(contours, inline=1, fontsize=15)
plt.axis([-grid_size, grid_size, -grid_size, grid_size])
map.show()
 
test_np = 1500
test_nn = 1500
 
test_positive = np.transpose(np.random.multivariate_normal(mean1,cov1,test_np).T)
test_negative = np.transpose(np.random.multivariate_normal(mean2,cov2,test_nn).T)

    
f = interpolate.RectBivariateSpline(np.linspace(-grid_size, grid_size, grid_points), 
                                    np.linspace(-grid_size, grid_size, grid_points), Z)



roc = []
for threshold in np.linspace(1.0, 0.0, num=50):
    true_positives = 0.0
    false_positives = 0.0
    for i in range(test_np):
        prediction = f.ev(test_positive[i][0], test_positive[i][1])
        if (prediction > threshold): # true positive
            true_positives+=1
    for j in range (test_nn):
        prediction = f.ev(test_negative[j][0], test_negative[j][1])
        if (prediction > threshold): #false positive
           false_positives+=1
    roc.append((false_positives/test_nn, true_positives/test_np,))
            
 
roc_plot = plt.figure()           
plt.scatter(*zip(*roc))
plt.ylim((0.0, 1.0))
plt.xlim((0.0, 1.0))
plt.xlabel('False Positives Rate')
plt.ylabel('True Positives Rate')
roc_plot.show()

auc = integrate.trapz(np.asarray(zip(*roc)[1]), np.asarray(zip(*roc)[0]))

point = (interpolate.interp1d(np.asarray(zip(*roc)[0]), np.asarray(zip(*roc)[1])))(0.2)

print auc, point