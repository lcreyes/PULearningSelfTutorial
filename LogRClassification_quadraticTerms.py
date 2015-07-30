print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import interpolate
from scipy import integrate

#define constants

n_l = 30
n_p = 100 #number of positive points
n_n = 130 # number of negative points

#define Gaussians mean and covariance matrices

mean1 = [0,0]
mean2 = [1,1]

cov1 = [[1, 0.5], [0.5, 1]]
cov2 = [[1, -0.5], [-0.5, 1]]


#random generated data
positive = np.transpose(np.random.multivariate_normal(mean1,cov1,n_p).T)
negative = np.transpose(np.random.multivariate_normal(mean2,cov2,n_n).T)

positive_x = positive[:,0]
positive_y = positive[:,1]

positive_x2 = positive_x*positive_x
positive_y2 = positive_y*positive_y


negative_x = negative[:,0]
negative_y = negative[:,1]

negative_x2 = negative_x*negative_x
negative_y2 = negative_y*negative_y

positive=np.column_stack((positive_x, positive_y, positive_x2, positive_y2))
negative=np.column_stack((negative_x, negative_y, negative_x2, negative_y2))

data = np.vstack((positive, negative))
label = [1]*n_l + [0]*(n_p+n_n-n_l)


# run classification
logreg = linear_model.LogisticRegression()
logreg.fit(data, label)


grid_points = 500
grid_size = 7.5

#xy grid of points to be evaluated with classifier
xx, yy = np.meshgrid(np.linspace(-grid_size, grid_size, grid_points),  
                     np.linspace(-grid_size, grid_size, grid_points))
xx2 = xx*xx
yy2 = yy*yy

#calculate "distances" to hyperplane
Z = logreg.predict_proba(np.c_[xx.ravel(), yy.ravel(), xx2.ravel(), yy2.ravel()])[:,1]
Z = Z.reshape(xx.shape)

e1=0
n_e1=0
for i in range(0,n_p+n_n):
    if (label[i]==1):
        e1+=logreg.predict_proba(np.c_[positive_x[i], positive_y[i], 
                                       positive_x2[i], positive_y2[i]])[0][1]
        n_e1+=1
        
e1/=n_e1

Z = Z/e1

#plot 
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0.5], linewidths=2,
                       linetypes='--')

plt.scatter(data[:, 0], data[:, 1], s=30, c=label, cmap=plt.cm.Paired)

plt.clabel(contours, inline=1, fontsize=15)

plt.xticks(())
plt.yticks(())
plt.axis([-10, 10, -10, 10])
plt.show()




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