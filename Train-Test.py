# Train/Test 
import numpy as np 
from pylab import *

np.random.seed(2) #Allows for data to stop being randomized
websiteSpeed = np.random.normal(3.0, 1.0, 100)
userPurchase = np.random.normal(50.0, 30.0, 100) / websiteSpeed
#plt.scatter(websiteSpeed, userPurchase)
#plt.show()
#plt.xlabel('Website Speed')
#plt.ylabel('User Purchase')
#plt.title('Entire data set')

# 80% of the data will be train-data
trainX = websiteSpeed[:80]
trainY = userPurchase[:80]
# 20% of the data will be test-data
testX = websiteSpeed[80:]
testY = userPurchase[80:]

#Training data
import matplotlib.pyplot as plt1
#plt1.scatter(trainX, trainY)
#plt1.show()
plt1.xlabel('Website Speed')
plt1.ylabel('User Purchase')
plt1.title('Training Data Set')

#Testing data
import matplotlib.pyplot as plt2
#plt2.scatter(testX, testY)
#plt2.show()
plt2.xlabel('Website Speed')
plt2.ylabel('User Purchase')
plt2.title('Testing Data Set')

# 8th Degree Over-Fitting Polynomial to fit on the training data:
x = np.array(trainX)
y = np.array(trainY)
p4 = np.poly1d(np.polyfit(x,y,8))

import matplotlib.pyplot as plt
xp = np.linspace(0,7,100)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0,200])
#plt.scatter(x,y)
#plt.plot(xp, p4(xp), c='r')
#plt.xlabel('Website Speed')
#plt.ylabel('User Purchase')
#plt.title('Overfitting Training Data Set')
#plt.show()

# 8th Degree Over-Fitting Polynomial to fit on the testing data:
x2 = np.array(testX)
y2 = np.array(testY)
p5 = np.poly1d(np.polyfit(x,y,8))

import matplotlib.pyplot as plt
xp2 = np.linspace(0,7,100)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0,200])
plt.scatter(x2,y2)
plt.plot(xp2, p5(xp2), c='r')
plt.xlabel('Website Speed')
plt.ylabel('User Purchase')
plt.title('Overfitting Testing Data Set')
plt.show()

# Testing r-squared value for test data:
from sklearn.metrics import r2_score
#r2 = r2_score(testY, p4(testX))
#print r2
# 0.3 is a very weak value

# Testing r-squared value for training data
r2 = r2_score(np.array(trainY), p4(np.array(trainX)))
print r2
# 0.6 stronger value for training data

# A smaller model (smaller degree polynomial will pass the data set)
