from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()

#Train a Logistic Regression Classifier to predict whether a flower is iris virginica or not!
x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)

#astype converts true condition to 1 and others to 0.


#Train logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)

example = clf.predict(([[2.6]]))
print(example)

#Visualization

x_new = np.linspace(0,3,1000).reshape(-1,1)
print(x_new)

y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1], "-g", label = "virginica")
plt.show()




#print(iris['data'].shape)
# print(list(iris.keys()))
# print(list(iris['target']))
# print(list(iris['data']))
# print(list(iris['DESCR']))