import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
x,y=iris.data,iris.target

X_train,x_test,y_train,y_test=train_test_split(x,y,random_state=123,test_size=0.2)

from knn import KNN
knn_model=KNN(k=3)
knn_model.fit(X_train,y_train)
pred=knn_model.predict(x_test)

accuracy=np.sum((pred==y_test)/len(y_test))
print(accuracy)