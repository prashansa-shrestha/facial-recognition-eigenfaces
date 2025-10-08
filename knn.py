from collections import Counter
import numpy as np


#returns predictions corresponding to each images in the test set
def kNN(X_train,y_train,X_test,k=5):
    y_prediction=[]
    for test in X_test: #iterates through each row in X_test
        distances=np.linalg.norm(X_train-test,axis=1)
        knn_labels=y_train[np.argsort(distances)[:k]] 
        prediction=Counter(knn_labels).most_common(1)[0][0]

        y_prediction.append(prediction)

    return np.array(y_prediction) #returns a one D array consisting of consecutive predictions for each x_test