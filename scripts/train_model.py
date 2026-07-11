import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib

X = []
y = []

dataset = "dataset"

for label in os.listdir(dataset):

    folder = os.path.join(dataset,label)

    for file in os.listdir(folder):

        if file.endswith(".npy"):

            data = np.load(os.path.join(folder,file))

            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X,y)

joblib.dump(model,"model/gesture_model.pkl")

print("Model saved to model/gesture_model.pkl")
