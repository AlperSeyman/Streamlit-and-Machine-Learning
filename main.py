import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


st.write("""
# Explore Different Classifier
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris Dataset","Breast Cancer Dataset", "Wine dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier",("KNN", "SVM", "Random Forest"))

def access_dataset(dataset_name):
    if dataset_name == "Iris Dataset":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer Dataset":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y 

X, y = access_dataset(dataset_name)
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 13)
        params["K"] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0 )
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)


def create_classifier(classifier_name, params):
    if classifier_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        classifier = SVC(C=params["C"])
    else:
        classifier = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"], random_state=42)
    return classifier

classifier = create_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier.fit(X_train, y_train)
y_prediction = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_prediction)
st.write(f"Classifier: {classifier_name}")
st.write(f"Accuracy Score: {accuracy}")
