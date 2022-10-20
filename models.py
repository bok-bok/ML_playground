import streamlit as st 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class ModelGenerator:

    def __init__(self):
        self.model = None

    def model_generator(self, model_name):
        if model_name == 'Logistic Regression':
            self.lr_generator()
        elif model_name == "Decision Tree":
            self.dt_generator()
        elif model_name == "Random Forest":
            self.rf_generator()
        elif model_name == "K Nearest Neighbors":
            self.knn_generator()
        elif model_name == "SVC":
            self.svc_generator()

    def rf_generator(self):

        criterion = st.selectbox("criterion", ["gini", "entropy"])
        n_estimators = st.number_input("n_estimators", 50, 300, 100, 10)
        max_depth = st.number_input("max_depth", 1, 50, 5, 1)
        min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
        max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

        params = {
            "criterion": criterion,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "n_jobs": -1,
        }

        self.model = RandomForestClassifier(**params)

    def lr_generator(self):
        max_iter = st.number_input("max_iter", 100, 2000, 500, 100)
        self.model = LogisticRegression(max_iter = max_iter)

    def dt_generator(self):
        criterion = st.selectbox("criterion", ["gini", "entropy"])
        max_depth = st.number_input("max_depth", 1, 50, 5, 1)
        min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
        max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

        params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
        }
        self.model = DecisionTreeClassifier(**params)


    def knn_generator(self):
        n_neighbors = st.number_input("n_neighbors",3, 15, 5, 1)

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def svc_generator(self):
        C = st.number_input("C", 0.01, 2.0, 1.0, 0.01)
        kernel = st.selectbox("kernel", ("rbf", "linear", "poly", "sigmoid"))
        params = {"C": C, "kernel": kernel}
        self.model = SVC(**params)



    def train_predict_model(self,data, n_center):
        average = 'binary'
        if n_center is not None and n_center > 2:
            average = 'weighted'
        X_train, y_train, X_test, y_test = data 
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=average)
        recall = recall_score(y_test, pred, average = average)
        f1 = f1_score(y_test, pred, average = average) 
        return acc, precision, recall, f1 

    def predict(self,X_test, y_test, n_center):
        average = 'binary'
        if n_center is not None and n_center > 2:
            average = 'weighted'
        pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=average)
        recall = recall_score(y_test, pred, average =average)
        f1 = f1_score(y_test, pred, average = average)
        return acc, precision, recall, f1 
    