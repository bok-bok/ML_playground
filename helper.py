from matplotlib.pyplot import plot
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mlxtend.plotting import plot_decision_regions
class Helper:
    def __init__(self):
        pass 

    def create_data(self, data, n_data, noise_str, test_proportion, n_center):
        if noise_str == 'No noise':
            noise = 0
        elif noise_str == 'Little noise':
            noise = 0.1
        else:
            noise = 0.15

        if data == "moons":
            X, y = make_moons(n_samples=n_data, noise= noise)
            
        elif data == "circles":
            X, y = make_circles(n_samples=n_data, noise=noise)
            
        elif data == "blobs":
            X, y = make_blobs(
                n_features=2,
                n_samples=n_data,
                centers=n_center,
                cluster_std=  noise * 47 + 0.57,
                random_state=42,
            )
           

            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion/100, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, y_train, X_test, y_test 



    def plot_scores(self, metrics, scores):
        fig = make_subplots(rows = 2, cols = 2,
                    specs=[[{"type": "indicator"},{"type": "indicator"} ],
                    [{"type": "indicator"},{"type": "indicator"}]])
        self.plot_score(fig, metrics[0], scores[0],1,1)
        self.plot_score(fig, metrics[1], scores[1],1,2)
        self.plot_score(fig, metrics[2], scores[2],2,1)
        self.plot_score(fig, metrics[3], scores[3],2,2)
        st.plotly_chart(fig)
    def plot_score(self,fig, metrics, score,col,row):
        value = np.round(score * 100,0)
        if value > 80:
            color = '#36AE7C'
        elif value > 50:
            color = '#F9D923'
        else:
            color = '#EB5353'
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = value,
            
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1,'visible': False},
                'bar': {'color': color},
            },
            #domain = {'x': [0, .25], 'y': [0, .25]},
            title = {'text': metrics}),
            row = row,
            col = col)
        
        