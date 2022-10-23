from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mlxtend.plotting import plot_decision_regions
import plotly.graph_objects as go

from ui import Ui 
from helper import Helper 
from models import ModelGenerator

class App:

    def __init__(self):
        self.ui = Ui()
        self.helper = Helper()
        st.set_page_config(
            page_title="Machine learning playground",
            layout="centered",
            page_icon="./images/neural.png"
        )


    def init_sideBar(self):
        data, n_data, noise, test_size  = self.ui.data_sidebar()
        self.data = self.helper.create_data(data, n_data, noise, test_size,)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data 
        
        self.model, model_name = self.ui.model_sidebar()
        accuracy, precision, recall, f1=self.model.train_predict_model(self.data)
        self.metrics = [accuracy, precision, recall, f1]

        self.test_metrics = self.model.predict(self.X_train, self.y_train, )

    

    def main(self):
        self.ui.explanation()
        
        
        
        st.markdown("### Train data({0})".format(len(self.y_train)))
        col1, col2 = st.columns(2)
        with col1:
            test_plot_display = st.empty()
            fig = plt.figure()
            plot_decision_regions(self.X_train, self.y_train, clf =  self.model.model, legend= 2)
            test_plot_display = st.pyplot(fig = fig)
        
        with col2:
            metrics_name = ["Accuracy", "Precision", "Recall", "F1"]
            self.helper.plot_scores(metrics_name, self.metrics)
        
        st.markdown("### Test data ({0})".format(len(self.y_test)))
        col3, col4 = st.columns(2)
        

        with col3:
            
            
            plot_display = st.empty()
            fig = plt.figure()
            plot_decision_regions(self.X_test, self.y_test, clf =  self.model.model, legend= 2)
            plot_display = st.pyplot(fig=fig)


        with col4:
             
            metrics_name = ["Accuracy", "Precision", "Recall", "F1"]

            self.helper.plot_scores(metrics_name, self.test_metrics)
        
 



    def run(self):
        self.init_sideBar()
        self.main()

if __name__ == '__main__':
    app = App()
    app.run()


        