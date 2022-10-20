import numpy as np
import streamlit as st 
from models import ModelGenerator
class Ui:
    
    def __init__(self):
        self.model_generator = ModelGenerator() 

    def explanation(self):
        st.title("Machine Learning Playground")
        st.subheader("In this Playground, you are able to play with various ML algorithms and their hyper-parameters")

        st.markdown("""
            ### How to play here
            - Choose a data to play with 🥕
            - Choose model and hyper-parameters 🔪
            - Enjoy the live result 🍲
        """)

    def data_sidebar(self):
        data_container = st.sidebar.expander("Generate data",True)
        with data_container:
            data = st.selectbox("Choose a data", ("moons", "circles", "blobs"))
            n_data = st.slider(
                "Number of data",
                100,
                1000,
                200
                
            )
            self.noise = st.selectbox("Amount of noise in data", ("No noise", "Little noise", "A lot of noise"))
            n_center = None
            if data == "blobs":
                n_center=st.number_input(
                    "Number of center",
                    1,
                    5,
                    2,
                    1
                )



            test_size = st.slider("Test data proportion",0,100,20)
        return data, n_data, self.noise, test_size, n_center


    def model_sidebar(self):
        model_container = st.sidebar.expander("Your model", True)
        with model_container:
            model_name = st.selectbox("Choose a model",
            (
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "K Nearest Neighbors",
                "SVC",
            ))
            self.model_generator.model_generator(model_name)
            
        return self.model_generator, model_name 

            



