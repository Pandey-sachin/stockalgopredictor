import streamlit as st
import pandas as pd
from introduction import introduction
from dataset import show_dataset
from preprocessing import preprocess_dataset
from training import train_models
class App:
    def __init__(self):
        # Initialize session_state to store dataset
        self.session_state = st.session_state
        if not hasattr(self.session_state, 'dataset'):
            self.session_state.dataset = None

    def introduction_page(self):
        introduction()

    def dataset_page(self):
        st.title("Dataset")
        st.write("Upload a dataset. Explore and analyze the dataset on this page.")
        # Upload a dataset
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            # Read the dataset
            self.session_state.dataset = pd.read_csv(uploaded_file)
            # Display dataset specifications
            show_dataset(self.session_state.dataset)

    def preprocessing_page(self):
        st.title("Preprocessing")
        if self.session_state.dataset is not None:  # Check if the dataset is loaded
            preprocess_dataset()
        else:
            st.warning("Please upload a dataset on the 'Dataset' page first.")

    def training_page(self):
        st.title("Training")
        if self.session_state.dataset is not None:  # Check if the dataset is loaded
            train_models()
        else:
            st.warning("Please upload a dataset on the 'Dataset' page and then perform preprocessing on the 'Preprocessing'page first.")


    def results_page(self):
        st.title("Results")
        st.write("View and analyze the results of your trained model on this page.")

    def final_page(self):
        st.title("Final Thoughts")
        st.write("Wrap up and provide final thoughts on this page.")

    def main(self):
        st.set_page_config(page_title="stockAlgoprediXpert",page_icon="favicon.png",layout="wide",)
        st.sidebar.image("logo.png")
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ("Introduction", "Dataset", "Preprocessing", "Training", "Results", "Final"))
       
        if page == "Introduction":
            self.introduction_page()
        elif page == "Dataset":
            self.dataset_page()
        elif page == "Preprocessing":
            self.preprocessing_page()
        elif page == "Training":
            self.training_page()
        elif page == "Results":
            self.results_page()
        elif page == "Final":
            self.final_page()

if __name__ == "__main__":
    app = App()
    app.main()
