import streamlit as st
import pandas as pd

def show_dataset(dataset):
    # Display basic information about the DataFrame
    st.write("## Dataset Specifications:")
    st.write("Number of Rows:", dataset.shape[0])
    st.write("Number of Columns:", dataset.shape[1])

    # Display data types of columns
    st.write("\n**Data Types:**")
    st.table(dataset.dtypes)

    # Display null values in each column
    st.write("\n**Null Values in Each Column:**")
    null_values = dataset.isnull().sum()
    st.table(null_values[null_values > 0].reset_index().rename(columns={0: 'Null Count'}))

    # Display summary statistics
    st.write("\n## Summary Statistics:")
    st.table(dataset.describe())

    # Display a sample of the dataset
    st.write("## Sample of the Dataset:")
    st.dataframe(dataset.head())
    


