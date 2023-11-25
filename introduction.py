import streamlit as st
from PIL import Image

def introduction():
    st.title("Machine Learning Algorithm Comparison for Stock Price Prediction")
    image = Image.open('stock1.jpg')
    st.image(image, width=800)


    st.write(
        "Welcome to the Machine Learning Algorithm Comparison App! This app aims to compare the performance of various "
        "machine learning algorithms in predicting stock prices. Choosing the right algorithm is crucial for accurate predictions, "
        "and this tool allows you to explore, analyze, and compare their effectiveness."
    )

    st.subheader("Key Objectives:")
    st.markdown(
        "1. **Explore Diverse Algorithms**: Understand the strengths and weaknesses of different machine learning algorithms used "
        "for stock price prediction."
    )
    st.markdown(
        "2. **Evaluate Performance Metrics**: Compare key performance metrics such as Mean Absolute Error (MAE), Root Mean Squared "
        "Error (RMSE), and others to assess the accuracy of predictions."
    )
    st.markdown(
        "3. **Visualize Results**: Visualize the predictions made by each algorithm and observe how well they align with actual stock prices."
    )

    st.write(
        "By navigating through the pages of this app, you'll be able to explore the dataset, perform necessary preprocessing, train "
        "different machine learning models, and analyze the results. Let's dive in and see how different algorithms perform in the "
        "complex world of stock price prediction!"
    )


