import streamlit as st
from sklearn.svm import  SVR
from sklearn.ensemble import  RandomForestRegressor
from xgboost import  XGBRegressor
from sklearn.metrics import  mean_squared_error

def train_svm(C):
    model = SVR(C=C)
    model.fit(st.session_state.x_train, st.session_state.y_train)
    y_pred = model.predict(st.session_state.x_test)
    mse = mean_squared_error(st.session_state.y_test, y_pred)
    return mse

def train_random_forest(n_estimators, max_depth):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(st.session_state.x_train, st.session_state.y_train)
    y_pred = model.predict(st.session_state.x_test)
    mse = mean_squared_error(st.session_state.y_test, y_pred)
    return mse

def train_xgboost(learning_rate, max_depth):
    model = XGBRegressor(learning_rate=learning_rate, max_depth=max_depth)
    model.fit(st.session_state.x_train, st.session_state.y_train)
    y_pred = model.predict(st.session_state.x_test)
    mse = mean_squared_error(st.session_state.y_test, y_pred)
    return mse

def train_models():
    
    # SVM Hyperparameters
    st.subheader("Support Vector Machine (SVM)")
    svm_C = st.slider("SVM C (Regularization parameter)", 0.1, 10.0, 1.0)
    svm_mse = train_svm(svm_C)
    st.write(f"Mean Squared Error: {svm_mse:.2f}")

    # Random Forest Hyperparameters
    st.subheader("Random Forest")
    rf_n_estimators = st.slider("Number of Estimators", 1, 100, 10)
    rf_max_depth = st.slider("Max Depth", 1, 20, 5)
    rf_mse = train_random_forest(rf_n_estimators, rf_max_depth)
    st.write(f"Mean Squared Error: {rf_mse:.2f}")

    # XGBoost Hyperparameters
    st.subheader("XGBoost")
    xgboost_learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    xgboost_max_depth = st.slider("Max Depth", 1, 20, 3)
    xgboost_mse = train_xgboost(xgboost_learning_rate, xgboost_max_depth)
    st.write(f"Mean Squared Error: {xgboost_mse:.2f}")
