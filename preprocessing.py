import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Initialize session state variables
# st.session_state.x = None
# st.session_state.y = None
st.session_state.scalar = None
st.session_state.x_train = None
st.session_state.y_train = None
st.session_state.x_test = None
st.session_state.y_test = None

def preprocess_dataset():
    # Display the original dataset
    st.write("## Original Dataset:")
    st.dataframe(st.session_state.dataset)

    # Step 1: Drop Columns
    st.write("### Step 1: Drop Columns")
    columns_to_drop = st.multiselect("Select columns to drop", st.session_state.dataset.columns)
    if st.button("Drop Columns"):
        st.session_state.dataset = st.session_state.dataset.drop(columns=columns_to_drop, axis=1)
        st.write("### Result after dropping columns:")
        st.dataframe(st.session_state.dataset)

    # Step 2: Impute Values
    st.write("### Step 2: Impute Values")
    impute_strategy = st.selectbox("Select imputation strategy", ["mean", "median", "most_frequent"])
    if st.button("Impute Values"):
        if st.session_state.dataset is not None:
            # Identify columns with missing values
            columns_with_missing_values = st.session_state.dataset.columns[st.session_state.dataset.isnull().any()].tolist()

            if not columns_with_missing_values:
                st.write("No missing values to impute.")
            else:
                # Initialize the SimpleImputer
                imputer = SimpleImputer(strategy=impute_strategy)

                # Impute missing values only in numerical columns
                st.session_state.dataset[columns_with_missing_values] = imputer.fit_transform(st.session_state.dataset[columns_with_missing_values])

                # Display the result after imputing values
                st.write(f"### Result after imputing values using {impute_strategy} strategy:")
                st.dataframe(st.session_state.dataset)


    # Step 3: Choose Target Variable
    # st.write("### Step 3: Choose Target Variable")
    # target_options = ["Next Day's Closing Price", "Next Week's Closing Price", "Next Month's Closing Price",
    #               "Price Change", "Percentage Price Change"]
    # target_variable = st.selectbox("Select target variable", target_options)

    # if target_variable == "Next Week's Closing Price":
    #     st.session_state.dataset['target'] = st.session_state.dataset['Close'].shift(-5)  # Assuming 5 trading days in a week
    #     st.session_state.x, st.session_state.y = st.session_state.dataset.drop(columns=['Close', 'target']), st.session_state.dataset['target']

    # elif target_variable == "Next Month's Closing Price":
    #     st.session_state.dataset['target'] = st.session_state.dataset['Close'].shift(-20)  # Assuming 20 trading days in a month
    #     st.session_state.x, st.session_state.y = st.session_state.dataset.drop(columns=['Close', 'target']), st.session_state.dataset['target']

    # elif target_variable == "Price Change":
    #     st.session_state.dataset['target'] = st.session_state.dataset['Close'].shift(-1)
    #     st.session_state.dataset['price_change'] = st.session_state.dataset['Close'] - st.session_state.dataset['target']
    #     st.session_state.x, st.session_state.y = st.session_state.dataset.drop(columns=['Close', 'target', 'price_change']), st.session_state.dataset['price_change']

    # elif target_variable == "Percentage Price Change":
    #     st.session_state.dataset['target'] = st.session_state.dataset['Close'].shift(-1)
    #     st.session_state.dataset['percentage_price_change'] = ((st.session_state.dataset['Close'] - st.session_state.dataset['target']) / st.session_state.dataset['target']) * 100
    #     st.session_state.x, st.session_state.y = st.session_state.dataset.drop(columns=['Close', 'target', 'percentage_price_change']), st.session_state.dataset['percentage_price_change']
    
    
    # Step 4: Train-Test Split
    st.write("### Step 4: Train-Test Split")
    test_size = st.slider("Select test set size (%)", 10, 50, 20)
    if st.button("Split Data"):
        x = st.session_state.dataset.drop("Close", axis=1)
        y = st.session_state.dataset["Close"]
        # Calculate the index to split the data
        split_index = int(len(x) * (1 - test_size/100))

        # Split the data
        st.session_state.x_train, st.session_state.x_test = x.iloc[:split_index, :], x.iloc[split_index:, :]
        st.session_state.y_train, st.session_state.y_test = y.iloc[:split_index], y.iloc[split_index:]
        st.write(f"### Result after train-test split (Test set size: {test_size}%):")
        st.write("Training Data:")
        st.dataframe(st.session_state.x_train.join(st.session_state.y_train))
        st.write("Testing Data:")
        st.dataframe(st.session_state.x_test.join(st.session_state.y_test))

    # Step 5: Min-Max Normalization or Standardization
    st.write("### Step 5: Min-Max Normalization or Standardization")
    normalize_method = st.selectbox("Select normalization method", ["Min-Max Normalization", "Standardization"])
    if st.button("Normalize/Standardize"):
        if normalize_method == "Min-Max Normalization":
            st.session_state.scalar = MinMaxScaler()
        else:
            st.session_state.scalar = StandardScaler()

        x_train_normalized = pd.DataFrame(st.session_state.scalar.fit_transform(st.session_state.x_train), columns=st.session_state.x_train.columns)
        x_test_normalized = pd.DataFrame(st.session_state.scalar.transform(st.session_state.x_test), columns=st.session_state.x_test.columns)

        st.write(f"### Result after {normalize_method} (Training Data):")
        st.dataframe(x_train_normalized)
        st.write(f"### Result after {normalize_method} (Testing Data):")
        st.dataframe(x_test_normalized)
