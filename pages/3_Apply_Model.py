import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import predict_model as c_predict_model, load_model as c_load_model
from pycaret.regression import predict_model as r_predict_model, load_model as r_load_model
import sys

st.set_page_config(page_title="Apply Model", page_icon='rocket')

try:
    metrics = st.session_state.metrics
except:
    st.warning("Did you forget to train a model?")
    sys.exit(1)


# take out later
# st.session_state.type = 'classification'
st.info("Apply Trainied Model To Predict Based On New Data")
st.markdown("---")
metrics = st.session_state.metrics
# st.write(metrics)
score = metrics.head(1)
# st.write(f"{score['Accuracy'][10]:.0%}")
col1, col2, col3, col4, col5 = st.columns(5)
st.markdown("---")
if st.session_state.type == 'regression':

    final_model = st.session_state.model
    with col3:
        st.metric(label="Model Performance Score ",
                  value=f"{score['R2'][0]: .2%}")
else:
    final_model = st.session_state.model
    with col3:
        st.metric(label="Model Performance Score",
                  value=f"{score['Accuracy'][0] : .2%}")

inference_method = st.selectbox(
    'Method', ['Single Prediction', 'Batch Prediction'])

df = st.session_state.data

if inference_method == "Single Prediction":
    # Create a dictionary to store user input for each column
    # .tolist().remove(st.session_state.target)  # .pop(st.session_state.target)
    input_columns = df.columns
    # st.write(input_columns)
    # print(df.columns.tolist())
    user_input = dict.fromkeys(input_columns)
    del user_input[st.session_state.target]
    # user_input.pop(st.session_state.target)
    # st.write(user_input)
    # Form for capturing new data from users - single inference

    form = st.form("new_data")

    with form:
        # Allow users to enter data based on the columns in the training data
        for key in user_input:

            if df[key].nunique() <= 10 and df[key].dtype == 'object':
                values = df[key].unique()
                values = np.insert(values, 0, '')
                user_input[key] = form.selectbox(
                    label=key, options=values, index=0,)

            else:
                input_value = st.text_input(f"Enter value for {key}: ")
                # Store the user input for the selected option
                user_input[key] = input_value

        submit_button = st.form_submit_button(label="Submit")
    # When the user clicks the submit button
    if submit_button:
        if not input_value:
            st.error('🚨 Enter some data !! 🚨')
        else:
            st.success('Submitted successfully !')
            # Display the user input for each column
            new_df = pd.DataFrame(user_input, index=[0]).convert_dtypes()
            st.dataframe(new_df)

            # Convert the data types to match the types of training dataframe
            for col in new_df.columns:
                if df[col].dtype != new_df[col].dtype:
                    new_df[col] = new_df[col].astype(df[col].dtype)

            # st.dataframe(new_df.dtypes)
            # print(st.session_state.type)
            if st.session_state.type == 'regression':
                # pass data to predict function
                single_prediction = r_predict_model(final_model, data=new_df)
                # return predicted value with a narrative
                # st.write(single_prediction['prediction_label'][0])
                st.success(
                    f"The predicted value for {st.session_state.target} is   :  {single_prediction['prediction_label'][0] }")
            else:
                # pass data to predict function
                single_prediction = c_predict_model(final_model, data=new_df)
                # return predicted value with a narrative
                st.success(
                    f"The predicted value for {st.session_state.target} is   :  {single_prediction['prediction_label'][0] }")
else:
    # Upload file for prediction - batch inference
    # if st.button('Upload file'):
    new_file = st.file_uploader("Upload Your New Dataset")
    if new_file:
        new_df = pd.read_csv(new_file, index_col=None)
    st.write("New DataFrame:")
    st.dataframe(new_df.head(100))
    
    # Debugging: Check data types
    # st.write("Data types in new DataFrame (new_df) before conversion:")
    # st.write(new_df.dtypes)

    # Handle boolean and object columns
    new_df_bool_col_names = new_df.select_dtypes(include=['bool', 'object']).columns
    new_df[new_df_bool_col_names] = new_df[new_df_bool_col_names].ffill()

    # Convert the data types to match the types of the training dataframe
    for col in new_df.columns:
        if df[col].dtype != new_df[col].dtype:
            try:
                if df[col].dtype == 'int64' or df[col].dtype == 'int32':
                    new_df[col] = new_df[col].replace([np.inf, -np.inf], np.nan)
                    new_df[col] = new_df[col].fillna(0).astype(int)
                elif df[col].dtype == 'float64' or df[col].dtype == 'float32':
                    new_df[col] = new_df[col].replace([np.inf, -np.inf], np.nan)
                    new_df[col] = new_df[col].fillna(0).astype(float)
                elif df[col].dtype == 'object':
                    new_df[col] = new_df[col].fillna('')
            except Exception as e:
                st.error(f"Error converting column {col}: {e}")

    # Final check before prediction
    st.write("Processed new DataFrame:")
    st.dataframe(new_df.head(100))

    if st.button('Generate Predictions'):
        if not new_file:
            st.error('🚨 Upload some data !! 🚨')
        else:
            target_column = st.session_state.target  # Get the target column name

            try:
                if st.session_state.type == 'regression':
                    pred = r_predict_model(final_model, data=new_df)
                else:
                    pred = c_predict_model(final_model, data=new_df)

                # Debugging: Check the columns in the prediction DataFrame
                # st.write("Prediction DataFrame Columns:")
                # st.write(pred.columns)

                # Check for duplicate columns and handle them
                if target_column in pred.columns:
                    # st.warning(f"Duplicate column found: {target_column}. It will be removed.")
                    pred.drop(columns=[target_column], inplace=True)

                # Rename 'prediction_label' to the target column name if it exists
                if 'prediction_label' in pred.columns:
                    pred.rename(columns={'prediction_label': target_column}, inplace=True)

                st.write(f"Predicted Values for {target_column} :")
                st.dataframe(pred.head(100))
                
                # Save predictions
                pred.to_csv('predictions.csv', index=False)
                with open('predictions.csv', 'rb') as f:
                    st.download_button('Download Predictions', f, file_name="predictions.csv")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
