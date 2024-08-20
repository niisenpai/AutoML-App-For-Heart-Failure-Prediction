import streamlit as st
import pandas as pd
from pycaret.classification import setup as c_setup, compare_models as c_compare_models, tune_model as c_tune_model, pull as c_pull, finalize_model as c_finalize_model, predict_model as c_predict_model, save_model as c_save_model, load_model as c_load_model, models as c_model
from pycaret.regression import setup as r_setup, compare_models as r_compare_models, tune_model as r_tune_model,  pull as r_pull, finalize_model as r_finalize_model, predict_model as r_predict_model, save_model as r_save_model, load_model as r_load_model, models as r_model
import time
import sys
# import os

st.set_page_config(page_title="Train Model", page_icon="⚙️")

try:
    df = st.session_state.data
except:
    st.warning("Did you forget to upload your dataset?")
    sys.exit(1)

options = [None] + df.columns.tolist()


chosen_target = st.selectbox('Choose Target Column', options)

if chosen_target:
    st.session_state.target = chosen_target

# check boolean columns and fill in missing values with the previous value? This handles pycaret's issue with parsing boolean columns with missing entries
# st.write(df.dtypes)
bool_col_names = df.select_dtypes(include=['bool', 'object']).columns
df[bool_col_names] = df[bool_col_names].ffill()
# st.write(df[bool_col_names].dtypes)
# st.write(df.dtypes)


# Remove target column from list of booleans
# try:
#    if df[chosen_target].dtype == 'bool':
#        bool_cols = df[bool_col_names].drop(columns=chosen_target)
#        # st.write(bool_cols.columns)
# except:
#    pass


if st.button('Start Modelling'):
    # Determine problem type
    if df[chosen_target].nunique() > 10:
        st.session_state.type = 'regression'
        st.write("Problem type: regression")
        start_time = time.time()
        r_setup(df, target=chosen_target  # , categorical_features=bool_cols.columns.tolist()
                )
        # setup_df = r_pull()
        # st.dataframe(setup_df)

        # Train and select best model
        best_model = r_compare_models(
            cross_validation=True)

        end_time = time.time()
        duration = end_time - start_time

        compare_df = r_pull()
        st.success('Model Training Completed!')
        st.info(f"Training time: {duration:.2f} seconds.")
        st.dataframe(compare_df)

        # save model metrics
        compare_df.to_csv('model_perfomance.csv', index=False)
        st.session_state.metrics = compare_df

        # save model
        r_save_model(best_model, 'best_model')
        st.session_state.model = best_model
    else:
        st.write("Problem type: classification")
        st.session_state.type = 'classification'

        start_time = time.time()

        c_setup(df, target=chosen_target,  # categorical_features= # bool_cols.columns.tolist()
                )
        #setup_df = c_pull()
        # st.dataframe(setup_df)

        # Train and select best model
        best_model = c_compare_models(cross_validation=False)
        end_time = time.time()
        duration = end_time - start_time

        compare_df = c_pull()
        st.success('Model Training Completed!')
        st.info(f"Training time: {duration:.2f} seconds.")
        st.dataframe(compare_df)

        # save model metrics
        compare_df.to_csv('model_perfomance.csv', index=False)
        st.session_state.metrics = compare_df

        # save model
        c_save_model(best_model, 'best_model')
        st.session_state.model = best_model

        metrics = st.session_state.metrics
        # st.dataframe(metrics)

    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
