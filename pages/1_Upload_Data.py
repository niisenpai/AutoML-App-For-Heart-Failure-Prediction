import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Data Upload", page_icon="⌨️")

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
    st.session_state.data = df


# if choice == "Upload Data":
st.title("Upload Your Dataset")
file = st.file_uploader("Upload Your Dataset")
if file:
    df = pd.read_csv(file, index_col=None)  # .convert_dtypes()
    df.to_csv('dataset.csv', index=None)

    st.dataframe(df.head(100))
    # st.write(df.dtypes)

    # st.write(bool_cols.columns)
    st.session_state.data = df

# try:
#    st.dataframe(st.session_state.data.head(100))
# except:
#    pass
