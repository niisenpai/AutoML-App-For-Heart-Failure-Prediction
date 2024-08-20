import streamlit as st
import pandas as pd
import os


st.set_page_config(
    page_title="Home",
    page_icon=":house:"
)

left_co, cent_co, last_co = st.columns(3)

with cent_co:
    st.image("https://www.4th-ir.com/static/media/logo.ad9a4e1e.png", width=250)
st.markdown("---")
st.title("Automated Machine Learning App")

st.info("An application to help you explore and apply predictive analytics to your data!")
col1, col2, col3 = st.columns(3)
with col2:
    st.image("https://www.4th-ir.com/static/media/product.88e0970f.gif", width=400)
