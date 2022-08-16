import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# LAYOUT
# https://docs.streamlit.io/en/stable/api.html#lay-out-your-app
# https://discuss.streamlit.io/t/amend-streamlits-default-colourway-like-on-the-spacy-streamlit-app/4184/10

st.set_page_config(
    page_title="tigtec",
    layout="wide",
    page_icon="streamlit_app/assets/favicon-eki.png",

)
    
with open("streamlit_app/navbar-bootstrap.html","r") as navbar:
    st.markdown(navbar.read(),unsafe_allow_html=True)


# From https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
with open("streamlit_app/style.css") as f:
    st.markdown("""<link href='http://fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900italic,900' rel='stylesheet' type='text/css'>""", unsafe_allow_html=True)
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)



left_column, center_column,right_column = st.beta_columns([1,4,1])
# You can use a column just like st.sidebar:

with left_column:
    st.info("**tigtec** tool using streamlit to showcase the capabilities developed in tigtec")

with right_column:
    st.write("#### Authors\nThis tool has been developed by [Ekimetrics](https://www.ekimetrics.com)")

with center_column:
    st.write("# tigtec")
    st.write("Edit the file ``streamlit_index.py`` to edit your custom prototype")


