import streamlit as st
import paths

text, images = st.columns([3, 1])

with text:
    md_file = open(paths.get_page_location('Home.md'))
    st.markdown(md_file.read())
