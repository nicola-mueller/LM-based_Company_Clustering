import streamlit as st
import PIL.Image as Image

st.title("Team")
placeholder = Image.open("pages/images/placeholder_person.png")
col1, col2 = st.columns([1, 1])
with col1:
    st.image(placeholder)
    st.subheader("Nicola Müller")
    st.write("Bachelor of Science - Data Science and Artificial Intelligence")
with col2:
    st.image(placeholder)
    st.subheader("Robert Leist")
    st.write("Bachelor of Science - Computer Science")
col3, col4 = st.columns([1, 1])
with col3:
    st.image(placeholder)
    st.subheader("Paul Eichler")
    st.write("Computer Science")
with col4:
    st.image(placeholder)
    st.subheader("Tobias Recktenwald")
    st.write("Bachelor of Science - Economics")
col5, _ = st.columns([1, 1])
with col5:
    st.image(placeholder)
    st.subheader("Hameed")
    st.write("Bachelor of Science - Economics")
