import streamlit as st
import os
import paths
import base64


st.set_page_config("Paper", layout="wide")


def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" ' \
                  F'width="1700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
  

st.markdown("# WIP Research Paper")
displayPDF(os.path.join(paths.ROOT, "paper", "paper.pdf"))
