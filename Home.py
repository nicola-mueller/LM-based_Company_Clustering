import streamlit as st

text, images = st.columns([3, 1])

with text:
    st.markdown("# Team17 AI Analytics")

    st.markdown("At **Team17 AI Analytics**, we specialize in providing cutting-edge AI-powered solutions for"
                " comprehensive company clustering and market segmentation. Leveraging advanced techniques such as "
                "kernel PCA (Principal Component Analysis) and t-SNE (t-Distributed Stochastic Neighbor Embedding), "
                "we offer businesses in Saarland an unparalleled insight into their industry landscape.")
        
    st.markdown("## Our Approach:")

    st.markdown("By harnessing the power of state-of-the-art transformers, we embed company descriptions into "
                "high-dimensional representations, enabling us to identify hidden patterns and similarities. "
                "Through our innovative clustering methodology, we analyze these embeddings and provide businesses with"
                " a holistic view of companies sharing similar characteristics based on their descriptions.")

    st.markdown("[Read up on our approach in our WIP paper!](Paper)")
        
    st.markdown("## Unveiling Opportunities:")
        
    st.markdown("With our clustering analysis, Saarland companies gain a competitive edge by understanding their "
                "positioning "
                "within the market, identifying potential collaborators or competitors, and targeting marketing efforts"
                " more effectively. By exploring the relationships between companies and their descriptions, "
                "our solutions "
                "facilitate informed decision-making and strategic planning.")

    st.markdown("[Analyze your own company now](Analysis) or "
                "[check out our clustering for companies in Saarland](Clustering).")
        
    st.markdown("## Expert Team:")
        
    st.markdown("Our multidisciplinary team consists of five highly skilled professionals, blending expertise "
                "from diverse "
                "backgrounds. One member holds a Bachelor of Science in Data Science and AI, possessing a deep"
                " understanding of cutting-edge AI techniques and models. One member brings a Bachelor of "
                "Science in Computer Science, ensuring robust implementation and system optimization. "
                "Finally, our two team members with Bachelor of Science degrees in Economics bring valuable insights "
                "into market dynamics and business strategy.")

    st.markdown("[Get to know the team!](Team)")
        
    st.markdown("At **Team17 AI Analytics**, we are passionate about leveraging the power of AI and data "
                "science to help "
                "businesses thrive in the ever-evolving landscape of Saarland's industries. We are dedicated to "
                "delivering tailored solutions that empower companies to make informed decisions, unlock new "
                "opportunities, and drive sustainable growth.")





