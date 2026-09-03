import streamlit as st
st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")

from langchain_community.document_loaders import WebBaseLoader
from llm import Chain
from database import PortfolioDB
from utils import clean_text

def create_streamlit_app(llm, database, clean_text):
    st.title("Cold Email Generator")
    url_input = st.text_input("Enter the URL:", value="https://careers.nike.com/software-engineer-iii-itc/job/R-49969")
    submit_button = st.button("Generate Cold Email")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            page_data = clean_text(loader.load().pop().page_content)
            database.load_portfolio()
            jobs = llm.extract_jobs(page_data)
            for job in jobs:
                skills = job.get("skills", [])
                links = database.query_links(skills)
                email = llm.cold_email(job, links)
                st.code(email.content, language="text")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    llm = Chain()
    database = PortfolioDB(file_path=r"D:\Python\MachineLearning\EVERYDAY_ML_DL\GenAI\Cold_Email_generator\data\my_portfolio.csv")
    create_streamlit_app(llm, database, clean_text)