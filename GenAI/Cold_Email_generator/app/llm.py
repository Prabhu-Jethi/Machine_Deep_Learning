import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from scraper import Scrape
from database import PortfolioDB


class Chain():
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-120b",
            temperature=0
        )

    def extract_jobs(self, scraped_text):
        prompt_extract = PromptTemplate.from_template(
            """### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the 
            following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):   
            """
        )

        chain_extract = prompt_extract | self.llm
        result = chain_extract.invoke(input={"page_data": scraped_text})
        try:
            json_parser = JsonOutputParser()
            result = json_parser.parse(result.content)
        except OutputParserException as e:
            raise OutputParserException(f"Unable to parse {result} {e}")
        return result if isinstance(result, list) else [result]
    

    def cold_email(self, job, link):
        prompt_email = PromptTemplate.from_template(

            """### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Rohan, a business development executive at Infosys. Infosys is a global leader in next-generation 
            digital services and consulting, helping clients in more than 56 countries navigate their digital transformation. 
            With over four decades of experience in managing the systems and workings of global enterprises, Infosys expertly 
            steers clients through their digital journey by enabling enterprises with an AI-powered core, empowering businesses 
            with agile digital at scale, and driving continuous improvement with always-on learning through the transfer of 
            digital skills, expertise, and ideas from its innovation ecosystem.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Infosys 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Infosys's portfolio: {link_list}
            Remember you are Rohan, BDE at Infosys. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):"""
        )

        chain_email = prompt_email | self.llm
        result = chain_email.invoke({'job_description': str(job), 'link_list': link})
        return result



if __name__ == "__main__":

    chain = Chain()

    scraper = Scrape()
    page_content = scraper.scrape_page_content("https://careers.nike.com/software-engineer-iii-itc/job/R-49969")

    jobs = chain.extract_jobs(page_content)

    db = PortfolioDB(file_path=r"D:\Python\MachineLearning\EVERYDAY_ML_DL\GenAI\Cold_Email_generator\data\my_portfolio.csv")
    db.load_portfolio()

    for job in jobs:
        skills = job.get("skills", [])
        links = db.query_links(skills)
        email = chain.cold_email(job, links)
        print(email.content)
