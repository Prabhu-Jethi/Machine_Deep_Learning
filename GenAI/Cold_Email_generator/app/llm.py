import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


class Chain():
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="meta-llama/llama-prompt-guard-2-22m",
            temperature=0
        )

    def extract_jobs(self):
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
        result = chain_extract.invoke(input=["page_data"])
        try:
            json_parser = JsonOutputParser()
            result = json_parser.parse(result.content)
        except OutputParserException as e:
            raise OutputParserException("Unable to parse {result} {e}")
        return result if isinstance(result, list) else [result]
    

if __name__ == "__main__":
    parse = Chain()
    result = parse.extract_jobs()
    print(result)

