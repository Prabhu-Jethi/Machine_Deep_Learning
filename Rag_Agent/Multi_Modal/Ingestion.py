import json
import pytesseract
from typing import List

## Unstructured for document parsing
import unstructured_pytesseract
unstructured_pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

file_path = "../docs/attention-is-all-you-need.pdf"

## 1. partitioning pdf into atomic elements
def partition_document(filepath: str):
    print(f"Partitioning pdf: {filepath} \n")

    if not file_path:
        raise FileNotFoundError(f"There is no such {file_path} exists.")
    
    elements = partition_pdf(
        filename=filepath,
        strategy="hi_res",   ## Accurate processing of method partitioning
        infer_table_structure=True,     ## Keep tables as structured html
        extract_image_block_types=["Image"],     ## Images found in .pdf
        extract_image_block_to_payload=True     ## Store images as base64 to view it
    )

    print(f"\n Extracted: {len(elements)} elements")

    ## Access all the unique atomic elements type like 'Header', 'Figurecaption', 'Image' etc...
    unique_el = set([(type(el)) for el in elements])
    print(f"\n Unique atomic elements: {unique_el}")

    ## Entire context, metadata of an atomic element
    atom_el = elements[31].to_dict()
    print(f"\n{atom_el}")

    ## Context, metadata for image
    images = [el for el in elements if el.category == 'Image']
    print(f"\nFound {len(images)} images")
    img_el = images[0].to_dict()
    print(f"\n{img_el}")

    ## For tables
    tables = [el for el in elements if el.category == 'Table']
    print(f"\nFound {len(tables)} tables")
    tab_el = tables[0].to_dict()
    print(f"\n{tab_el}")
    return elements


## 2. Creating chunks using title based strategy
def create_chunks_by_title(elements):
    print(f"\nCreating Smart Chunks..")

    chunks = chunk_by_title(
        elements,   ## parsed pdf elements from previous steps
        max_characters=3000,    ## Hard Limit for each chunk size
        new_after_n_chars=2400,     ## Try to start new characters after 2400 chars
        combine_text_under_n_chars=500  ## Merge tiny chunks under 500 chars with neighbours
    )

    print(f"Created: {len(chunks)} chunks")

    ## View unique chunk types
    uni_chunk = set([(type(chunk)) for chunk in chunks])
    print(f"\n Unique Chunk type {uni_chunk}")

    ## Single chunk
    single_chk = chunks[4].to_dict()
    print(f"\n{single_chk}")

    ## Original chunk elements
    chunk_el = chunks[11].metadata.orig_elements[-1].to_dict()
    print(f"\n Original chunk elements {chunk_el}")
    return chunks


def separate_chunk_types(chunk):
    """Analyze what types of content are in a chunk"""

    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data


def enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    try:
        # We MUST use a Vision model to process image_urls!
        llm = ChatGroq(model="llama-3.2-11b-vision-preview")

        prompt_text = f"""You are reating a searchable description for document content retrieval.
        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}"""
        
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        prompt_text += """
        YOUR TASK:
        Generate a comprehensive, searchable description that covers:

        1. Key facts, numbers, and data points from text and tables
        2. Main topics and concepts discussed  
        3. Questions this content could answer
        4. Visual content analysis (charts, diagrams, patterns in images)
        5. Alternative search terms users might use

        Make it detailed and searchable - prioritize findability over brevity.

        SEARCHABLE DESCRIPTION:"""
        
        ## Message content startng with text
        message_count = [{"type": "text", "text": prompt_text}]
        ## Add images to message
        for image_base64 in images:
            message_count.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        ## Get Human response
        message = HumanMessage(content=message_count)
        response = llm.invoke([message])
        return response.content
    
    except Exception as e:
        print(f"Enhanced Summary Failed! {e}")
        # Fallback to simple summary
        summary = f"{text[:300] if text else ''}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary



def summarized_chunks(chunks):
    ## Process all chunks with Enhanced Summary
    print(f"Processing Chunks with Enhanced Summary")

    langchain_documents = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        curr_chunk = i + 1
        print(f"{curr_chunk}/{total_chunks}")

        ## Analyze chunk content data
        content_data = separate_chunk_types(chunk)
        print(f"Type found: {content_data['types']}")
        print(f"Table_type found: {len(content_data['tables'])}, Image_type found: {len(content_data['images'])}")
        

def main():
    # 1. Get elements from partition_document
    elements = partition_document(filepath=file_path)

    # 2. Get chunks from create_chunks_by_title
    chunks = create_chunks_by_title(elements=elements)
    
    # 3. separate_chunk_types 
    processed_chunks = []
    for chunk in chunks:
        content_data = separate_chunk_types(chunk=chunk)
        processed_chunks.append(content_data)
    print(f"\nSuccessfully processed {len(processed_chunks)} chunks!")

    # 4. Ai enhanced summary (using llm)
    print("\nGenerating AI enhanced summaries for each chunk...")
    summarized_docs = []
    for idx, data in enumerate(processed_chunks):
        print(f"Summarizing chunk {idx + 1}/{len(processed_chunks)}...")
        summary = enhanced_summary(
            text=data.get('text', ''),
            tables=data.get('tables', []),
            images=data.get('images', [])
        )
        # Store the summary along with the original raw data
        data['summary'] = summary
        summarized_docs.append(data)
        
    print("\nAll chunks summarized successfully!")

if __name__ == "__main__":
    main()
