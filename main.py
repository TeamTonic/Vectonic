# main.py 

import json
from pathlib import Path

from llama_index.llms.together import TogetherLLM
import os
from src.dataloader import DataProcessor , DocumentLoader
from vectara_cli.core import VectaraClient, QueryRequest, QueryResponse
from vectara_cli.rebel_span.noncommercial.nerdspan import Span
from vectara_cli.rebel_span.commercial.enterprise import EnterpriseSpan
import nest_asyncio
import requests
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import unstructured
import os
from src.dataloader import DataProcessor, DocumentLoader
from src.chunking import MarkdownProcessor
from unstructured.partition.md import partition_md as partition_md
from typing import List, Dict, Optional

nest_asyncio.apply()

class DataLoading:
    def __init__(self, folder_path: str, text_folder_path: str):
        self.folder_path = folder_path
        self.text_folder_path = text_folder_path
        os.makedirs(self.text_folder_path, exist_ok=True)

    def process_files(self) -> List[str]:
        """
        Method to process each file in the specified directory, extract Markdown content, and save it to files.
        """
        markdown_paths = []
        print(f"Processing files in folder: {self.folder_path}")
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    reader = DataProcessor.choose_reader(file_path)
                    if not reader:
                        continue

                    documents = reader.load_data(file_path)
                    for doc in documents:
                        text = doc['text']  # Assuming each document has a 'text' key
                        markdown_file_path = os.path.join(self.text_folder_path, f"{Path(file).stem}.md")
                        with open(markdown_file_path, 'w') as md_file:
                            md_file.write(text)
                        markdown_paths.append(markdown_file_path)
                        print(f"Markdown saved to {markdown_file_path}")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
        return markdown_paths

class Chonker:
    def __init__(self, markdown_files: List[str]):
        self.markdown_files = markdown_files

    def process_markdown_files(self) -> Dict[str, List]:
        """
        Chunks markdown files to extract chunks (elements) using partition_md function.
        """
        md_structure = {}
        for md_file in self.markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                elements = partition_md(text=markdown_content)  # adapted to use text directly
                md_structure[md_file] = elements
                print(f"Processed {md_file}: {len(elements)} elements found")
            except Exception as e:
                print(f"Error processing {md_file}: {str(e)}")
        return md_structure

class VectaraDataIndexer:
    def __init__(self, customer_id: int, api_key: str):
        self.vectara_client = VectaraClient(customer_id, api_key)

    def create_corpus(self, corpus_name: str) -> int:
        corpus_data = {
            "name": corpus_name,
            "description": "A corpus for " + corpus_name,
            "metadata": json.dumps({
                "category": "Vectonic",
            })
        }
        response = self.vectara_client.create_corpus(corpus_data)
        if response.get('status', {}).get('code') == 'OK':
            return response['corpusId']
        else:
            print("Failed to create corpus", response)
            return None

    def index_folder(self, corpus_id: int, folder_path: str):
        results = self.vectara_client.index_documents_from_folder(corpus_id, folder_path)
        return results

    def index_markdown_chunks(self, corpus_id: int, markdown_chunks: Dict[str, List]):
        for filepath, sections in markdown_chunks.items():
            for index, section in enumerate(sections):
                title = f"{Path(filepath).stem}-section-{index}"
                document_id = f"{Path(filepath).stem}-{index}"
                response, status = self.vectara_client.index_document(
                    corpus_id, document_id, title, {"section_number": index}, section
                )
                print(f"Indexed section '{title}' status: {status}")

    def index_markdown_chunks_with_entities(self, corpus_id: int, markdown_chunks: Dict[str, List]):
        self.span_processor = Span(vectara_client=self.vectara_client, text="", model_name="fewnerdsuperfine", model_type="span_marker")
        for filepath, sections in markdown_chunks.items():
            for index, section in enumerate(sections):
                title = f"{Path(filepath).stem}-section-{index}"
                document_id = f"{Path(filepath).stem}-{index}"

                # Set text for NER processing
                self.span_processor.text = section
                output_str, entities = self.span_processor.analyze_text()

                enriched_text = output_str + "\n" + section
                metadata = json.dumps({ent['label']: ent['span'] for ent in entities})

                response, status = self.vectara_client.index_document(
                    corpus_id, document_id, title, {}, enriched_text, metadata_json=metadata
                )
                print(f"Indexed enriched section '{title}' status: {status}")

class Retriever:
    def __init__(self, client: VectaraClient):
        self.client = client

    def retrieve_information(self, query: str, corpus_id: int, num_results: int = 10, 
                             context_config: Optional[dict] = None, summary_config: Optional[dict] = None) -> List[QueryResponse]:
        if not context_config:
            context_config = {}
        if not summary_config:
            summary_config = {}
        
        response = self.client.advanced_query(query, num_results, corpus_id, context_config, summary_config)
        if 'error' in response:
            print(response['error'])
            return []
        context = self.client._parse_query_response(response)
        return context

    def prompt_formatting(self, systemprompt : str, context: str, query: str) -> str:
        """
        Formats the prompt to include context and question in a structured manner.
        
        Args:
            systemprompt (str): System Prompt
            context (str): Contextual information or background details.
            query (str): The actual question for the model.

        Returns:
            str: Formatted and JSON escaped string that is safe to be used as API payloads.
        """
        formatted_prompt = f"System Message:{systemprompt}\n\nContext:\n{context}\n\nQuestion:\n{query}"
        # Return a JSON-escaped string
        return json.dumps(formatted_prompt)

    def query_together_llm(self, context: str, query: str, model: str, tokens_limit: int = 150, temperature: float = 0.7) -> str:
        """
        Sends a query to the Together LLM using the given model and generates a response based on the provided context and query.
        
        Args:
            context (str): Contextual information preceding the query.
            query (str): The user query or prompt to send to the LLM.
            model (str): The model identifier.
            tokens_limit (int): Maximum number of response tokens.
            temperature (float): Controls the randomness of the generation.

        Returns:
            str: The assistant's response as a string.
        """
        prompt = self.prompt_formatting(context, query)
        llm = TogetherLLM(model=model, max_tokens=tokens_limit, temperature=temperature)
        
        # Prepare and send the request
        response = llm.complete(prompt)
        
        # Extract and return the response
        if response['choices']:
            return response['choices'][0]['message']['content']
        else:
            return "No response generated."
        
    def use_together_api(self, completion_context: str, model_info: dict):
        """
        Sends a completion request to the Together API based on provided model information and context.
        """
        headers = {
            "Authorization": "Bearer YOUR_BEARER_TOKEN"  # You need to replace this with your actual bearer token
        }
        
        data = {
            "model": model_info['model_string'],
            "prompt": completion_context,
            "max_tokens": model_info['max_tokens'],
            "temperature": model_info['temperature'],
            "top_p": model_info['top_p'] if 'top_p' in model_info else 1,
            "top_k": model_info['top_k'] if 'top_k' in model_info else 40,
            "repetition_penalty": model_info['repetition_penalty'] if 'repetition_penalty' in model_info else 1,
        }
        
        response = requests.post('https://api.together.xyz/v1/completions', json=data, headers=headers)
        return response.json()

    def process_user_questions(client: VectaraClient, questions: List[str], corpus_id: int, model_infos: Dict[str, dict]):
        retriever = Retriever(client)
        
        for question in questions:
            print(f"\nProcessing question: {question}")
            results = retriever.retrieve_information(question, corpus_id)
            
            # Assuming you want to use the first result's context for simplicity:
            if results:
                context = results[0].extracted_text
                # Iterate over available models and fetch responses
                for model_name, model_info in model_infos.items():
                    print(f"Using model: {model_name}")
                    response = retriever.use_together_api(context + '\n' + question, model_info)
                    print(f"Response from {model_name}: {response.get('choices')[0]['text'] if 'choices' in response else 'No response'}")

if __name__ == "__main__":
    customer_id = 123456  # Replace with your customer ID
    api_key = 'your_vectara_api_key'  # Replace with your API key

    folder_to_process = './your_data_here'
    markdown_output_folder = './processed_markdown'

    data_loading = DataLoading(folder_path=folder_to_process, text_folder_path=markdown_output_folder)
    markdown_paths = data_loading.process_files()

    chonker = Chonker(markdown_files=markdown_paths)
    md_chunks = chonker.process_markdown_files()


    vectara_indexer = VectaraDataIndexer(customer_id, api_key)
    folder_corpus_id = vectara_indexer.create_corpus("Folder Corpus")
    markdown_corpus_id = vectara_indexer.create_corpus("Markdown Corpus")
    enriched_corpus_id = vectara_indexer.create_corpus("Enriched Markdown Corpus")
    if folder_corpus_id:
        print("Indexing entire folder...")
        vectara_indexer.index_folder(folder_corpus_id, folder_to_process)
    if markdown_corpus_id:
        print("Indexing processed Markdown chunks...")
        vectara_indexer.index_markdown_chunks(markdown_corpus_id, md_chunks)
    if enriched_corpus_id:
        print("Processing and indexing enriched Markdown chunks...")
        vectara_indexer.index_markdown_chunks_with_entities(enriched_corpus_id, md_chunks)

    vectara_indexer = VectaraDataIndexer(customer_id, api_key)
    folder_corpus_id = vectara_indexer.create_corpus("Folder Corpus")
    markdown_corpus_id = vectara_indexer.create_corpus("Markdown Corpus")
    if folder_corpus_id:
        print("Indexing entire folder...")
        vectara_indexer.index_folder(folder_corpus_id, folder_to_process)
    if markdown_corpus_id:
        print("Indexing processed Markdown chunks...")
        vectara_indexer.index_markdown_chunks(markdown_corpus_id, md_chunks)
    # Define model information based on Together API details
    model_infos = {
        "Qwen": {
            "model_string": "Qwen/Qwen1.5-72B",
            "max_tokens": 4096,
            "temperature": 0.7
        },
        "Meta-Llama": {
            "model_string": "meta-llama/Meta-Llama-3-70B",
            "max_tokens": 8192,
            "temperature": 1,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.0
        }
    }
    
    # Sample questions - Place where user questions array is expected
    user_questions = [
        "What are the current trends in AI?",
        "How is climate change impacting ocean levels?",
        "Discuss the advancements in renewable energy technologies."
    ]
    
    # Assume a predefined corpus ID (replace with actual corpus ID you want to use)
    corpus_id = 87654321
    
    # Process user questions
    Retriever.process_user_questions(vectara_indexer, user_questions, corpus_id, model_infos)


# class DataLoading:
#     def __init__(self, folder_path: str):
#         self.folder_path = folder_path

#     def process_files(self):
#         """
#         Method to process each file in the specified directory.
#         """
#         # Validate if the folder exists
#         if not os.path.exists(self.folder_path):
#             print(f"Error: The folder {self.folder_path} does not exist.")
#             return

#         print(f"Processing files in folder: {self.folder_path}")
#         for root, dirs, files in os.walk(self.folder_path):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 try:
#                     # Use DataProcessor's static method to determine appropriate reader
#                     reader = DataProcessor.choose_reader(file_path)
#                     if reader:
#                         print(f"Processing file: {file} with {type(reader).__name__}")
#                         documents = reader.load_data(file_path)  # This assumes a `load_data` method in each reader
#                         print(f"Loaded documents from '{file}': {documents}")
#                     else:
#                         print(f"No appropriate reader found for {file}. Skipping.")
#                 except Exception as e:
#                     print(f"Error processing {file}: {str(e)}")


# if __name__ == "__main__":
#     # Instance of DataLoading, pointing to the desired directory
#     data_loader = DataLoading(folder_path='./your_data_here')
#     data_loader.process_files()
#     #Continue