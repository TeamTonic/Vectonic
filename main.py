# main.py 

from pathlib import Path

from llama_index.llms.together import TogetherLLM
import os
from src.dataloader import DataProcessor , DocumentLoader
from vectara_cli.core import VectaraClient
from vectara_cli.span_marker.noncommercial.nerdspan import Span
from vectara_cli.span_marker.commercial.enterprise import EnterpriseSpan
import nest_asyncio

nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

import os
from src.dataloader import DataProcessor, DocumentLoader
from src.chunking import MarkdownProcessor
from unstructured.partition.md import partition_md


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


if __name__ == "__main__":
    customer_id = 123456789
    api_key = "your_vectara_api_key_here"  
    folder_to_process = './your_data_here'
    markdown_output_folder = './processed_markdown'
    data_loading = DataLoading(folder_path=folder_to_process, text_folder_path=markdown_output_folder)
    markdown_paths = data_loading.process_files()
    chonker = Chonker(markdown_files=markdown_paths)
    md_chunks = chonker.process_markdown_files()
    vectara_indexer = VectaraDataIndexer(customer_id, api_key)
    folder_corpus_id = vectara_indexer.create_corpus("Folder Corpus")
    markdown_corpus_id = vectara_indexer.create_corpus("Markdown Corpus")
    if folder_corpus_id:
        print("Indexing entire folder...")
        vectara_indexer.index_folder(folder_corpus_id, folder_to_process)
    if markdown_corpus_id:
        print("Indexing processed Markdown chunks...")
        vectara_indexer.index_markdown_chunks(markdown_corpus_id, md_chunks)
```

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