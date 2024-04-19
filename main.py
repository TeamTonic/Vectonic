# main.py 

from pathlib import Path
from llama_index.core import SimpleDirectoryReader # https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader/
from llama_index.readers.file import UnstructuredReader # https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/unstructured

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

class DataLoading:
    def __init__(self, folder_path: str, text_folder_path: str):
        self.folder_path = folder_path
        self.text_folder_path = text_folder_path
        os.makedirs(self.text_folder_path, exist_ok=True)

    def process_files(self) -> List[str]:
        """
        Method to process each file in the specified directory, store markdown, and return paths.
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
                        text = doc['text']  # Assuming each document has a 'text' key with markdown content
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

    def process_markdown_files(self) -> List[Tuple[str, List]]:
        """
        Processes markdown files to extract structures using partition_md function.
        """
        md_structure = []
        for md_file in self.markdown_files:
            try:
                reader = UnstructuredReader(md_file)
                content = reader.load_data()
                elements = partition_md(content)
                md_structure.append((md_file, elements))
                print(f"Processed {md_file}: {elements}")
            except Exception as e:
                print(f"Error processing {md_file}: {str(e)}")
        return md_structure


if __name__ == "__main__":
    folder_to_process = './your_data_here'
    markdown_output_folder = './processed_markdown'
    data_loading = DataLoading(folder_path=folder_to_process, text_folder_path=markdown_output_folder)
    markdown_paths = data_loading.process_files()

    chonker = Chonker(markdown_files=markdown_paths)
    md_chunks = chonker.process_markdown_files()
    # Further processing can be done here with `md_chunks`

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


if __name__ == "__main__":
    # Instance of DataLoading, pointing to the desired directory
    data_loader = DataLoading(folder_path='./your_data_here')
    data_loader.process_files()
    #Continue