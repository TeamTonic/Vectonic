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

class DataLoading:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def process_files(self):
        """
        Method to process each file in the specified directory.
        """
        # Validate if the folder exists
        if not os.path.exists(self.folder_path):
            print(f"Error: The folder {self.folder_path} does not exist.")
            return

        print(f"Processing files in folder: {self.folder_path}")
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    reader = DataProcessor.choose_reader(file_path)
                    if reader:
                        print(f"Processing file: {file} with {type(reader).__name__}")
                        documents = reader.load_data(file_path)  # This assumes a `load_data` method in each reader
                        print(f"Loaded documents from '{file}': {documents}")
                    else:
                        print(f"No appropriate reader found for {file}. Skipping.")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Instance of DataLoading, pointing to the desired directory
    data_loader = DataLoading(folder_path='./add_your_files_here')
    data_loader.process_files()