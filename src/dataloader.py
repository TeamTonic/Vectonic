# ./src/dataloader.py

import os
from typing import Any, Optional
import llama_index
from llama_index.readers.file.docs import DocxReader, HWPReader, PDFReader
from llama_index.readers.file.epub import EpubReader
from llama_index.readers.file.flat import FlatReader
from llama_index.readers.file.html import HTMLTagReader
from llama_index.readers.file.image import ImageReader
from llama_index.readers.file.image_caption import ImageCaptionReader
from llama_index.readers.file.image_deplot import ImageTabularChartReader
from llama_index.readers.file.image_vision_llm import ImageVisionLLMReader
from llama_index.readers.file.ipynb import IPYNBReader
from llama_index.readers.file.markdown import MarkdownReader
from llama_index.readers.file.mbox import MboxReader
from llama_index.readers.file.paged_csv import PagedCSVReader
from llama_index.readers.file.pymu_pdf import PyMuPDFReader
from llama_index.readers.file.slides import PptxReader
from llama_index.readers.file.tabular import PandasCSVReader, CSVReader
from llama_index.readers.file.unstructured import UnstructuredReader 
from llama_index.readers.file.xml import XMLReader
from llama_index.readers.file.rtf import RTFReader
from llama_index.readers.file import VideoAudioReader 
from llama_index.readers.file import ImageVisionLLMReader # https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_vision_llm/base.py
# from llama_index.readers.file import SimpleDirectoryReader
# from llama_index.core.node_parser import SentenceSplitter

import llama_parse
from llama_parse import LlamaParse

# from langchain_community_core.documents.base import Document

import os


class DataProcessor:
    def __init__(self, source_file: str, collection_name: str, persist_directory: str):
        self.source_file = source_file
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def load_data_from_source_and_store(self) -> Any:
    # def load_data_from_source_and_store(source: Union[str, dict], collection_name: str, persist_directory: str) -> Any:
        """
        Loads data from various sources and stores the data in ChromaDB.

        :param source: A string representing a file path or a URL, or a dictionary specifying web content to fetch.
        print("Data loaded and stored successfully in ChromaDB.")
        """
        # Determine the file extension
        if isinstance(self.source_file, str):
            ext = os.path.splitext(self.source_file)[-1].lower()
        else:
            raise TypeError("Source must be a string (file path or URL).")

        # Load data using appropriate reader
        if ext == '.csv':
            reader = PandasCSVReader(self.source_file, return_full_document=True)
        elif ext == '.docx':
            reader = DocxReader(self.source_file, return_full_document=True)
        elif ext == '.epub':
            reader = EpubReader(self.source_file, return_full_document=True)
        elif ext == '.html':
            reader = HTMLTagReader(self.source_file, return_full_document=True)
        elif ext == '.hwp':
            reader = HWPReader(self.source_file, return_full_document=True)
        elif ext == '.ipynb':
            reader = IPYNBReader(self.source_file, return_full_document=True)
        elif ext in ['.png', '.jpg', '.jpeg']:
            reader = ImageReader(self.source_file, return_full_document=True)  # Assuming ImageReader can handle common image formats
        elif ext == '.md':
            reader = MarkdownReader(self.source_file, return_full_document=True)
        elif ext == '.mbox':
            reader = MboxReader(self.source_file, return_full_document=True)
        elif ext == '.pdf':
            reader = PDFReader(self.source_file, return_full_document=True)
        elif ext == '.pptx':
            reader = PptxReader(self.source_file, return_full_document=True)
        elif ext == '.rtf':
            reader = RTFReader(self.source_file, return_full_document=True)
        elif ext == '.xml':
            reader = XMLReader(self.source_file, return_full_document=True)
        else:
            raise ValueError(f"Unsupported source type: {self.source_file}")

    def choose_reader(self, file_path: str) -> Optional[object]:
        """Selects the appropriate reader for a given file based on its extension."""
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        readers = {
            '.csv': CSVReader,
            '.docx': DocxReader,
            '.epub': EpubReader,
            '.html': HTMLTagReader,  # Assuming HTMLTagReader is for .html files
            '.hwp': HWPReader,
            '.ipynb': IPYNBReader,
            '.jpg': ImageReader,
            '.jpeg': ImageReader,
            '.png': ImageReader,
            '.bmp': ImageReader,
            '.tiff': ImageReader,
            '.gif': ImageReader,  # Assuming ImageReader can handle .gif
            '.md': MarkdownReader,
            '.mbox': MboxReader,
            '.pdf': PDFReader,
            '.pptx': PptxReader,
            '.rtf': RTFReader,
            '.xml': XMLReader,
            '.txt': FlatReader,  # Assuming FlatReader is for .txt files
            # .csv extension has multiple readers, I'm assuming PagedCSVReader and PandasCSVReader
            # are specific cases that would be handled elsewhere, hence using CSVReader as default
        }
        image_readers = {
            '.jpg': ImageCaptionReader(self.source_file, return_full_document=True),  # or ImageTabularChartReader, ImageVisionLLMReader based on content
            '.jpeg': ImageCaptionReader(self.source_file, return_full_document=True),
            '.png': ImageTabularChartReader(self.source_file, return_full_document=True),
        }
        
        # If the file is an image and has a specialized reader, use that.
        if file_extension in image_readers:
            return image_readers[file_extension]()
        
        reader_class = readers.get(file_extension)
        return reader_class() if reader_class else None

class DocumentLoader:

    @staticmethod
    # def load_documents_from_folder(folder_path: str) -> list[Document]:
    def load_documents_from_folder(folder_path:str="./add_your_files_here"):
        """Loads documents from files within a specified folder"""
        folder_path = "./add_your_files_here"
        documents = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                
                reader = DataProcessor.choose_reader(full_path)

                if reader:
                    print(f"Loading document from '{filename}' with {type(reader).__name__}")
                    
                    try:
                        docs = list(reader.load_data(input_files=[full_path]))
                        documents.extend(docs)
                        
                    except Exception as e:
                        print(f"Failed to load document from '{filename}'. Error: {e}")
        # Convert to langchain format
        documents = [ doc.to_langchain_format()
        for doc in documents
        ]                       
        return documents
