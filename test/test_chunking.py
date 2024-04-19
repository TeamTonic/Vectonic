import pytest
from src.dataloader import DataProcessor, DocumentLoader
# # from src.chunking import  get_ff
# from src.chunking import DataLoaderforChunking
# from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from vectara_cli.rebel_span.noncommercial.nerdspan import Span
from vectara_cli.core import VectaraClient
from vectara_cli.utils.config_manager import ConfigManager
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

@pytest.fixture
def doc_list():
    # Set up real dependency here, such as connecting to a database or API
    document_loader = DocumentLoader()
    list_of_docs = document_loader.load_documents_from_folder()
    return list_of_docs

def test_chunking(doc_list):
    # pass
    dd = doc_list
    
    get_text:List[str] = [i.text for i in dd]
    vectara_client = VectaraClient(
        api_key=os.getenv("VECTARA_USER_ID"),
        customer_id=os.getenv("VECTARA_API_KEY"),
    )
    
    vectara_client.create_corpus(
        {"corpus_id":2}
    )
    
    vectara_client.index_text(
        1,
        8666,
        "text",
        "",
        "{}",
        None,
        30
    )
    data = []
    # for i in get_text:
    temp = partition_text(
        text=get_text[0]
    )
        
    data.extend(temp)
    
    
        
    x = 0
    # my_class_instance = doc_list
    
#     # sample_location = "Vectonic/add_your_files_here/One Attention Head Is All You Need for Sorting Fixed-Length Lists.pdf"
    
#     # ff = partition_pdf(
#     #     filename=sample_location
#     # )
    
    # get_text = [i.text for i in my_class_instance]
    
    
    
#     # chunker.chunk_elements(get_text[0])
#     x= 0 
    
# def test_one_one():
#     assert 1+1 ==2

def test_one_one():
    assert 1+1 ==2
    