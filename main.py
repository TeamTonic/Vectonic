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

parser = LlamaParse(
    api_key="llx-...",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
) # can be replaced by Unstructured!!

# plaintext = parser(DocumentLoader("./add_your_files_here"))
# vectara_client = VectaraClient(api keys)
# Add Process and Upload Logic Here
# chunks = UnstructuredReader.chunk_text(plaintext)
# for chunk in chunks :
# spantext , entities = Span().analyse_text
# prepend spantext to chunk , Span().convert_entities_to_metadata
# VectaraClient().upload_document(corpus_id, enhanced chunk, metadata)

## Query
# VectaraClient().query

## FormatPrompt("# Context {Context} \n \n # Query {query} \n\n you are a helpful assistant , answer the question based on the above.")

## Todo : Model Mapper for Model Selection
llm = TogetherLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key="your_api_key"
)
resp = llm.complete("Who is Paul Graham?")
    print(resp)