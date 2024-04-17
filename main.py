# main.py 

from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import UnstructuredReader
from llama_index.readers.file import VideoAudioReader
from llama_index.readers.file import ImageVisionLLMReader
from llama_index.llms.together import TogetherLLM


dir_reader = SimpleDirectoryReader(
    "./data",
    file_extractor={
        ".pdf": UnstructuredReader(),
        ".html": UnstructuredReader(),
        ".eml": UnstructuredReader(),
    },
)
documents = dir_reader.load_data()


        # set api key in env or in llm
        # import os
        # os.environ["TOGETHER_API_KEY"] = "your api key"

llm = TogetherLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key="your_api_key"
)
resp = llm.complete("Who is Paul Graham?")
    print(resp)

# from llama_index.readers.microsoft_sharepoint import SharePointReader
# loader = SharePointLoader(
#     client_id="<Client ID of the app>",
#     client_secret="<Client Secret of the app>",
#     tenant_id="<Tenant ID of the Microsoft Azure Directory>",
# )

# documents = loader.load_data(
#     sharepoint_site_name="<Sharepoint Site Name>",
#     sharepoint_folder_path="<Folder Path>",
#     recursive=True,
# )
