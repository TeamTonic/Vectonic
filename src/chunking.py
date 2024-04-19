# ./src/chunking.py


import os
from src.dataloader import DataProcessor, DocumentLoader
from unstructured.partition.md import partition_md

class MarkdownProcessor:
    def __init__(self, folder_path: str, output_folder: str):
        self.folder_path = folder_path
        self.output_folder = os.path.join(output_folder, "markdown_texts")
        os.makedirs(self.output_folder, exist_ok=True)
        
    def process_files(self):
        """Process each file to find Markdown and save the output."""
        print(f"Processing files in folder: {self.folder_path}")
        documents = DocumentLoader.load_documents_from_folder(self.folder_path)

        for doc in documents:
            file_path = os.path.join(self.output_folder, doc.name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc.content)
            print(f"Saved Markdown content to {file_path}")
        return [os.path.join(self.output_folder, f) for f in os.listdir(self.output_folder)]
        
    def analyze_markdown_files(self, file_paths):
        """Loads markdown files and uses the partition_md function for analysis."""
        elements_list = []
        for file_path in file_paths:
            elements = partition_md(filename=file_path)
            print(f"Processed '{file_path}' to Markdown elements:")
            elements_list.append((file_path, elements))
            
            with open(file_path.replace('.md', '_analyzed.txt'), 'w') as f:
                for element in elements:
                    f.write(str(element) + '\n\n')
            print(f"Analysis for '{file_path}' saved to '{file_path.replace('.md', '_analyzed.txt')}'")
        
        return elements_list
    
    def analyze_markdown_files(self, file_path):
        """
        Loads markdown files and uses the partition_md function for analysis.
        Returns the elements represented in the markdown file.
        """
        elements = partition_md(filename=file_path)
        print(f"Processed '{file_path}' to Markdown elements:")
        with open(file_path.replace('.md', '_analyzed.txt'), 'w', encoding='utf-8') as f:
            for element in elements:
                f.write(str(element) + '\n\n')
        print(f"Analysis saved to '{file_path.replace('.md', '_analyzed.txt')}'")
        return elements
# from unstructured.ingest.interfaces import ChunkingConfig
# from unstructured.documents.elements import Element
# from unstructured.partition.pdf import partition_pdf
# import unstructured
# from unstructured.partition.text import partition_md
# from unstructured.partition.auto import partition

# text_folder_path = "./texts"
# elements = partition_md(filename="")
# print("\n\n".join([str(el) for el in elements]))


# sample_location = "Vectonic/add_your_files_here/"
# ff = partition_text(
#         filename=sample_location
#     )

# def get_ff():
#     return ff
# class DataLoaderforChunking:
#     pass
# # chunker = ChunkingConfig(
    
# # )

