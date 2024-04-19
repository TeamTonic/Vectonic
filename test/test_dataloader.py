# tests/test_dataloader.py

import os
import pytest
from src.dataloader import DataProcessor, DocumentLoader
from llama_index.core.schema import Document
from typing import List

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w") as f:
        f.write("Name, Age\nJohn, 30\nJane, 25\n")
    return str(csv_file)

def test_data_processor_load_csv(sample_csv):
    """Test loading data from a CSV file."""
    processor = DataProcessor(sample_csv, "test_collection", "/tmp")
    data = processor.load_data_from_source_and_store()
    assert data == [("John", 30), ("Jane", 25)]

@pytest.fixture
def sample_folder(tmp_path):
    """Create a sample folder with files for testing."""
    folder = tmp_path / "test_folder"
    folder.mkdir()
    files = ["test.docx", "test.pdf"]
    for file in files:
        with open(folder / file, "w") as f:
            f.write("Test content")
    return str(folder)

def test_document_loader_load_documents(sample_folder, capsys):
    """Test loading documents from a folder."""
    loader = DocumentLoader()
    documents = loader.load_documents_from_folder(sample_folder)
    assert len(documents) == 2
    captured = capsys.readouterr()
    assert "Loading document from 'test.docx'" in captured.out
    assert "Loading document from 'test.pdf'" in captured.out
    
    
def assert_list_of_documents(obj: List[Document]) -> None:
    assert isinstance(obj, list), "Object is not a list"
    for item in obj:
        assert isinstance(item, Document), "Element is not a Document"

def test_load_documents_from_folder():
    "Test"
    document_loader = DocumentLoader()
    list_of_docs = document_loader.load_documents_from_folder()
    
    assert_list_of_documents(list_of_docs)
    
    assert isinstance(list_of_docs, list), "Object is not a list"
    for sublist in list_of_docs:
        assert isinstance(sublist, Document), "Sub-element is not a list"