from unstructured.ingest.interfaces import ChunkingConfig
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text

sample_location = "Vectonic/add_your_files_here/One Attention Head Is All You Need for Sorting Fixed-Length Lists.pdf"
ff = partition_pdf(
        filename=sample_location
    )


def get_ff():
    return ff


class DataLoaderforChunking:
    pass
# chunker = ChunkingConfig(
    
# )

