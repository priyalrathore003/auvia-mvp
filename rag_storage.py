from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize persistent ChromaDB vector store
vector_store = Chroma(
    collection_name="audio_transcripts",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Seed with mock audio issue documents
mock_documents = [
    Document(
        page_content="File_A has a severe low-frequency hum at 60Hz, likely caused by electrical interference from the recording environment.",
        metadata={"source": "File_A", "issue_type": "low_frequency_noise"},
    ),
    Document(
        page_content="Vocal_B contains harsh sibilance in the upper frequencies (6kHz-10kHz), making 's' and 'sh' sounds overly sharp and fatiguing.",
        metadata={"source": "Vocal_B", "issue_type": "sibilance"},
    ),
    Document(
        page_content="Track_C exhibits clipping distortion during the chorus sections, with peak levels exceeding 0dBFS by approximately 3dB.",
        metadata={"source": "Track_C", "issue_type": "clipping_distortion"},
    ),
]

vector_store.add_documents(mock_documents)


@tool
def query_audio_transcripts(query: str) -> str:
    """Query the audio transcripts vector store for relevant audio issue information."""
    results = vector_store.similarity_search(query, k=2)
    return "\n\n".join(doc.page_content for doc in results)