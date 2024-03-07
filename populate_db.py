from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb

DATA_PATH = 'data/train.txt'

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

def tokenize_and_chunk_text(file_path: str) -> List[str]:
    """
    Reads text from a file, then tokenizes and chunks the text into manageable segments for further processing.

    This function performs two main tasks:
    1. Text Splitting: Breaks the entire text into smaller chunks based on specified separators.
    2. Tokenization: Further splits these chunks into tokenized segments suitable for NLP tasks.

    The function first reads the text from the given file path, ensuring proper handling of UTF-8 encoded text. 
    It utilizes a recursive character text splitter for initial chunking based on characters like newline and period, 
    followed by a sentence transformers token text splitter for more granular tokenization.

    Parameters:
        file_path (str): The path to the text file to be read and processed.

    Returns:
        List[str]: A list of tokenized and chunked text segments.

    Raises:
        FileNotFoundError: If the specified file does not exist or cannot be opened.
        UnicodeDecodeError: If the text file contains characters that cannot be decoded using UTF-8.
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except UnicodeDecodeError as e:
        print(f"Error decoding file. Ensure the file is UTF-8 encoded: {e}")
        raise

    # Initialize a RecursiveCharacterTextSplitter instance for initial chunking
    char_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=250,
        chunk_overlap=5
    )

    char_chunks = char_splitter.split_text(text)

    # Initialize a SentenceTransformersTokenTextSplitter for tokenization
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=5, tokens_per_chunk=250)

    token_chunks = []
    for chunk in char_chunks:
        token_chunks.extend(token_splitter.split_text(chunk))

    return token_chunks


def setup_text_retrieval(tokens_chunks, encoder_function):
    """
    Initializes and configures a text retrieval model with a specific embedding function. This process involves 
    creating or accessing a collection within a ChromaDB database and populating it with tokenized text chunks. 
    Each chunk is assigned a unique identifier and stored alongside its computed embeddings for efficient retrieval.

    Parameters:
    - tokens_chunks (list of str): This parameter represents the pre-processed and tokenized segments of text 
      that are to be indexed and made retrievable. Each element in the list is a string corresponding to a 
      chunk of the original text that has been tokenized.
    
    - encoder_function (callable): A function capable of generating embeddings from text. This function is used 
      to convert the text chunks into a vector space, facilitating their retrieval based on semantic similarity. 
      The encoder function should accept a list of strings as input and return a list of embedding vectors.

    The function first establishes a connection to a persistent ChromaDB instance, targeting a database file named 
    "local_chroma.db". It then either retrieves an existing collection named "pubmed_data_collection" or creates a new 
    one if it doesn't exist, specifying the `encoder_function` as the mechanism for generating text embeddings.

    Each tokenized chunk is assigned a unique identifier based on its index in the `tokens_chunks` list. These identifiers 
    are used to add the chunks to the collection along with their embeddings, facilitating later retrieval based on query 
    similarity.

    After successfully adding the documents to the collection, the function prints the name of the initialized collection 
    to confirm the operation's success.
    
    This function simplifies the integration of tokenized texts into a ChromaDB collection, making them readily 
    retrievable based on semantic similarity queries.
    """
    db_client = chromadb.PersistentClient("local_chroma.db")
    text_collection = db_client.get_or_create_collection("pubmed_data_collection", embedding_function=encoder_function)
    
    document_ids = [str(idx) for idx, _ in enumerate(tokens_chunks)]
    text_collection.add(ids=document_ids, documents=tokens_chunks)

    print(f"Initialized Collection: {text_collection.name}")


def execute_pipeline():
    """
    Orchestrates the execution of the text processing pipeline.

    This function carries out a series of steps to prepare text data for a retrieval system. It begins by tokenizing 
    and chunking the text located at a predefined path (DATA_PATH). Then, it initializes an embedding function 
    based on the Sentence Transformers library to generate embeddings for each text chunk. Lastly, these preprocessed 
    text segments and their embeddings are used to set up the retrieval model in a ChromaDB database, making them 
    searchable based on semantic similarity.

    Steps:
    1. Tokenize and chunk text: Breaks down the raw text into smaller, manageable pieces for processing.
    2. Initialize embedding function: Sets up a mechanism for converting text chunks into vector representations.
    3. Setup text retrieval: Incorporates the text chunks and their embeddings into the retrieval database.

    Error Handling:
    - The function includes error handling to ensure graceful failure and meaningful error messages in case of 
      issues during the text processing or database setup phases.

    Upon successful completion, the function prints a confirmation message indicating the retrieval system is ready.
    """
    try:
        # Tokenize and chunk the text from the specified data path
        text_segments = tokenize_and_chunk_text(DATA_PATH)
        
        # Initialize the embedding function with Sentence Transformers
        embedding_func = SentenceTransformerEmbeddingFunction()
        
        # Setup the retrieval model with the processed text and embeddings
        setup_text_retrieval(text_segments, embedding_func)
        
        print("Text retrieval system initialization finished.")
    except Exception as e:
        # Log the exception and re-raise to halt the program with an error message
        print(f"An error occurred during the execution of the pipeline: {e}")
        raise


if __name__ == "__main__":
    execute_pipeline()
