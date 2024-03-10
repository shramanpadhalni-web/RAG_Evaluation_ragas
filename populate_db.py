from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
import pandas as pd
from typing import List
import openai


DATA_PATH = 'data/medical_tc_train.csv'

# def tokenize_and_chunk_text(file_path: str) -> List[str]:
#     """
#     Reads text from a file, then tokenizes and chunks the text into manageable segments for further processing.

#     This function performs two main tasks:
#     1. Text Splitting: Breaks the entire text into smaller chunks based on specified separators.
#     2. Tokenization: Further splits these chunks into tokenized segments suitable for NLP tasks.

#     The function first reads the text from the given file path, ensuring proper handling of UTF-8 encoded text. 
#     It utilizes a recursive character text splitter for initial chunking based on characters like newline and period, 
#     followed by a sentence transformers token text splitter for more granular tokenization.

#     Parameters:
#         file_path (str): The path to the text file to be read and processed.

#     Returns:
#         List[str]: A list of tokenized and chunked text segments.

#     Raises:
#         FileNotFoundError: If the specified file does not exist or cannot be opened.
#         UnicodeDecodeError: If the text file contains characters that cannot be decoded using UTF-8.
#     """

#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             text = file.read()
#     except FileNotFoundError as e:
#         print(f"File not found: {e}")
#         raise
#     except UnicodeDecodeError as e:
#         print(f"Error decoding file. Ensure the file is UTF-8 encoded: {e}")
#         raise

#     # Initialize a RecursiveCharacterTextSplitter instance for initial chunking
#     char_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ". ", " "],
#         chunk_size=1000,
#         chunk_overlap=20
#     )

#     char_chunks = char_splitter.split_text(text)

#     # Initialize a SentenceTransformersTokenTextSplitter for tokenization
#     token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=20, tokens_per_chunk=1000)

#     token_chunks = []
#     for chunk in char_chunks:
#         token_chunks.extend(token_splitter.split_text(chunk))

#     return token_chunks


def tokenize_and_chunk_text_from_csv(file_path: str, text_column: str, max_batch_size: int = 5461) -> List[List[str]]:
    """
    Reads text from a column in a CSV file, tokenizes and chunks the text into manageable segments, and organizes these segments into batches. Each batch's size does not exceed the specified maximum, ensuring compatibility with systems or applications that have a limit on processing capacity.
    
    The function performs the following main tasks:
    1. CSV Reading: It reads the specified text column from a given CSV file path.
    2. Initial Chunking: Utilizes a character-based text splitter to divide the text into smaller chunks based on common separators such as newline characters and periods.
    3. Tokenization: Applies a sentence-level tokenization process to each chunk, creating finer-grained text segments suitable for natural language processing tasks.
    4. Batching: Organizes these tokenized text segments into batches, with each batch containing segments up to a specified maximum size to comply with processing limits.

    Parameters:
        file_path (str): The path to the CSV file from which text data will be read.
        text_column (str): The name of the column within the CSV file that contains the text data to be processed.
        max_batch_size (int, optional): The maximum allowable size of each batch in terms of the number of token chunks. Defaults to 5461, which should be adjusted based on the system or application's limitations.

    Returns:
        List[List[str]]: A list of batches, where each batch is a list of tokenized and chunked text segments. Each batch is guaranteed not to exceed the specified maximum batch size.

    Raises:
        FileNotFoundError: If the specified file path does not lead to an existing or accessible CSV file.
        KeyError: If the specified text column does not exist within the CSV file.
    
    """

    try:
        # Efficiently read the specified text column from the CSV file
        df = pd.read_csv(file_path, usecols=[text_column], encoding='utf-8')
        # Combine all non-null text data into a single string, separated by newlines
        text_data = '\n'.join(df[text_column].dropna())
    except FileNotFoundError as e:
        print(f"CSV file not found: {e}")
        raise
    except KeyError as e:
        print(f"Column '{text_column}' not found in the CSV file: {e}")
        raise

    # Use a character-based splitter for initial chunking based on common text separators
    char_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " "], chunk_size=300, chunk_overlap=10)
    char_chunks = char_splitter.split_text(text_data)

    # Further split each chunk into tokenized segments
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10, tokens_per_chunk=300)

    # Organize tokenized segments into batches that comply with the max_batch_size limit
    batches = []
    current_batch = []
    current_batch_size = 0

    for chunk in char_chunks:
        token_chunks = token_splitter.split_text(chunk)
        for token_chunk in token_chunks:
            # Start a new batch if adding the current chunk would exceed the max batch size
            if current_batch_size + len(token_chunk) > max_batch_size:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
            current_batch.append(token_chunk)
            current_batch_size += len(token_chunk)

    # Ensure the last batch is added if it contains any segments
    if current_batch:
        batches.append(current_batch)

    return batches





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
    text_collection = db_client.get_or_create_collection("medical_abstract_data_collection", embedding_function=encoder_function)
    
    document_ids = [str(idx) for idx, _ in enumerate(tokens_chunks)]
    text_collection.add(ids=document_ids, documents=tokens_chunks)

    print(f"Initialized Collection: {text_collection.name}")

def get_openai_embedding(text, model="text-embedding-ada-002"):
    """
    Generates an embedding for the given text using the specified OpenAI model.

    Parameters:
    - text (str): The text for which to generate an embedding.
    - model (str): The OpenAI model to use for generating the embedding.

    Returns:
    - list: The embedding vector for the input text.
    """
    response = openai.embeddings.create(input=text, model=model)
    embedding = response.data[0].embedding
    return embedding

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
        text_segments = tokenize_and_chunk_text_from_csv(DATA_PATH, "medical_abstract")
        
        # Process text segments and generate embeddings using the OpenAI model

        # if you want to use openai embeddings
        # embeddings = [get_openai_embedding(text_segment, "text-embedding-3-small") for text_segment in text_segments]

        #Just to make things quick we are using sentence transformer embeddings
        embeddings = SentenceTransformerEmbeddingFunction()

        # Setup the retrieval model with the processed text and embeddings
        setup_text_retrieval(text_segments, embeddings)
        
        print("Text retrieval system initialization finished.")
    except Exception as e:
        # Log the exception and re-raise to halt the program with an error message
        print(f"An error occurred during the execution of the pipeline: {e}")
        raise


if __name__ == "__main__":
    execute_pipeline()
