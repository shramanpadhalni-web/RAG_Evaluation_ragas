from typing import Optional, List, Union
import openai
import dspy
import backoff
from dsp.utils import dotdict

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    chromadb_import_success = True
except ImportError as exc:
    chromadb_import_success = False
    error_message = (
        "Failed to import the 'chromadb' library or one of its components. "
        "This library is essential for certain functionalities. Please ensure it is installed properly. "
        "If not installed, you can install it using: pip install dspy-ai[chromadb]"
    )
    print(error_message)
    raise ImportError(error_message) from exc

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ModuleNotFoundError(
        "The Hugging Face transformers library is missing, which is essential for using local embedding models with ChromadbRM. "
        "Please install it by running: pip install transformers"
    ) from exc

if chromadb is None:
    raise ImportError(
        "The 'chromadb' library has not been found. ChromadbRM relies on this library for its core functionalities. "
        "To resolve this, install the required library using the command: pip install dspy-ai[chromadb]"
    )

class PassageRetriever(dspy.Retrieve):
    """
    Implements a passage retrieval mechanism leveraging the chromadb database for returning top-ranked passages based on a query.

    This class assumes an existing chromadb index filled with passages and their metadata.

    Parameters:
        db_name (str): The name of the chromadb collection.
        storage_path (str): Directory path for chromadb persistence.
        embedding_model (str, optional): Identifier for the OpenAI embedding model. Defaults to "text-embedding-ada-002".
        api_key (str, optional): OpenAI API key. Defaults to None.
        organization (str, optional): Specifies the OpenAI organization. Defaults to None.
        result_limit (int, optional): Number of passages to retrieve per query. Defaults to 7.

    Returns:
        dspy.Prediction: Object containing the matched passages.

    Usage Example:
        Initialize the retriever and configure it as the default:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever = PassageRetriever('my_collection', 'path_to_db')
        dspy.settings.configure(lm=llm, rm=retriever)
        # Query the retriever
        retriever.query("search term")
        ```

        Integrate within the forward method of a dspy module:
        ```python
        self.retriever = PassageRetriever('my_collection', 'path_to_db', result_limit=5)
        ```
    """

    def __init__(
        self,
        db_name: str,
        storage_path: str,
        embedding_model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        local_model_path: Optional[str] = None,
        result_limit: int = 7,
    ):
        self.embedding_model = embedding_model
        self._setup_chromadb(db_name, storage_path)

        self.embedding_function = embedding_functions.OpenAIEmbedFunction(
            api_key=api_key,
            base_url=api_base_url,
            version=api_version,
            model=embedding_model,
        )

        self.is_local_model = False
        if local_model_path is not None:
            self._local_model = AutoModel.from_pretrained(local_model_path)
            self._local_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.is_local_model = True
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        super().__init__(result_limit=result_limit)

    def _setup_chromadb(self, db_name: str, storage_path: str) -> None:
        """Initializes and loads the chromadb index.

        Parameters:
            db_name (str): Name of the chromadb collection.
            storage_path (str): Path to the chromadb storage directory.
        """
        self.chromadb_client = chromadb.Client(Settings(storage_path=storage_path, persistent=True))
        self.db_collection = self.chromadb_client.get_collection(name=db_name)
        
        print(f"Loaded Collection Size: {self.db_collection.count_documents()}")
        if not self.chromadb_client.list_collections():
            raise ValueError(f"The collection '{db_name}' is missing. Please initialize it in chromadb.")
    
    @backoff.on_exception(backoff.expo,(openai.RateLimitError),max_time=15, jitter=None)

    def _generate_query_embeddings(self, search_terms: List[str]) -> List[List[float]]:
            """Generates embeddings for the input search terms.

            Parameters:
                search_terms (List[str]): List of search terms.

            Returns:
                List[List[float]]: List of embeddings for the input search terms.
            """
            
            if not self.is_local_model:
                # Utilize the OpenAI API to generate embeddings for the provided search terms.
                embeddings_response = openai.Embedding.create(
                    input=search_terms, model=self.embedding_model)
                return [emb.vector for emb in embeddings_response.data]

            # Switch to using the local embedding model when available.
            tokenized_inputs = self._local_tokenizer(search_terms, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._local_model(**tokenized_inputs.to(self.device))

            pooled_embeddings = self.aggregate_embeddings(outputs, tokenized_inputs['attention_mask'])
            final_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)
            return final_embeddings.cpu().numpy().tolist()

    def aggregate_embeddings(self, model_output, attention_masks):
        """
        Applies mean pooling on the output of a transformer model to get sentence-level embeddings.

        Parameters:
            model_output: The output from the transformer model, typically containing token embeddings.
            attention_masks: A tensor indicating which tokens are padding and which are not.

        Returns:
            Torch tensor representing aggregated sentence embeddings.
        """
        token_embeddings = model_output[0]  # Assuming the first output contains token embeddings
        expanded_attention_masks = attention_masks.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * expanded_attention_masks, 1)
        mean_embeddings = sum_embeddings / torch.clamp(expanded_attention_masks.sum(1), min=1e-9)
        return mean_embeddings

    def forward(
            self, query_or_queries: Union[str, List[str]], k: Optional[int] = None
            ) -> dspy.Prediction:
        """
        Executes a search operation in the database using provided queries and returns top-ranked passages.

        This method accepts either a single query string or a list of query strings. It then generates embeddings for these queries, performs a search in the chromadb collection based on these embeddings, and retrieves the top `k` passages that match the query or queries.

        Parameters:
            query_or_queries (Union[str, List[str]]): A single query string or a list of query strings for which to retrieve matching passages.
            k (Optional[int]): The number of top passages to retrieve for each query. If not specified, defaults to the class attribute `k`.

        Returns:
            dspy.Prediction: An object encapsulating the retrieved passages. Each passage is represented as a dictionary with key 'long_text', containing the passage text.

        Note:
            The method filters out any empty query strings before processing. It relies on the `_get_embeddings` method to convert query strings into embeddings and uses these embeddings to query the chromadb collection.
        """
        # Ensure input is in list format and filter out empty strings
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q.strip() for q in queries if q.strip()]  # Enhanced filtering to remove only whitespace queries

        # Generate embeddings for the non-empty queries
        embeddings = self._generate_query_embeddings(queries)

        # Determine the number of results to retrieve
        k = self.k if k is None else k

        # Execute the search in the database
        search_results = self._chromadb_collection.query(
            query_embeddings=embeddings, n_results=k)

        # Extract and format passages from the search results
        passages = [dotdict({"long_text": doc}) for doc in search_results["documents"][0]]
        return passages
