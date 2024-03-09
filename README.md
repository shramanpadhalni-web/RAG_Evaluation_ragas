# QA RAG (Retrieval-Augmented Generation) on Specific Medical Dataset with DSPy-Enhanced Evaluation

This project introduces a Question and Answer RAG (Retrieval Augmented Generation) Language Model, leveraging a specialized medical abstract dataset covering five distinct patient conditions. Utilizing DSPy, a cutting-edge framework designed to streamline LM prompt and weight optimization,  approach in this project significantly reduces the need for intricate prompt engineering. By requiring only a minimal set of examples for prompt "training," DSPy enhances the model's efficiency and applicability. Furthermore, the project employ synthetic test data generation and validation alongside the use of ragas, an evaluation framework dedicated to assessing the efficacy of RAG pipelines, ensuring both the model's robustness and its practical utility, making sure that RAG model is giving context aware answers.

## Setup
1. Clone the Repository:
   ```
   git clone [repository URL]
   ```
   

3. Create and Activate virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the requirements:

   ```
   pip install -r requirements.txt
   ```
## Usage/Examples

This project uses the ChromaDB for the vector embeddings. The script populate_db.py populates the ChromaDB with embeddings from a [medical abstract dataset](https://github.com/sebischair/Medical-Abstracts-TC-Corpus), enabling advanced text retrieval capabilities. Follow the steps below to use it in your project:

```python

python populate_db.py
```
The above step will construct the database and store it within the **local_chroma.db folder**. It leverages the dataset found in the **data/medical_tc_train.csv** to create the embeddings automatically. The collection created in the chroma DB is named as : **medical_abstract_data_collection**, Please make sure while retrieving the collection you use the name correctly. In order to run the things very fast this project uses the **SentenceTransformerEmbeddingFunction**, which by default is "**all-MiniLM-L6-v2**" embeddings. The "**all-MiniLM-L6-v2**" model is a smaller, distilled version of larger language models, designed to maintain high performance while being more efficient for deployment. Specifically, it has 6 layers, which contributes to its compactness and efficiency. There is **get_openai_embedding** in the script above which you can use to explore the open ai embeddings.

The **execute_pipeline()** function does the following steps to setup the Retrieval system:
Steps:
    1. Tokenize and chunk text: Break down the raw text into smaller, manageable pieces for processing.
    2. Initialize embedding function: Sets up a mechanism for converting text chunks into vector representations.
    3. Setup text retrieval: Incorporates the text chunks and their embeddings into the retrieval database (ChromsDB).
