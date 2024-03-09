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
