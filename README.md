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
1. Tokenize and chunk text: Break down the raw text into smaller, manageable pieces for processing.
2. Initialize embedding function: Sets up a mechanism for converting text chunks into vector representations.
3. Setup text retrieval: Incorporates the text chunks and their embeddings into the retrieval database (ChromsDB).

To give the chatbot like experience the QA system is being integrated with the streamlit library, hence run the command:
```python

streamlit run main.py
```
When running above command the chat UI opens up and after user inputs the question into the UI here is what happens:

This system leverages the dspy framework, Streamlit, and a retrieval-augmented generation (RAG) approach to provide answers to questions based on a database of medical abstracts. Below, is the outline of how things are happening step by step:

### Step 1: Import Necessary Libraries
The system begins by importing essential Python libraries and modules:
* os and dotenv for managing environment variables.
* DSPy framework which signifies a transformative advancement in the interaction between developers and Large Language Models (LLMs), fundamentally altering the landscape of prompt engineering. Traditional approaches necessitated manual crafting of prompts, a method that was not only labor-intensive but also lacked precision. DSPY streamlines this process, offering developers an innovative toolkit that enhances efficiency and accuracy in prompt engineering. This shift empowers developers to focus more on creating impactful applications with LLMs, rather than getting bogged down in the minutiae of prompt construction.
The following picture depicts the workflow how we can create the LLM Application based on DSPy framework.
![GitHub Logo](https://github.com/shramanpadhalni-web/RAG_Evaluation_ragas/blob/main/DSpy_workflow.PNG "DSPy workflow to create LLM Apps")

* ChromadbRetrieverModule for the retrieval model.
* streamlit for the web app interface.
### Step 2: Define Signature and Module Classes
* **GenerateAnswer**: A DSPy signature class for generating answers to questions given a context.
* **MedicalAbstractRag**: A DSPy module class that employs retrieval-augmented generation (RAG) for answering questions. It retrieves relevant passages and then generates an answer.
### Step 3: Setup and Configuration
* The setup() function configures the environment by loading API keys and setting up DSPy and retrieval models. It initializes the MedicalAbstractRag model with specified parameters.
### Step 4: Initialize the Model
* The system initializes the RAG model using the setup() function. This model will be used to process the questions and generate answers.
### Step 5: Training Data Preparation
* A set of example questions and answers related to medical abstracts are compiled into a training set. This data will be used to fine-tune the question-answering capabilities of the model. And this is what I like about DSPy framework to build RAG modules, this really makes the step from Prompt Engineering to Prompt Programming, as Data/Concept drift will happen hence you are not guaranteed to get the consistent answers over the period of time.
### Step 6: Compile the RAG Program
* The BootstrapFewShot class from dspy is used to compile the RAG program with the prepared training set. This step optimizes the model's ability to answer questions accurately.
### Step 7: Integrate with Streamlit for Web Interface
* Streamlit is used to create a simple web interface where users can input their questions related to medical abstracts. The interface includes an input field for the question and a submit button.
### Step 8: Generate and Display Answers
* Upon clicking the submit button, the system processes the input question through the compiled RAG model to generate an answer. The answer is then displayed on the web interface.
