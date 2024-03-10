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
4. Make sure you have the .env file in the root and have defined the openai key as OPENAI_API_KEY = "YOUR-OPENAI_KEY"
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

## DSPy versus Other Frameworks
**LangChain**, **LlamaIndex**, and **DSPy** each occupy unique positions in the ecosystem of tools designed to enhance the capabilities and applications of Large Language Models (LLMs). Here’s a closer look at the roles and advantages of each::
* **LangChain** : This framework is designed to chain together language models to create complex applications. It allows for the seamless integration of multiple LLMs into a single workflow, enabling developers to build sophisticated applications that leverage the strengths of various models. LangChain's focus is on facilitating the construction of complex applications by orchestrating different language models to work in concert, thus amplifying their utility and effectiveness in solving diverse problems.
* **LlamaIndex**:Specializes in enhancing search capabilities within texts using language models. Its primary goal is to improve the efficiency and accuracy of searching through large volumes of text by leveraging the contextual understanding capabilities of LLMs. LlamaIndex is particularly useful in scenarios where traditional search algorithms fall short, offering advanced search functionalities that are more nuanced and capable of understanding the semantics of the search query and the content.
* **DSPy**: Carves out its niche by focusing on optimizing prompt construction for better interaction with LLMs. The challenge of crafting effective prompts is crucial, as the quality of prompts directly impacts the performance of language models. DSPy addresses this challenge with a programmable approach that offers significant advantages in terms of precision and adaptability. By providing tools and frameworks that streamline the process of prompt engineering, DSPy enables developers to fine-tune their interactions with LLMs, leading to more accurate and relevant responses. This precision and adaptability make DSPy a valuable tool for developers looking to maximize the effectiveness of their LLM-based applications.

Each of these frameworks—LangChain, LlamaIndex, and DSPy—contributes to expanding the potential of LLMs in unique ways, from application building and enhanced search capabilities to optimized prompt engineering. DSPy's emphasis on programmable prompt construction, in particular, offers a distinct advantage for developers seeking to improve their LLM interactions through more precise and adaptable prompts, highlighting the importance of targeted tools in the evolving landscape of AI and machine learning. Though I feel that DSPy is currently in a very early stage, From the LangChain perspective we need to look towards HyDE.
## Reports
Have used [Weights and Biases Platform](https://wandb.ai) in order to track experiments in RAG LLM, iterate on Test, Synthetic etc. datasets, evaluate model performance, You can find out the integration with wandb in couple of jupiter notebooks.
## Generating Synthetic Questions and Answers:
Drawing inspiration from HotpotQA, I crafted the subsequent prompt to elicit questions and answers for the dataset.
Using the following prompt:
```
Below is the context for generating questions and answers.

---------------------
{context}
 ---------------------

Given the context above without using external information,
develop {questions_per_section} question(s) with their brief answer(s),
suitable for a quiz or examination. Keep answers concise, within 1-10 words. 
Ensure the generated content varies and aligns closely with the provided context."
    
"""

```
## Synthetic Questions and Answers (Generated):
```
* What was the effect of oropharyngeal anesthesia on obstructive sleep apnea in the study subjects?
* Oropharyngeal anesthesia led to an increase in obstructive apneas and hypopneas, as well as a higher frequency of oxyhemoglobin desaturations during sleep.
```
```
* What was the prognostic value of low neutrophil function for late pyogenic infections in bone marrow transplant recipients?
* Low neutrophil function, particularly defective skin window migration and combined defects, predisposed patients to late pyogenic infections after bone marrow transplantation.
```
```
* What was the treatment that resulted in both clinical and electrophysiological improvement in a patient with paraneoplastic vasculitic neuropathy?
* ""
```
```
* What was the conclusion regarding the role of CNS radioprophylaxis in the therapeutic management of childhood rhabdomyosarcoma with meningeal extension?
* The conclusion was that CNS prophylaxis with radiotherapy is questionable in the management of childhood RMSA with meningeal extension."
```
```
* What are the advantages of using duplex Doppler ultrasound in examining abdominal vasculature?
* The advantages include absence of toxicity, providing both physiologic and anatomic information, and avoiding the risks associated with contrast angiography.
```
```
* Why is congenital hypertrophy of the retinal pigment epithelium important in identifying patients with familial adenomatous polyposis (FAP)?
* Congenital hypertrophy of the RPE serves as a clinical marker for FAP patients who are at risk for cancer, aiding in their identification and management.
```
```
* What is the main conclusion drawn from the study on intraluminal Ca++ regulatory site defect in sarcoplasmic reticulum from malignant hyperthermia pig muscle?
* The study suggests an abnormality in an intraluminal, low affinity Ca++ binding site regulating Ca++ release occurs in the SR membrane of MH pig muscle."
```
```
* What is the purpose of the back isometric dynamometer (BID-2000) developed for elderly patients with osteopenia or osteoporosis?
* ""
```
```
* What was the effect of prophylactic peroral acyclovir (ACV) on the development of ultraviolet radiation-induced herpes labialis lesions?
* Prophylactic peroral ACV prevented the development of delayed lesions but not immediate lesions when started 7 days before or 5 minutes after UVR exposure.
```
```
* What unique myopathic changes were observed in the hypertrophied muscle of patients with hereditary internal anal sphincter myopathy?
* Vacuolar changes with periodic acid-Schiff-positive polyglycosan bodies in smooth muscle fibers and increased endomysial fibrosis were observed.
```
```
* What was found regarding the immune response of peripheral blood mononuclear cells to HBxAg in patients with hepatitis B virus infection?
* HBxAg-specific stimulation was observed in patients with acute (6 of 6) and chronic (6 of 17) hepatitis B virus infection, but not in healthy individuals.
```
```
* What is the role of neural cell adhesion molecule (N-CAM) in neuroendocrine tissues?
* N-CAM is involved in direct cell-cell adhesion in neuroendocrine tissues, as shown by its expression in most neuroendocrine cells and tumors with secretory granules.
```
```
* What role do genetics play in the development of cerebrovascular disease like stroke?
* Genetics have been shown to have a significant influence on the development of stroke, with studies indicating major genetic influences in the development of risk factors for cardiovascular disease.
```
```
* What was the predominant type of IgA found in gut lavage fluid in the study?
* Secretory IgA comprised 92%, 81.6%, and 76.7% of the total IgA gut lavage fluid content in the control, coeliac, and Crohn's groups, respectively.
```
```
* What are some alternative dosage regimens for rt-PA in patients with myocardial infarction?
* Some alternative dosage regimens for rt-PA in patients with myocardial infarction include bolus, front-loaded, and accelerated infusions.
```
```
* What was the cholestanol content of the cataractous lens nucleus from a patient with cerebrotendinous xanthomatosis (CTX)?
* The cholestanol content was quantified to be 0.27 micrograms per mg freeze-dried lens tissue.
```
```
* What was the overall risk of developing secondary leukaemia in patients treated for Hodgkin's disease in the British National Lymphoma Investigation studies?
* ""
```
## [RAGAS](https://github.com/explodinggradients/ragas) for Validation of LLM performance
What all metrics are considered:
* Faithfulness
* Relevance of Answers by RAG
* Context Precision
* Context Relevancy
* Context Recall
* Answer Semantic Similarity
* Correctness of Answers

## What all is Checked to validate the model performance:

### 1. Check for Memorization without Retrieval
```
class FactoidResponse(dspy.Signature):
    """Generates concise, fact-based responses to queries."""

    question = dspy.InputField()
    answer  = dspy.OutputField(desc="typically ranging from 1 to 5 words")

# Define the predictor.
generate_answer = dspy.Predict(FactoidResponse)
```
The primary objective of this experiment is to determine whether the model has memorized the data. For each question-answer duo, we will evaluate the similarity and accuracy of the response in comparison to the actual facts. It's important to mention that this experiment does not involve any retrieval process.
```
answer_similarity : 0.85
answer_correctness : 0.45
```

### 2. LLMs performance against the synthetically generated data with ground truth as context
Primary objective: To verify the model's capability to extract an accurate response from the factual context, assess for any hallucination, and evaluate the response's accuracy.

![GitHub Logo](https://github.com/shramanpadhalni-web/RAG_Evaluation_ragas/blob/main/synthetic.PNG "Performance against synthetic data")

### 3. Hyper-parameter Search
### 4. Evaluating Passage Relevance Through Composite Query Comparison
Steps:
* Initially, create augmented versions of all queries in the test dataset.
* Next, identify the most similar passages for each query.
* Keep a tally of the most frequently retrieved passages.
* Subsequently, determine the most common passages for each query and utilize these as the query's context.
* The context to identify the answer most closely matches the query.
### 5. Query Expansion
* Question + Hypothetical Answer -> Retrieved Context
* Context + Question -> Answer


