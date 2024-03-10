import os
import dspy
from dotenv import load_dotenv
from db_retriever_module import ChromadbRetrieverModule
from dspy.teleprompt import BootstrapFewShot
import streamlit as st


class GenerateAnswer(dspy.Signature):
    """Generates answers to questions given the context.

    Attributes:
        context (dspy.InputField): May contain relevant facts for generating the answer.
        question (dspy.InputField): The question to be answered.
        answer (dspy.OutputField): Outputs the answer to the question, aiming for 1-100 words.
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer the question in 1-100 words")


class MedicalAbstractRag(dspy.Module):
    """Retrieval-Augmented Generation for answering questions using a retrieval model and generative LLM.

    Attributes:
        num_passages (int): Number of passages to retrieve for context.
    """

    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        """Retrieves context passages and generates an answer to the question.

        Args:
            question (str): The question for which an answer is generated.

        Returns:
            dspy.Prediction: Contains the context and the generated answer.
        """
        try:
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)
        except Exception as e:
            print(f"Failed to generate an answer: {e}")
            # Consider a default or an error-specific response
            return dspy.Prediction(context="", answer="Unable to generate an answer.")

def setup():
    """Configures the dspy and retrieval models with necessary settings.

    Returns:
        RAG: An instance of the RAG model ready for generating answers.
    """
    try:
        load_dotenv()

        # NOTE: This example uses the local_embedding_model for ChromaDBRetrieverModule, If you want to
        # use open ai embeddings please go ahead

        # Load API key securely
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")
        
        # Configuration for dspy models
        turbo = dspy.OpenAI(model='gpt-3.5-turbo')
        chroma_rm = ChromadbRetrieverModule(
            db_collection_name="medical_abstract_data_collection",  # The name of the ChromaDB collection
            persist_directory="local_chroma.db",  # Directory path for ChromaDB persistence
            local_embed_model="sentence-transformers/paraphrase-MiniLM-L6-v2",  # The local embedding model
            api_key=openai_api_key,  # OpenAI API key (if using that, i am just sentence transformer embedding)
            result_limit=7  # Default number of passages to retrieve per query, adjust as needed
        )

        dspy.settings.configure(lm=turbo, rm=chroma_rm)
        return MedicalAbstractRag()
    except Exception as e:
        print(f"Failed to set up the models: {e}")
        # Exiting or returning a specific value could be considered here
        raise

# OK Lets setup the model
model = setup()

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)

# Lets prepare some training set
trainset = [
    dspy.Example(question="What are the main categories of diseases discussed in the medical abstracts?", 
                 answer="The main categories include neoplasms, digestive system diseases, nervous system diseases, cardiovascular diseases, and general pathological conditions.").with_inputs('question'),
            
    dspy.Example(question="Which disease category is most frequently addressed in the abstracts?", 
                 answer="Neoplasms are the most frequently addressed disease category in the abstracts.").with_inputs('question'),
    
    dspy.Example(question="What methodologies are commonly used in the studies described in the medical abstracts?", 
                answer="Common methodologies include clinical trials, observational studies, meta-analyses, and case reports.").with_inputs('question'),
    
    dspy.Example(question="How is the effectiveness of a new treatment evaluated in the medical abstracts?", 
                answer="The effectiveness of a new treatment is often evaluated through randomized controlled trials, comparing outcomes with a control group receiving standard treatment or placebo.").with_inputs('question'),
    
    dspy.Example(question="Can you describe the role of genetics in the development of neoplasms as discussed in the abstracts?", 
                answer="Genetics plays a crucial role in the development of neoplasms, with many abstracts discussing genetic mutations, hereditary risk factors, and the molecular mechanisms driving oncogenesis.").with_inputs('question'),
    
    dspy.Example(question="What advancements in cardiovascular disease treatment are highlighted in the abstracts?", 
                answer="Advancements in cardiovascular disease treatment highlighted in the abstracts include new pharmacological therapies, minimally invasive surgical techniques, and improvements in diagnostic imaging.").with_inputs('question'),
    
    dspy.Example(question="How do the abstracts address the impact of lifestyle factors on digestive system diseases?", 
                answer="The abstracts address the impact of lifestyle factors such as diet, alcohol consumption, smoking, and physical activity on the incidence and progression of digestive system diseases.").with_inputs('question'),
    
    dspy.Example(question="What are the emerging trends in the management of nervous system diseases according to the abstracts?", 
                answer="Emerging trends in the management of nervous system diseases include the use of precision medicine, advancements in neuroimaging techniques, and novel therapeutic approaches like gene therapy.").with_inputs('question'),
    
    dspy.Example(question="What challenges in diagnosing general pathological conditions are discussed?", 
                answer="Challenges in diagnosing general pathological conditions discussed include the variability of symptoms, the need for advanced diagnostic tools, and the importance of differential diagnosis.").with_inputs('question'),
    
    dspy.Example(question="How is patient quality of life addressed in the context of chronic diseases in the abstracts?", 
                answer="Patient quality of life in the context of chronic diseases is addressed through discussions on pain management, mental health support, lifestyle modifications, and palliative care.").with_inputs('question'),]
#
# Compile!
compiled_rag = teleprompter.compile(model, trainset=trainset)

st.title('Medical Abstract RAG Question Answering System')

# Streamlit UI components for input and interaction
user_prompt = st.text_input("Enter your question here:")

if st.button('Submit'):
    if user_prompt:  # Check if the input is not empty
        # Generate and display the response for the given prompt
        response = compiled_rag(user_prompt)
        st.write(f"Answer: {response.answer}")
    else:
        st.write("Please enter a question to get an answer.")

