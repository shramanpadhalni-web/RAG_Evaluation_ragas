import os
import dspy

from db_retriever_module import PassageRetriever

EXIT_PROMPT = "exit"

class GenerateAnswer(dspy.Signature):
    """Generates answers to questions given the context.

    Attributes:
        context (dspy.InputField): May contain relevant facts for generating the answer.
        question (dspy.InputField): The question to be answered.
        answer (dspy.OutputField): Outputs the answer to the question, aiming for 1-5 words.
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer the question in 1-5 words")


class RAG(dspy.Module):
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
        # Load API key securely
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")
        
        # Configuration for dspy models
        turbo = dspy.OpenAI(model='gpt-3.5-turbo')
        chroma_rm = PassageRetriever(
            collection_name="test",
            persist_directory="chroma.db",
            local_embed_model="sentence-transformers/paraphrase-MiniLM-L6-v2",
            openai_api_key=openai_api_key
        )

        dspy.settings.configure(lm=turbo, rm=chroma_rm)
        return RAG()
    except Exception as e:
        print(f"Failed to set up the models: {e}")
        # Exiting or returning a specific value could be considered here
        raise

if __name__ == "__main__":
    try:
        rag = setup()

        while True:
            print(f"\n\nEnter the prompt or type '{EXIT_PROMPT}' to exit:\n")
            prompt = input().strip()

            if prompt.lower() == EXIT_PROMPT:
                print("Exiting...")
                break
            
            if not prompt:
                print("Empty input. Please enter a valid question.")
                continue

            try:
                response = rag(prompt)
                print(f"\nAnswer: {response.answer}")
            except Exception as e:
                print(f"Error processing your question: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
