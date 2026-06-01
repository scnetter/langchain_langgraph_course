"""
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

def demo_basic_chain():
    """Demonstrate a basic chain using LCEL and Runnables"""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant. Answer in one sentence: {question}"
    )
    model = ChatAnthropic(model="claude-haiku-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))
    parser = StrOutputParser()

    # Compose with the pipe operator
    chain = prompt | model | parser
    
    # Run the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})

    print(f"Result: {result}")

    return result

if __name__ == "__main__":
    demo_basic_chain()