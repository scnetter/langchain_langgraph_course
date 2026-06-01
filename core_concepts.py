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


def demo_batch_execution():
    """Demonstrate batch execution for multiple inputs"""
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser
    
    # Batch - run with multiple inputs
    inputs = [{"text": "Hello, how are you?"}, {'text': 'What is your name?'}, {'text': 'Is the weather nice today?'}]

    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")
    
    return results

if __name__ == "__main__":
    # demo_basic_chain()
    demo_batch_execution()