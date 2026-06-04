"""
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

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


def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming=True
    print("Streaming output:")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)

    print() # newline after streaming

def demo_schema_inspection():
    """Demonstrate schema inspection."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input schema: {input_schema}")
    print(f"Output schema: {output_schema}")


def exercise_first_chain():
    """Given a product name and audience, generate a marketing tagline."""
    prompt = ChatPromptTemplate.from_template("Generate a single catchy marketing tagline for the product {product} where the intended audience is {audience}")
    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result =chain.invoke({"product": "AI Course", "audience": "developers"})

    print(f"Chain output: {result}")

def new_way():
    # the universal way to initiate a model
    # see init_chat_model documentation and import

    prompt = ChatPromptTemplate.from_template("Generate a single catchy marketing tagline for the product {product} where the intended audience is {audience}")
    parser = StrOutputParser()

    # calls the right interface based on the model name as opposed to calling the model wrapper like ChatAnthropic()
    model = init_chat_model(model="soc-parser", model_provider="ollama",
                                  temperature=0.7,
                                  max_tokens=500)
    chain = prompt | model | parser 

    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Chain output: {result}")
    return result


def azure_openai():
    # use an azure openai model
    prompt = ChatPromptTemplate.from_template("Generate a single catchy marketing tagline for the product {product} where the intended audience is {audience}")
    parser = StrOutputParser()
    model = init_chat_model(deployment_name="gpt-4o-mini",   # deployment name, NOT the model name
                            model_provider="azure_openai",
                            model="gpt-4o-mini",
                            api_version="2024-08-01-preview",
                            temperature=0.7)
    chain = prompt | model | parser
    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Chain output: {result}")
    return result

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_execution()
    # demo_streaming()
    # demo_schema_inspection()
    # exercise_first_chain()
    # new_way()
    azure_openai()