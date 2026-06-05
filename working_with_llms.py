"""
Multiple providers, configuration, streaming and cost optimization
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()

# Azure is not as generic so set it seprately:
azure_chat_model = init_chat_model(model_provider="azure_openai",
                                    model=os.getenv("AZURE_OPENAI_MODEL"),  
                                    temperature=0.7,
                                    streaming=True,
                                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                                    max_retries=3)


def demo_init_model(model, model_provider, temperature=0.7, streaming=True, max_retries=3):
    chat_model = init_chat_model(model_provider=model_provider,
                    model=model,
                    temperature=temperature,
                    streaming=streaming,
                    max_retries=max_retries)
    return chat_model

if os.getenv("ANTHROPIC_API_KEY"):
    anthropic_chat_model = init_chat_model(model_provider="anthropic",
                                            model="claude-haiku-4-5",
                                            temperature=0.7,
                                            streaming=True,
                                            max_retries=3)
else:
    anthropic_chat_model = None

def demo_model_comparison():
    prompt = "Explain recursion in one sentence."
    models = {
        "gpt-4o-mini": init_chat_model(model_provider="azure_openai",
                                        deployment_name="gpt-4o-mini",
                                        model="gpt-4o-mini",
                                        temperature=0.7,
                                        streaming=False,
                                        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                                        max_retries=3),
    }
    
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-haiku-4-5"] = demo_init_model(model_provider="anthropic",
                                                    model="claude-haiku-4-5",
                                                    temperature=0.7,
                                                    streaming=True,
                                                    max_retries=3)
    print(f"Prompt: {prompt}")
    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")


# Next step is to look at the messages object and see how to use it to create a conversation history.


# response = anthropic_chat_model.invoke("Say 'setup complete' in one word.")
# print(f"Response from Anthropic: {response.content}")
# # response = azure_chat_model.invoke("Say 'setup complete!' in one word.")
# # print(f"Response from Azure OpenAI: {response.content}")


def demo_message():
    model = ChatAnthropic(model="claude-haiku-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages = [
        SystemMessage(content="You are a pirate. Always answer in pirate speak."),
        HumanMessage(content="Is the weather like today?")
    ]
    
    response = model.invoke(messages)
    print(f"{messages[0]} | {messages[1]}")
    print(f"Response from pirate: {response.content}")

    # Multi-turn conversation using message objects
    messages.append(response)
    messages.append(HumanMessage(content="What about tomorrow?"))
    response = model.invoke(messages)
    print(f"Pirate says: {response.content}")
# TODO: For symmetric instatiation of models, use another claude model or add an actual OpenAI API key
def exercise_multi_model(prompt, models) -> dict[str, str]:
    model_responses = {}
    for model_name in models:
        model_responses[model_name] = demo_init_model(model=model_name, 
                                            model_provider='anthropic', 
                                            temperature=0.7, 
                                            streaming=False, 
                                            max_retries=3).invoke(prompt)

    return model_responses



if __name__ == "__main__":
    # demo_model_comparison()
    # demo_message()
    model_responses = exercise_multi_model("What is AI?", ["claude-sonnet-4-5", "claude-haiku-4-5"])
    for model, response in model_responses.items():
        print(f"Model: {model} | Response: {response.content}")