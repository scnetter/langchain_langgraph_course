from dotenv import load_dotenv
from importlib.metadata import version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os

core_version = version("langchain-core")
lg_version = version("langgraph")

load_dotenv()

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
    llm = ChatAnthropic(model="claude-haiku-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = llm.invoke("Say 'setup complete!' in one word.")
    print(f"Response from Anthropic: {response.content}")

    print("Setup complete!")


if __name__ == "__main__":
    main()
