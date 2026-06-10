from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

import os
load_dotenv()

# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")

# #format the prompt and inspect
# messages = prompt.format_messages(adjective="funny", topic="chickens")

# print(messages)

# multi-message templates
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates from {input_language} to {output_language}"),
    ("human", "Translate the following text: {text}.")
])

messages = prompt.format_messages(
    input_language="English",
    output_language="French",
    text="Hello, how are you?"
)

model = init_chat_model(model_provider="anthropic",
                        model="claude-haiku-4-5",
                        temperature=0.7)

response = model.invoke(messages)
print(response.content)