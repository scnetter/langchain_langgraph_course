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
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that translates from {input_language} to {output_language}"),
#     ("human", "Translate the following text: {text}.")
# ])

# messages = prompt.format_messages(
#     input_language="English",
#     output_language="French",
#     text="Hello, how are you?"
# )
# print(messages)
# model = init_chat_model(model_provider="anthropic",
#                         model="claude-haiku-4-5",
#                         temperature=0.7)

# response = model.invoke(messages)
# print(response.content)

# # Message Types:
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, Document, TraceableMessage, FunctionMessage

# messages = [HumanMessage(content="What is the weather like in Tokyo?"),
#             AIMessage(content="How can I assist you?"),
#             SystemMessage(content="You are a helpful assistant that translates from English to French."),
#             ToolMessage(content="Tool executed successfully.", tool_call_id="123"),
#             Document(page_content="The weather in Tokyo is sunny and warm.", metadata={"source": "weather_api"}),
#             TraceableMessage(content="The weather in Tokyo is sunny and warm.", trace_id="123"),
#             FunctionMessage(content="The weather in Tokyo is sunny and warm.", function_call={"name": "get_weather", "arguments": {"city": "Tokyo"}}),
#             ]


#fewshot example
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "Happy", "output": "Sad"},
    {"input": "Sad", "output": "Happy"},
    {"input": "Angry", "output": "Calm"},
    {"input": "Calm", "output": "Angry"},
    {"input": "Fearful", "output": "Confident"},
    {"input": "Confident", "output": "Fearful"},
    {"input": "Surprised", "output": "Calm"},
    {"input": "Calm", "output": "Surprised"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Give the opposite of each word"),
    fewshot_prompt,
    ("human", "{input}"),
])

messages = final_prompt.format_messages(input="Tedious")

model = init_chat_model(model_provider="anthropic",
                        model="claude-haiku-4-5",
                        temperature=0.7)
response = model.invoke(messages)
print(response.content)

# resusable components
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}")
])

user_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

# Combine
full_prompt = system_prompt | user_prompt
print(full_prompt)