# Learning Notes and Future Concepts to try

Routing between models and prepping multiple models for use:

```python
# Local Ollama — general/non-sensitive work
local_model = init_chat_model("deepseek-r1:8b", model_provider="ollama")

# Azure OpenAI — sensitive security workflows
soc_model = init_chat_model("your-deployment-name", model_provider="azure_openai")

def call_model(state: MessagesState):
    # Route based on workflow context
    model = soc_model if state.get("sensitive") else local_model
    response = model.invoke(state["messages"])
    return {"messages": response}
```