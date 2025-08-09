
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file (for API key)
load_dotenv()

# Initialize the HuggingFace LLM endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",  # Use a chat-capable HF model
    task="text-generation",
    api_key=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

# Wrap it in ChatHuggingFace for chat history support
model = ChatHuggingFace(llm=llm_endpoint)

# Initialize chat history
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

# Chat loop
while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == 'exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

# Optional: Print chat history at the end
for msg in chat_history:
    print(f"{msg.type.title()}: {msg.content}")
