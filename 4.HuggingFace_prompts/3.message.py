from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Initialize Hugging Face endpoint
hf_llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",  # You can change the model here
    task="text-generation",
    api_key=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

# Wrap it as Chat Model
model = ChatHuggingFace(llm=hf_llm)

# Prepare message history
messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

# Generate response
result = model.invoke(messages)

# Append AI response
messages.append(AIMessage(content=result.content))

# Print the conversation
for msg in messages:
    print(f"{msg.type.title()}: {msg.content}")
