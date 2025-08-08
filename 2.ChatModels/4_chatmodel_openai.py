from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)



# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI(
#     model='gpt-3.5-turbo',  # ✅ Available to free-tier
#     temperature=1.5,
#     max_tokens=10  # Fix param name
# )

# result = model.invoke("Write a 5 line poem on cricket")
# print(result.content)
