from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal
import os

load_dotenv()

# Define the schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes in the review")
    summary: str = Field(description="Brief summary")
    sentiment: Literal["pos", "neg"] = Field(description="Sentiment of the review")
    pros: Optional[list[str]]
    cons: Optional[list[str]]
    name: Optional[str]

# HuggingFace endpoint
hf_llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",
    task="text-generation",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    temperature=0.3,
    max_new_tokens=1024
)

model = ChatHuggingFace(llm=hf_llm)

# Construct prompt to simulate structured output
prompt = """
Extract structured information from the following product review.

Review:
---
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Subhadip Sinha
---

Return JSON with the following keys:
- key_themes: list of core topics discussed
- summary: brief summary
- sentiment: 'pos' or 'neg'
- pros: list of pros
- cons: list of cons
- name: reviewer's name
"""

# Get result
result = model.invoke([("user", prompt)])

print(result.content)
