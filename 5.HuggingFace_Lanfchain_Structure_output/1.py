from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # You can choose any suitable model
)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """Analyze the following review and extract the following information:
    
    Review: {review}

    Key themes: List the main topics discussed
    Summary: Brief summary of the review
    Sentiment: pos or neg
    Pros: List of positive aspects or None
    Cons: List of negative aspects or None
    Name: Reviewer's name if mentioned or None
    """
)

# You'll need to use a different model since we're not using OpenAI
# For example, you could use a HuggingFace model through LangChain
from langchain_community.llms import HuggingFaceHub

# Initialize a HuggingFace model
hf_model = HuggingFaceHub(
    repo_id="",  # You can choose any suitable model
    model_kwargs={"temperature": 0, "max_length": 512}
)

# Create a chain
chain = prompt | hf_model | StrOutputParser()

# Define your schema (same as before)
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return sentiment of the review either negative, positive or neutral"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

# Note: The structured output functionality might not be directly available with HuggingFace models
# You might need to parse the output manually or use a different approach

review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""

# Get the result
result = chain.invoke({"review": review_text})
print(result)

# You would then need to parse this result into your desired JSON structure