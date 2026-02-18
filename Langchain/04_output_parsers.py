# =============================================================================
# FILE: 04_output_parsers.py
# PART: 1 - Foundations  |  LEVEL: Beginner
# =============================================================================
#
# THE STORY:
#   The LLM speaks in plain text. But your app needs a Python dict,
#   a typed Pydantic object, or a list of items.
#
#   Output Parsers are translators that turn the LLM's words into
#   data structures your code can actually use.
#
#   Imagine you ask the LLM: "Give me a person's details in JSON."
#   Without a parser, you get a string like '{"name": "Alice", "age": 30}'.
#   With a parser, you get a Python dict: {"name": "Alice", "age": 30}
#   With Pydantic, you get a typed object: person.name, person.age
#
# WHAT YOU WILL LEARN:
#   1. StrOutputParser — the simplest parser (you already saw this)
#   2. JsonOutputParser — LLM output → Python dict
#   3. PydanticOutputParser — LLM output → validated, typed Pydantic object
#   4. CommaSeparatedListOutputParser — LLM output → Python list
#   5. get_format_instructions() — inject parser instructions into the prompt
#
# HOW THIS CONNECTS:
#   Previous: 03_lcel_first_chain.py — first chain with StrOutputParser
#   Next:     05_few_shot_prompting.py — teaching the LLM by example
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.3,
)

print("=" * 60)
print("  CHAPTER 4: Output Parsers")
print("=" * 60)

# =============================================================================
# SECTION 1: StrOutputParser (revisited)
# =============================================================================
# You've seen this. It's the simplest parser — just extracts .content
# from an AIMessage and returns it as a plain Python string.

print("\n--- Section 1: StrOutputParser (the baseline) ---")

str_chain = ChatPromptTemplate.from_template("Explain {topic} in one sentence.") | llm | StrOutputParser()

result: str = str_chain.invoke({"topic": "neural networks"})
print(f"\nResult type : {type(result).__name__}")
print(f"Result value: {result}")

# =============================================================================
# SECTION 2: JsonOutputParser — Get a Python Dict
# =============================================================================
# JsonOutputParser tells the LLM to produce valid JSON, then parses it.
# The key is get_format_instructions() — it adds instructions to your prompt
# telling the LLM exactly what format to output.

print("\n--- Section 2: JsonOutputParser ---")

# Create the parser
json_parser = JsonOutputParser()

# Build a prompt that includes format instructions from the parser
json_prompt = ChatPromptTemplate.from_template(
    "Create a fictional person with a name, age, city, and one hobby.\n"
    "Return ONLY valid JSON. {format_instructions}"
)

# Build the chain: prompt | llm | json_parser
json_chain = json_prompt | llm | json_parser

# Invoke — pass format_instructions from the parser into the prompt
result_dict: dict = json_chain.invoke({
    "format_instructions": json_parser.get_format_instructions()
})

print(f"\nResult type : {type(result_dict).__name__}")  # dict!
print(f"Result value: {result_dict}")
print(f"\nAccessing fields directly:")
print(f"  Name: {result_dict.get('name')}")
print(f"  City: {result_dict.get('city')}")

# =============================================================================
# SECTION 3: PydanticOutputParser — Typed, Validated Python Objects
# =============================================================================
# Define EXACTLY what you want using a Pydantic model.
# The parser instructs the LLM to match your schema, then validates the result.
# This is the most powerful parser — you get full IDE autocomplete.

print("\n--- Section 3: PydanticOutputParser ---")

# Define the data structure you expect the LLM to fill
class BookReview(BaseModel):
    title: str = Field(description="The title of the book")
    author: str = Field(description="The author's full name")
    rating: float = Field(description="Rating out of 10 (e.g. 8.5)")
    summary: str = Field(description="A one-sentence summary of the book")
    genres: List[str] = Field(description="List of genre tags (e.g. ['fiction', 'drama'])")

# Create parser with our Pydantic model
pydantic_parser = PydanticOutputParser(pydantic_object=BookReview)

# Build the prompt — include format instructions from the parser
pydantic_prompt = ChatPromptTemplate.from_template(
    "Give me a book review for any famous novel.\n\n"
    "{format_instructions}"
)

pydantic_chain = pydantic_prompt | llm | pydantic_parser

result_obj: BookReview = pydantic_chain.invoke({
    "format_instructions": pydantic_parser.get_format_instructions()
})

print(f"\nResult type      : {type(result_obj).__name__}")  # BookReview!
print(f"Title            : {result_obj.title}")
print(f"Author           : {result_obj.author}")
print(f"Rating           : {result_obj.rating}/10")
print(f"Summary          : {result_obj.summary}")
print(f"Genres           : {result_obj.genres}")
print(f"\nFull object      : {result_obj}")

# =============================================================================
# SECTION 4: CommaSeparatedListOutputParser — Python List
# =============================================================================
# Sometimes you just want a list. The LLM outputs "item1, item2, item3"
# and this parser splits it into ["item1", "item2", "item3"].

print("\n--- Section 4: CommaSeparatedListOutputParser ---")

list_parser = CommaSeparatedListOutputParser()

list_prompt = ChatPromptTemplate.from_template(
    "List 5 famous programming languages, comma-separated, no numbering.\n"
    "{format_instructions}"
)

list_chain = list_prompt | llm | list_parser

result_list: list = list_chain.invoke({
    "format_instructions": list_parser.get_format_instructions()
})

print(f"\nResult type : {type(result_list).__name__}")  # list!
print(f"Result value: {result_list}")
print(f"Count       : {len(result_list)} items")

# =============================================================================
# SECTION 5: UNDERSTANDING get_format_instructions()
# =============================================================================
# This method generates the text that gets injected into your prompt to
# tell the LLM exactly how to format its output.
# Let's print what it actually says!

print("\n--- Section 5: What does get_format_instructions() say? ---")

print(f"\nJsonOutputParser format instructions:")
print(f"  {json_parser.get_format_instructions()}")

print(f"\nCommaSeparatedListOutputParser format instructions:")
print(f"  {list_parser.get_format_instructions()}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 4")
print("=" * 60)
print("""
  1. StrOutputParser     → plain Python string
  2. JsonOutputParser    → Python dict (great for flexible structures)
  3. PydanticOutputParser → validated, typed Pydantic object (best for APIs)
  4. CommaSeparatedListOutputParser → Python list
  5. get_format_instructions() injects format guidance into the prompt
  6. Always inject format_instructions as a variable in your prompt template

  Next up: 05_few_shot_prompting.py
  Teach the LLM by example — show it worked examples before asking your question.
""")
