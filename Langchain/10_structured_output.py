# =============================================================================
# FILE: 10_structured_output.py
# PART: 2 - LCEL Deep Dive  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   Remember injecting {format_instructions} into every prompt?
#   Hoping the LLM would follow the format? Dealing with OutputFixingParser?
#
#   Modern LLMs have a better way: native structured output.
#   Instead of "please format as JSON", you hand the model a SCHEMA.
#   The model's API enforces the schema at the generation level.
#   No more format instructions. No more parsing failures.
#
#   llm.with_structured_output(YourSchema) is the cleanest way to get
#   typed, validated output from a language model.
#
# WHAT YOU WILL LEARN:
#   1. llm.with_structured_output(PydanticModel) — typed output
#   2. Using TypedDict as the schema (lighter than Pydantic)
#   3. Using a JSON Schema dict (most flexible)
#   4. include_raw=True — get both AIMessage and parsed result
#   5. How it works under the hood (tool calling / JSON mode)
#
# HOW THIS CONNECTS:
#   Previous: 09_output_parsers_advanced.py — parsers and OutputFixingParser
#   Next:     11_memory_in_memory.py — giving the LLM a memory of past messages
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.3,
)

print("=" * 60)
print("  CHAPTER 10: Structured Output")
print("=" * 60)

# =============================================================================
# SECTION 1: WITH PYDANTIC MODEL — Fully Typed and Validated
# =============================================================================
# Define exactly what you want using Pydantic.
# Each Field has a description — the model uses this to fill the field correctly.
# The return type is your Pydantic class. Full IDE autocomplete. Zero ambiguity.

print("\n--- Section 1: with_structured_output(PydanticModel) ---")

class MovieReview(BaseModel):
    """A structured movie review with all key components."""
    title: str = Field(description="The exact movie title")
    director: str = Field(description="The director's full name")
    release_year: int = Field(description="Year the movie was released")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    pros: List[str] = Field(description="3-4 things the reviewer liked")
    cons: List[str] = Field(description="1-2 things the reviewer disliked")
    verdict: str = Field(description="One sentence: should someone watch it?")
    recommended: bool = Field(description="True if the movie is recommended")

# Create a structured LLM — it will ALWAYS return a MovieReview object
structured_llm = llm.with_structured_output(MovieReview)

print("\nGenerating structured movie review...")
review: MovieReview = structured_llm.invoke(
    "Give me a detailed review of the movie Inception (2010) by Christopher Nolan."
)

print(f"\nTitle     : {review.title}")
print(f"Director  : {review.director}")
print(f"Year      : {review.release_year}")
print(f"Rating    : {review.rating}/10")
print(f"Pros      : {review.pros}")
print(f"Cons      : {review.cons}")
print(f"Verdict   : {review.verdict}")
print(f"Recommend : {'Yes!' if review.recommended else 'No'}")

# Full type safety — IDE knows the exact type of each field
print(f"\nType of result: {type(review).__name__}")
print(f"review.rating is: {type(review.rating).__name__}")   # float
print(f"review.pros is  : {type(review.pros).__name__}")     # list

# =============================================================================
# SECTION 2: WITH TypedDict — Lighter Alternative to Pydantic
# =============================================================================
# TypedDict is simpler than Pydantic — just annotated keys, no validators.
# Returns a plain Python dict with the right keys (still typed in IDE).
# Use when you don't need Pydantic's validation logic.

print("\n--- Section 2: with_structured_output(TypedDict) ---")

class WeatherReport(TypedDict):
    """Current weather conditions for a city."""
    city: str
    temperature_celsius: float
    condition: str          # e.g. "Sunny", "Cloudy", "Rainy"
    humidity_percent: int
    wind_speed_kmh: float
    uv_index: int
    recommendation: str     # e.g. "Bring an umbrella"

structured_weather_llm = llm.with_structured_output(WeatherReport)

weather: WeatherReport = structured_weather_llm.invoke(
    "Generate a realistic weather report for Mumbai in July (monsoon season)."
)

print(f"\nCity          : {weather['city']}")
print(f"Temperature   : {weather['temperature_celsius']}°C")
print(f"Condition     : {weather['condition']}")
print(f"Humidity      : {weather['humidity_percent']}%")
print(f"Wind Speed    : {weather['wind_speed_kmh']} km/h")
print(f"UV Index      : {weather['uv_index']}")
print(f"Recommendation: {weather['recommendation']}")

# =============================================================================
# SECTION 3: WITH JSON SCHEMA DICT — Maximum Flexibility
# =============================================================================
# You can also pass a raw JSON Schema dict.
# Useful when you get the schema from an external source (API, config file).

print("\n--- Section 3: with_structured_output(JSON Schema dict) ---")

json_schema = {
    "title": "BookRecommendation",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Book title"},
        "author": {"type": "string", "description": "Author's name"},
        "genre": {"type": "string", "description": "Genre of the book"},
        "page_count": {"type": "integer", "description": "Number of pages"},
        "difficulty": {
            "type": "string",
            "enum": ["Easy", "Medium", "Hard"],
            "description": "Reading difficulty level"
        },
        "why_read": {"type": "string", "description": "One reason to read this book"},
    },
    "required": ["title", "author", "genre", "page_count", "difficulty", "why_read"],
}

schema_llm = llm.with_structured_output(json_schema)
book: dict = schema_llm.invoke("Recommend a classic science fiction novel.")

print(f"\nBook: {book.get('title')} by {book.get('author')}")
print(f"Genre     : {book.get('genre')}")
print(f"Pages     : {book.get('page_count')}")
print(f"Difficulty: {book.get('difficulty')}")
print(f"Why read  : {book.get('why_read')}")

# =============================================================================
# SECTION 4: include_raw=True — Both Raw and Parsed
# =============================================================================
# Sometimes you want BOTH the raw AIMessage AND the parsed result.
# include_raw=True returns a dict with:
#   "raw"    : the original AIMessage
#   "parsed" : your Pydantic/TypedDict object
#   "parsing_error": any error if parsing failed

print("\n--- Section 4: include_raw=True ---")

structured_llm_raw = llm.with_structured_output(MovieReview, include_raw=True)

raw_and_parsed = structured_llm_raw.invoke(
    "Review the movie Interstellar (2014)."
)

print(f"\nKeys in result: {list(raw_and_parsed.keys())}")
print(f"\nRaw type      : {type(raw_and_parsed['raw']).__name__}")
print(f"Parsed type   : {type(raw_and_parsed['parsed']).__name__}")

if raw_and_parsed["parsed"]:
    print(f"Parsed title  : {raw_and_parsed['parsed'].title}")
    print(f"Parsed rating : {raw_and_parsed['parsed'].rating}")

if raw_and_parsed["raw"]:
    print(f"\nRaw message excerpt: {str(raw_and_parsed['raw'])[:100]}...")

# =============================================================================
# SECTION 5: USING IN A CHAIN
# =============================================================================
# with_structured_output() returns a Runnable — you can use it in LCEL chains.

print("\n--- Section 5: Structured Output in a Chain ---")

class EntityExtraction(BaseModel):
    """Named entities extracted from text."""
    people: List[str] = Field(description="Full names of people mentioned")
    organizations: List[str] = Field(description="Names of companies/organizations")
    locations: List[str] = Field(description="Cities, countries, places")
    dates: List[str] = Field(description="Any dates or time periods mentioned")

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting named entities from text."),
    ("human", "Extract all named entities from this text:\n\n{text}"),
])

# Chain: prompt | structured_llm
extraction_chain = extraction_prompt | llm.with_structured_output(EntityExtraction)

sample_text = (
    "On March 15, 2023, Elon Musk announced that Tesla would open a new Gigafactory "
    "in Berlin, Germany. The factory, located near Brandenburg, is expected to produce "
    "500,000 vehicles per year. Meanwhile, Apple CEO Tim Cook met with German Chancellor "
    "Olaf Scholz in Munich to discuss supply chain partnerships."
)

entities: EntityExtraction = extraction_chain.invoke({"text": sample_text})

print(f"\nPeople        : {entities.people}")
print(f"Organizations : {entities.organizations}")
print(f"Locations     : {entities.locations}")
print(f"Dates         : {entities.dates}")

# =============================================================================
# SECTION 6: HOW IT WORKS UNDER THE HOOD
# =============================================================================
print("\n--- Section 6: How It Works Under the Hood ---")
print("""
  with_structured_output() uses TWO mechanisms depending on the model:

  1. TOOL CALLING (preferred, for OpenAI-compatible models):
     - Your schema is converted to a "tool" definition
     - The model is instructed: "call this tool with the right arguments"
     - The API returns structured JSON in the tool_calls field
     - LangChain extracts and validates it → your Pydantic object

  2. JSON MODE (fallback):
     - The model is told to output JSON
     - The output is parsed as JSON and validated against your schema

  Why is this better than PydanticOutputParser?
  - Schema is enforced at the API level (no hallucinated formats)
  - No {format_instructions} tokens needed in the prompt
  - Much lower chance of parsing errors
  - Cleaner code: no parser.get_format_instructions() calls
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 10")
print("=" * 60)
print("""
  1. llm.with_structured_output(Schema) is the modern, preferred approach
  2. Pass a Pydantic model, TypedDict, or JSON Schema dict
  3. Returns fully typed Python objects — full IDE support
  4. include_raw=True gives you both the AIMessage and the parsed object
  5. It works by tool calling or JSON mode — schema enforced at API level
  6. Use this instead of PydanticOutputParser when possible

  PART 2 COMPLETE — You now know LCEL inside and out.

  Next up: 11_memory_in_memory.py
  Giving the LLM a memory so it remembers past conversations.
""")
