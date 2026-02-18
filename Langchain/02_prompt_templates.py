# =============================================================================
# FILE: 02_prompt_templates.py
# PART: 1 - Foundations  |  LEVEL: Beginner
# =============================================================================
#
# THE STORY:
#   Prompts are your molds. You pour different ingredients in,
#   and out comes a perfectly shaped message every time.
#
#   Hardcoding prompts ("Translate Hello to French") is fine for one use.
#   But what if you need to translate 1000 sentences to 10 different languages?
#   That's where PromptTemplates shine — define the shape once, fill it forever.
#
# WHAT YOU WILL LEARN:
#   1. PromptTemplate — simple string templates with {variables}
#   2. ChatPromptTemplate — multi-message templates for chat models
#   3. MessagesPlaceholder — inject a list of messages (conversation history)
#   4. Partial templates — pre-fill one variable, leave others open
#   5. .invoke() vs .format_messages() — two ways to use a template
#
# HOW THIS CONNECTS:
#   Previous: 01_llm_and_chat_models.py — messages and model calls
#   Next:     03_lcel_first_chain.py — chaining prompt | llm | parser
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

print("=" * 60)
print("  CHAPTER 2: Prompt Templates")
print("=" * 60)

# =============================================================================
# SECTION 1: PromptTemplate — The Simplest Template
# =============================================================================
# PromptTemplate works with plain string prompts (not chat-style).
# Variables are written as {variable_name} inside the template string.
# .format() fills in the variables and returns a plain string.
# .invoke() fills in variables and returns a PromptValue object.

print("\n--- Section 1: PromptTemplate ---")

# Define the template with placeholders
translator_template = PromptTemplate.from_template(
    "Translate the following text into {language}:\n\n\"{text}\""
)

# Fill in the variables
formatted_prompt = translator_template.format(language="Spanish", text="Hello, how are you?")
print(f"\nFormatted prompt string:\n  {formatted_prompt}")

# .invoke() returns a PromptValue (richer — can convert to string or messages)
prompt_value = translator_template.invoke({"language": "French", "text": "Good morning"})
print(f"\nPromptValue type: {type(prompt_value).__name__}")
print(f"As string: {prompt_value.to_string()}")

# =============================================================================
# SECTION 2: ChatPromptTemplate — Multi-Message Templates
# =============================================================================
# ChatPromptTemplate is the one you'll use 90% of the time.
# It creates a list of messages (System + Human + AI) from templates.
# .from_messages() takes a list of (role, template) tuples.
#   role = "system", "human", or "ai"

print("\n--- Section 2: ChatPromptTemplate ---")

# A customer support chat template
support_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support agent for {company}. Be polite and concise."),
    ("human", "{customer_question}"),
])

# .format_messages() returns a list of LangChain message objects
messages = support_prompt.format_messages(
    company="TechCorp",
    customer_question="My account is locked. What do I do?"
)

print(f"\nNumber of messages generated: {len(messages)}")
for msg in messages:
    print(f"  [{msg.type.upper()}]: {msg.content}")

# Now use it with the LLM
response = llm.invoke(messages)
print(f"\nLLM Response: {response.content}")

# =============================================================================
# SECTION 3: MessagesPlaceholder — Injecting a List of Messages
# =============================================================================
# Sometimes you have an existing list of messages (like conversation history)
# and you want to inject it into your prompt at a specific position.
# MessagesPlaceholder is the slot where that list gets inserted.

print("\n--- Section 3: MessagesPlaceholder (for conversation history) ---")

# This template leaves a {chat_history} slot for injecting past messages
history_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the conversation history to answer."),
    MessagesPlaceholder(variable_name="chat_history"),  # History goes here
    ("human", "{current_question}"),                    # Latest question
])

# Simulate an existing conversation history
fake_history = [
    HumanMessage(content="My name is Dinesh."),
    AIMessage(content="Nice to meet you, Dinesh! How can I help?"),
    HumanMessage(content="I work in AI automation."),
    AIMessage(content="That's fascinating! Automation is a growing field."),
]

# Fill template with history and a new question
filled_messages = history_prompt.format_messages(
    chat_history=fake_history,
    current_question="What do you know about me so far?"
)

print(f"\nTotal messages in prompt (system + 4 history + 1 human): {len(filled_messages)}")
response = llm.invoke(filled_messages)
print(f"\nLLM Response (with history context): {response.content}")

# =============================================================================
# SECTION 4: Partial Templates — Pre-Fill Some Variables
# =============================================================================
# What if you always translate TO French, but the input text varies?
# .partial() lets you fix some variables, leaving others open for later.

print("\n--- Section 4: Partial Templates ---")

general_template = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate everything into {language}."),
    ("human", "{text}"),
])

# Pre-fill the language — now this template only needs "text"
french_translator = general_template.partial(language="French")
spanish_translator = general_template.partial(language="Spanish")

# Each specialized translator only needs the text
print("\nFrench translator response:")
french_response = llm.invoke(french_translator.format_messages(text="The sky is blue."))
print(f"  {french_response.content}")

print("\nSpanish translator response:")
spanish_response = llm.invoke(spanish_translator.format_messages(text="The sky is blue."))
print(f"  {spanish_response.content}")

# =============================================================================
# SECTION 5: .invoke() is the Chain-Ready Way
# =============================================================================
# When building LCEL chains (next file!), you use .invoke() not .format_messages()
# The difference: .invoke() returns a PromptValue that works inside | chains.
# .format_messages() returns a plain Python list — not chain-compatible.

print("\n--- Section 5: .invoke() vs .format_messages() ---")

simple_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {number} squared?")
])

# .invoke() — use this in chains (LCEL)
prompt_value = simple_prompt.invoke({"number": 7})
print(f"\n.invoke() returns: {type(prompt_value).__name__}")
print(f"  .to_messages(): {[m.content for m in prompt_value.to_messages()]}")

# .format_messages() — use this when you need a plain list
msg_list = simple_prompt.format_messages(number=7)
print(f"\n.format_messages() returns: {type(msg_list).__name__} of {len(msg_list)} message(s)")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 2")
print("=" * 60)
print("""
  1. PromptTemplate — string templates with {variable} placeholders
  2. ChatPromptTemplate — multi-message templates (system/human/ai)
  3. MessagesPlaceholder — insert a dynamic list of messages at a slot
  4. .partial() — pre-fill some variables, leave others for later
  5. .invoke() is chain-compatible; .format_messages() returns a plain list
  6. Templates make prompts reusable, testable, and version-controllable

  Next up: 03_lcel_first_chain.py
  We'll chain prompt | llm | output_parser using the | operator.
""")
