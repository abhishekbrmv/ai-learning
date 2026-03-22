# practical_7_agent.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()  # Load environment variables from .env file

# ── Step 1: Define tools ──
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def get_word_length(word: str) -> int:
    """Get the number of characters in a word."""
    return len(word)

tools = [add_numbers, multiply_numbers, get_word_length]

# ── Step 2: Create the agent ──
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_react_agent(llm, tools)

# ── Step 3: Run it ──
questions = [
    "What is 25 multiplied by 4?",
    "How many characters are in the word 'artificial'?",
    "Add 150 and 230, then tell me the result."
]

for q in questions:
    print(f"\nQuestion: {q}")
    result = agent.invoke({"messages": [("human", q)]})
    print(f"Answer: {result['messages'][-1].content}")
    print("-" * 40)