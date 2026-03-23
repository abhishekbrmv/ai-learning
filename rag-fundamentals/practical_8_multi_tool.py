# practical_8_multi_tool.py

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import json
load_dotenv()
# ── Tools ──

@tool
def get_employee(name: str) -> str:
    """Get employee details by name."""
    employees = {
        "abhishek": {"role": "Rails Developer", "salary": 80000, "department": "Engineering"},
        "priya": {"role": "Designer", "salary": 70000, "department": "Product"},
        "rahul": {"role": "Data Scientist", "salary": 95000, "department": "AI"},
    }
    employee = employees.get(name.lower())
    if not employee:
        return f"Employee '{name}' not found"
    return json.dumps(employee)

@tool
def get_department_budget(department: str) -> str:
    """Get the total budget allocated for a department."""
    budgets = {
        "Engineering": 500000,
        "Product": 300000,
        "AI": 600000,
    }
    budget = budgets.get(department)
    if not budget:
        return f"Department '{department}' not found"
    return f"{department} department budget is ${budget}"

@tool
def calculate_budget_remaining(total_budget: float, salary: float) -> str:
    """Calculate remaining budget after subtracting an employee salary."""
    remaining = total_budget - salary
    return f"Remaining budget after salary deduction: ${remaining}"

@tool
def send_notification(recipient: str, message: str) -> str:
    """Send a notification message to a recipient."""
    return f"Notification sent to {recipient}: '{message}'"

tools = [get_employee, get_department_budget, calculate_budget_remaining, send_notification]

# ── Agent ──
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "hr_session"}}

# ── Questions that require chaining multiple tools ──
questions = [
    "What is Abhishek's role and salary?",
    "What is the budget for his department and how much remains after his salary?",
    "Send him a notification that his appraisal is due next week.",
]

for q in questions:
    print(f"\nQuestion: {q}")
    result = agent.invoke({"messages": [("human", q)]}, config=config)
    print(f"Answer: {result['messages'][-1].content}")
    print("-" * 50)