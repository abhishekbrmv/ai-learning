# practical_9_langgraph.py

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import operator
import json
load_dotenv()

# ── State ──
class AgentState(TypedDict):
    question: str
    employee: dict
    department_budget: float
    remaining_budget: float
    final_answer: str

# ── Tools (plain functions, no @tool decorator needed) ──
def get_employee(name: str) -> dict:
    employees = {
        "abhishek": {"role": "Rails Developer", "salary": 80000, "department": "Engineering"},
        "priya":    {"role": "Designer", "salary": 70000, "department": "Product"},
        "rahul":    {"role": "Data Scientist", "salary": 95000, "department": "AI"},
    }
    return employees.get(name.lower(), {})

def get_department_budget(department: str) -> float:
    budgets = {"Engineering": 500000, "Product": 300000, "AI": 600000}
    return budgets.get(department, 0)

# ── Nodes (each node is one step in the flow) ──
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def node_get_employee(state: AgentState) -> AgentState:
    print(">> Node 1: fetching employee...")
    name = state["question"].split("about")[-1].strip().rstrip("?")
    employee = get_employee(name)
    print(f"   found: {employee}")
    return {"employee": employee}

def node_get_budget(state: AgentState) -> AgentState:
    print(">> Node 2: fetching department budget...")
    department = state["employee"]["department"]
    budget = get_department_budget(department)
    print(f"   budget: ${budget}")
    return {"department_budget": budget}

def node_calculate(state: AgentState) -> AgentState:
    print(">> Node 3: calculating remaining budget...")
    remaining = state["department_budget"] - state["employee"]["salary"]
    print(f"   remaining: ${remaining}")
    return {"remaining_budget": remaining}

def node_answer(state: AgentState) -> AgentState:
    print(">> Node 4: generating final answer...")
    prompt = f"""
    Employee: {state['employee']}
    Department budget: ${state['department_budget']}
    Remaining budget after salary: ${state['remaining_budget']}
    
    Give a clear one paragraph summary of this information.
    """
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# ── Build the graph ──
graph = StateGraph(AgentState)

graph.add_node("get_employee", node_get_employee)
graph.add_node("get_budget",   node_get_budget)
graph.add_node("calculate",    node_calculate)
graph.add_node("answer",       node_answer)

# ── Define the flow (edges) ──
graph.set_entry_point("get_employee")
graph.add_edge("get_employee", "get_budget")
graph.add_edge("get_budget",   "calculate")
graph.add_edge("calculate",    "answer")
graph.add_edge("answer",       END)

# ── Compile and run ──
app = graph.compile()

result = app.invoke({
    "question": "Tell me about Abhishek",
    "employee": {},
    "department_budget": 0,
    "remaining_budget": 0,
    "final_answer": ""
})

print(f"\nFinal Answer:\n{result['final_answer']}")