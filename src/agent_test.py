from dotenv import load_dotenv
load_dotenv()

import ast
import operator as op

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from llm_smoke_test import make_llm

# Safe expression evaluator: + - * / ** and parentheses, numbers only
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression with + - * / ** and parentheses (numbers only)."""
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))

if __name__ == "__main__":
    llm = make_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when calculation is needed. Otherwise respond normally."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [calculator], prompt)
    executor = AgentExecutor(agent=agent, tools=[calculator], verbose=True)

    result = executor.invoke({"input": "What is 42 * 17? Then explain in one sentence what you did."})
    print(result["output"])
