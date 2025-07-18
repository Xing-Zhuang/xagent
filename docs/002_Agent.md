 
## **Custom tool**
Your can write tool by yourself,for example:

```python
from xagent.agent import Agent 
from xagent.models import DashScope


def my_calculate(num1:float, num2:float, operator:str):
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 != 0:
            return num1 / num2
        else:
            return "Error: Division by zero"
    else:
        return "Error: Invalid operator"

docs_my_calculate="""
This is a function to add, subtract, multiply, and divide

Args:
    num1(int):The first operand.
    num2(int):The Second operand.
    operator(str):Type of operation.
"""



 
agent =  Agent(
    name = "cal_agent",
    model = DashScope(model_id='qwen-max'),  
    tools = [[calculate,docs_calculate]],
)

agent.run("999/333=?")
```

 	
