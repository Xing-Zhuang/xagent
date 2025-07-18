 
 
# Install
 
 
```python
pip install -r requirements.txt
```



```python
export PYTHONPATH={path}/xagent/src
```

## **Quickstart**

```python
export DashScope_API_KEY="YOUR_API_KEY"

from xagent.agent import Agent 
from xagent.models import DashScope
from xagent.tools import calculate,docs_calculate


agent =  Agent(
    name = "cal_agent",
    model = DashScope(model_id='qwen-max'),  
    tools = [[calculate,docs_calculate]],
)

agent.run("999/333=?")
```

 	


              