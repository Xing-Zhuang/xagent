 
 
# Install
```python
pip install -r requirements.txt
```

 

```python
export PYTHONPATH={PATH}/xagent/src
```

## **Quickstart**

 
```python
import os
from xagent.agent import Agent 
from xagent.models import DashScope
from xagent.tools import calculate,docs_calculate

os.environ["DashScope_API_KEY"] = "YOUR_API_KEY" # 如果您已经提前将api-key提前配置到您的运行环境中，可以省略这个步骤

agent =  Agent(
    name = "cal_agent",
    model = DashScope(model_id='qwen-max'),  
    tools = [[calculate,docs_calculate]],
)

agent.run("999/333=?")
```

For more, please refer to the "docs/" folder.
 	


              