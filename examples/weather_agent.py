
from xagent.agent import Agent 
from xagent.models import DashScope
from agenxagenttos.tools import get_weather,docs_get_weather
import os

os.environ["AMAP_API_KEY"] = "YOUR_API_KEY"


agent =  Agent(
    name = "weather_agent",
    model = DashScope(model_id='qwen-max'),   
    tools = [[get_weather,docs_get_weather]],
    
)

agent.run("武汉今天适合外出吗？")