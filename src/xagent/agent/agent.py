from xagent.memory import TemporaryMemory,Message,Role
from xagent.prompt import DEFAULT_PROMPT
import re
from typing import List

import time
import requests
import json

from xagent.models.local_model import LocalModel
 
class Agent:
    def __init__(
        self,
        model,
        tools:List[List],
        name:str=None,
        prompt:str=None
    ):
        self.name = name   
        self.model = model
    
        def extract_info(docs):
            # Split the string into lines
            lines = docs.strip().split('\n')
            
            # Extract the description (first line after the initial empty one)
            desc = lines[0].strip()
            
            # Initialize parameters dictionary
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            # Process each line to extract parameter details
            for line in lines[1:]:
                match = re.match(r'\s*(\w+)\(([\w\s]+)\):\s*(.+)', line)
                if match:
                    param_name, param_type, param_desc = match.groups()
                    parameters["properties"][param_name] = {
                        "type": param_type.strip(),
                        "description": param_desc.strip()
                    }
                    parameters["required"].append(param_name)
            
            return desc, parameters

        def extract_name(full_string):
            parts = full_string.split('.')
            name = parts[-1]
            return name
        
        self.tools_map = {}
        self.tools=[]
        for obj in tools:
            if hasattr(obj[0], '__name__'):
                tool_name = obj[0].__name__
            else:
                tool_name = extract_name(obj[0]._function_name)
            tool_desc, parameters = extract_info(obj[1])
            
            info = {
                "type": "function",
                "function": {
                    "name": tool_name, 
                    "description": tool_desc,  
                    "parameters": parameters,
                }
            }
            self.tools.append(info)
            self.tools_map[tool_name] = obj[0]

        #print(self.tools) 
    
        self.memory = TemporaryMemory()
        if prompt:
            self.memory.add_memory(Message(Role.SYSTEM,prompt))
        else:
            self.memory.add_memory(Message(Role.SYSTEM,DEFAULT_PROMPT))
        
     
        #print(self.memory.memory[0]['content'])

    def submit_task(
        self,
        task_name:str,
        **args,
    ):  
         
        url = f"http://{self.server_addr}/task/"
        payload = {
            "task_name": task_name,
            **args,
        }

        response = requests.post(url, json=payload)
        if(task_name=="query_api_model"):
            return response.json()

        return response.json()['res']
    
    def query_model(self,query):
        pass


    def run(
        self,
        task:str,
        max_turn:int = 10
    ):
        self.memory.add_memory(Message(Role.USER,task))
        
        if isinstance(self.model, LocalModel):
            response = self.submit_task(
                task_name="query_local_model",
                model_id=self.model.model_id,
                messages=self.memory.memory,
                name = self.name 
            )

            #todo
        else:
            while(max_turn):
                response =  self.model.reply(messages=self.memory.memory,tools=self.tools)
                
               
                message = response["output"]["choices"][0]["message"]
                
                 
                if("tool_calls" in message):
                    print("-"*50)
                    print("Call Tools...")
                    call_res = ""
                    for tool in response["output"]["choices"][0]["message"]["tool_calls"]:
                        tool_name = tool['function']['name']
                        print(f"🔧 {tool_name}")
                        tool_args = json.loads(tool['function']['arguments'])
                        call_tool_res = self.tools_map[tool_name](**tool_args)
                        if not isinstance(call_tool_res,str):
                            call_tool_res = str(call_tool_res)
                       

                        call_res = call_res + "Action:" + tool_name + "\n"
                        call_res = call_res + "Observation:" + call_tool_res + "\n"
                    self.memory.add_memory(Message(Role.USER,call_res))
                    print("\n")

                else:        
                    answer = response["output"]["choices"][0]["message"]["content"]
                    self.memory.add_memory(Message(Role.ASSISTANT,answer))
                    print("-"*50)
                    print(f"{answer}")
                    break

                max_turn = max_turn-1

 
 

 
           

           
             






 