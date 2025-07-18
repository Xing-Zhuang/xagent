import sys
import os
import dashscope
from typing import List
from http import HTTPStatus


class DashScope:
    def __init__(self,model_id:str):
        self.model_id = model_id

        if "DashScope_API_KEY" not in os.environ:
            print("Please set 'DashScope_API_KEY'") 
            sys.exit(1)
    
 
    def reply(self,messages:List,tools:List):
        response = dashscope.Generation.call(
            model=self.model_id,
            messages=messages,
            api_key=os.environ["DashScope_API_KEY"],
            result_format='message',  
            tools=tools,
        )

        if response.status_code == HTTPStatus.OK:
            return response
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            sys.exit(1)
 