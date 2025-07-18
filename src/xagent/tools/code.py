import arxiv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
 
def exe_code(code: str):
    try:
        exec(code)
    except Exception as e:
        print(f"Error executing code: {e}")
docs_exe_code = """
This is a function for executing code. 

Args:
    code(str):The code you want to execute.
"""


