import random
 
def calculate(num1:float, num2:float, operator:str):
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
docs_calculate="""
This is a function to add, subtract, multiply, and divide

Args:
    num1(int):The first operand.
    num2(int):The Second operand.
    operator(str):Type of operation.
"""


 
def monte_carlo_pi(samples:int):
    inside = 0
    for _ in range(samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            inside += 1
    return 4 * inside / samples
docs_monte_carlo_pi="""
Monte Carlo method for calculating PI

Args:
    samples(int):Number of samples.
"""