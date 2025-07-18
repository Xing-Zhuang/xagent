 
import pandas as pd

 
def get_csv_col(file_path:str):
    import csv

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        column_names = next(reader)
    return ', '.join(column_names)
docs_get_csv_col="""
Get the csv file columns name.

Args:
    file_path(str):the csv file path.
"""


 
def group_and_count(file_path:str, column_name:str):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 检查列是否存在
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the CSV file.")
    
    # 按指定列分组并计数
    counts = df[column_name].value_counts()
    
    return counts
docs_group_and_count="""
Group and count by any specified column.

Args:
    file_path(str):the csv file path.
    column_name(str):the column name.
"""

 
 
def get_max_value(file_path:str, column_name:str):
    df = pd.read_csv(file_path)
    return df[column_name].max()
docs_get_max_value="""
Get the max value of column.

Args:
    file_path(str):the csv file path.
    column_name(str):the column name.
"""

 
def get_min_value(file_path:str, column_name:str):
    df = pd.read_csv(file_path)
    return df[column_name].min()
docs_get_min_value="""
Get the min value of column.

Args:
    file_path(str):the csv file path.
    column_name(str):the column name.
"""


 
def get_mean_value(file_path, column_name):
    df = pd.read_csv(file_path)
    return df[column_name].mean()
docs_get_mean_value="""
Get the mean value of column.

Args:
    file_path(str):the csv file path.
    column_name(str):the column name.
"""