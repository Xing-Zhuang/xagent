import arxiv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

  
 
def extract_keywords(query: str, max_results: int,topk:int):
    try:
        # 从arXiv搜索论文
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # 构建语料库
        corpus = []
        for result in search.results():
            corpus.append(f"{result.title}\n{result.summary}")
        
        if not corpus:
            return "未找到相关论文！"
        

        # TF-IDF计算
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # 提取TOP20关键词
        feature_names = vectorizer.get_feature_names_out()
        top_indices = np.argsort(tfidf_matrix.sum(axis=0))[0, -topk:]
        top_keywords = [feature_names[i] for i in top_indices]
        
        return top_keywords
    
    except Exception as e:
        return f"发生错误：{e}"
    
docs_extract_keywords = """
This is a arxiv tool for extracting keywords. 

Args:
    query(str):The paper topic you want to search for.
    max_results(str):You want to search for the number of papers.
    topk(str):The k keywords with the highest frequency are extracted.
"""


