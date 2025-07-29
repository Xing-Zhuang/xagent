import sentence_transformers
from typing import List
from xagent.rag.data import BaseData
from chromadb import Documents, Embeddings


# class EmbeddingModel:
#     def __init__(
#         self,
#         model_name:str,
#         cache_dir:str=None,
#         **kwargs
#     ):
#         self.model_name=model_name
    
#         self.embedding_model= sentence_transformers.SentenceTransformer(  
#             model_name, cache_folder=cache_dir,**kwargs
#         )
    
#     def __call__(self, input: Documents) -> Embeddings:
#         return  [self.embedding_model.encode(text) for text in input]

#     def encode(
#         self,
#         data:List[BaseData]
#     ):
#         content = [d.get_content() for d in data]
#         return self.embedding_model.encode(content)

class EmbeddingData:
    def __init__(self,
        metadata:List,
        content:List,
        embedding_vec:List
    ):
        self.metadata=metadata
        self.content=content
        self.embedding_vec=embedding_vec
        
     

class EmbeddingModel:
    def __init__(
        self,
        model_name:str,
        cache_dir:str=None,
        **kwargs
    ):
        self.model_name=model_name
        self.cache_dir=cache_dir
        self.kwargs=kwargs
        self.embedding_model = None  #lazy load
    
    def get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model= sentence_transformers.SentenceTransformer(  
                self.model_name, cache_folder=self.cache_dir,**self.kwargs
            )
        return self.embedding_model

    def __call__(self, input: Documents) -> Embeddings:
        embedding_model = self.get_embedding_model()        
        return  [embedding_model.encode(text) for text in input]
    
    def encode(
        self,
        data:List[BaseData],
        backend:str="default"
    ):
        if backend == "ray":
            pass
        else:
            metadata = [d.get_metadata() for d in data]
            content = [d.get_content() for d in data]
            embedding_vec = self.get_embedding_model().encode(content)
            assert(len(content)==len(embedding_vec))
            return EmbeddingData(metadata,content,embedding_vec)
        