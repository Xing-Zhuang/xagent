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
    
    
    def _split_into_batches(self,data_list, batch_size):
            """
            Splits a list into batches of specified size.

            :param data_list: The list to be split.
            :param batch_size: The size of each batch.
            :return: A list of lists, where each inner list is a batch of the original list.
            """
            if not isinstance(data_list, list):
                raise ValueError("data_list must be a list.")
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")

            return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

    def encode(
        self,
        data:List[BaseData],
        backend:str="default",
        
        device="cpu",
        batchsize:int=10
    ):
        
        if backend == "ray":
            import ray
            ray.init()

            @ray.remote
            def _encode(batch,model_name,device):
                embedding_model= sentence_transformers.SentenceTransformer(model_name).to(device)
                return embedding_model.encode(batch)
             
           

            metadata = [d.get_metadata() for d in data]
            content = [d.get_content() for d in data]
            batch_list = self._split_into_batches(content,batchsize)

            num_gpus = 0
            if(device=="gpu"):
                num_gpus = 1

            embedding_vec = []
            futures = []
            for batch in batch_list:
                futures.append(_encode.options(num_cpus=1, num_gpus=num_gpus).remote(batch,self.model_name,device))

            for data in futures:
                embedding_vec.extend(ray.get(data))
                
         
            assert(len(content)==len(embedding_vec))
            return EmbeddingData(metadata,content,embedding_vec)

        else:
            metadata = [d.get_metadata() for d in data]
            content = [d.get_content() for d in data]
            embedding_vec = self.get_embedding_model().encode(content)
            assert(len(content)==len(embedding_vec))
            return EmbeddingData(metadata,content,embedding_vec)
        