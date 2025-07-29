import uuid
import warnings
import chromadb
from xagent.rag.data import BaseData,PdfData,TextData,JsonData,CsvData,merge_content
from xagent.rag.rerank import *
from xagent.rag.embedding import *
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from uuid import uuid4

class ChromaDB():
    
    def __init__(
        self,
        chroma_client:ClientAPI,
        collection:Collection,
        embedding_model:EmbeddingModel
    ):
        self.chroma_client=chroma_client
        self.collection=collection
        self.embedding_model=embedding_model


    @classmethod
    def create_document(  # This method is to create a new chroma db
        cls,
        dir:str,
        collection_name:str,
        data:EmbeddingData
    ):
        chroma_client = chromadb.PersistentClient(path=dir)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
 
        ids = [str(uuid4()) for _ in data.content]
        assert(len(ids)==len(data.content))
        assert(len(ids)==len(data.embedding_vec))
        assert(len(ids)==len(data.metadata))
        collection.add(
            ids=ids,
            documents=data.content,
            embeddings=data.embedding_vec,
            metadatas=data.metadata
        )
        print("存储成功，collection当前记录数:", collection.count())


    @classmethod
    def load_document(
        cls,
        embedding_model:EmbeddingModel,
        collection_name:str,
        dir:str,
    ):
        chroma_client = chromadb.PersistentClient(path=dir) 
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_model)
           
        return cls(
            chroma_client,
            collection,
            embedding_model,
        )


    def query(
        self,
        query_text:str,
        query_num:int=10,
        reranker=None
    )->List[BaseData]:
        query_data=self.collection.query(query_texts=query_text,n_results=query_num)
        
        results=[]
        for i in range(len(query_data['metadatas'][0])):
            content=query_data['documents'][0][i]
            metadata=query_data['metadatas'][0][i]
            results.append(BaseData(content,metadata))

        if reranker is not None:
             
            passages=[]
            for i in range(len(results)):
                passages.append(results[i].get_content())
            reranker_res=reranker.rerank(query_text,passages)
 
            for i in range(len(reranker_res)):
                id=reranker_res[i]['corpus_id']
                content=reranker_res[i]['text']
                results[id].set_content(content)
            
        return results
      
  