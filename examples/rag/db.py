from alayalite import Client

from typing import Callable, Generator, Tuple

from utils import splitter, embedder


client = Client()

def reset_db():
    global client 
    client.reset()

def insert_text(collection_name: str, 
                docs: str, 
                embed_model_path: str, 
                chunksize: int = 256, 
                overlap: int = 25) -> bool:

    global client 

    chunks = splitter(docs, chunksize, overlap)
    print(f'Splitting text into {len(chunks)} chunks')
    
    embeddings = embedder(chunks, embed_model_path)
    print(f'Embedding {len(chunks)} chunks into vectors')

    if len(embeddings) == 0:
        print('Fail to embed chunks. Not to insert')
        return False
    
    print(f'Inserting {len(chunks)} chunks')
    try:
        collection = client.get_or_create_collection(collection_name)
        items = []      # List of (id, document, embedding, metadata)
        for i in range(0, len(chunks)):
            items.append((str(i), chunks[i], embeddings[i], None))
        collection.insert(items)
    except Exception as e:
        print(f"Error during index creation: {e}")
        return False
    print(f'Insertion done!')

    return True     # success


def query_text(collection_name: str, embed_model_path: str, query: str, top_k = 5) -> str:

    global client

    try:
        collection = client.get_collection(collection_name)
        if collection:
            processed_query = embedder([query], embed_model_path)
            # return type: DataFrame[id, document, distance, metadata]
            # use result[field_name][0] to get the column data
            result = collection.batch_query(processed_query, top_k)
            retrieved_docs: str = '\n\n'.join(result['document'][0])
        else:
            retrieved_docs: str = ''
    except Exception as e:
        print(f"Error during retrieval: {e}")
        retrieved_docs = ''
    
    return retrieved_docs

if __name__ == "__main__":
    from llm import ask_llm

    with open("test_docs.txt", "r") as fp:
        sample_text = fp.read()
    query = "What are higher-order chunking techniques?"

    llm_url = 'Your LLM service base URL here'
    llm_api_key = 'Your API key here'
    llm_model = 'deepseek-v3'
    embed_model_path = "BAAI/bge-small-zh-v1.5"

    insert_text(collection_name="test", embed_model_path=embed_model_path, docs=sample_text, chunksize=128)
    retrieved_docs = query_text(collection_name="test", embed_model_path=embed_model_path, query=query, top_k=5)
    result = ask_llm(llm_url, llm_api_key, llm_model, query, retrieved_docs, is_stream=False)
    print(f'=== Response ===\n{result}')
