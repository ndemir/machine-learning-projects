ES_NODES = "http://localhost:9200"

documents = [
    {"id": 1, "text": "How to start with Python programming.", "vector": [0.1, 0.2, 0.3]},
    {"id": 2, "text": "Advanced Python programming tips.", "vector": [0.1, 0.3, 0.4]},
    # More documents...
]

from elasticsearch import Elasticsearch

es = Elasticsearch(
    hosts=ES_NODES,
)
for doc in documents:
    es.index(index="documents", id=doc['id'], document={"text": doc['text']})

import numpy as np
import faiss

dimension = 3  # Assuming 3D vectors for simplicity
faiss_index = faiss.IndexFlatL2(dimension)
vectors = np.array([doc['vector'] for doc in documents])
faiss_index.add(vectors)

def hybrid_search(query_text, query_vector, alpha=0.5):
    # Perform a keyword search using Elasticsearch on the "documents" index, matching the provided query_text.
    response = es.search(index="documents", query={"match": {"text": query_text}})
    # Extract the document IDs and their corresponding scores from the Elasticsearch response.
    keyword_results = {hit['_id']: hit['_score'] for hit in response['hits']['hits']}

    # Prepare the query vector for vector search: reshape and cast to float32 for compatibility with Faiss.
    query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
    # Perform a vector search with Faiss, retrieving indices of the top 5 closest documents.
    _, indices = faiss_index.search(query_vector, 5)
    # Create a dictionary of vector results with scores inversely proportional to their rank (higher rank, higher score).
    vector_results = {str(documents[idx]['id']): 1/(rank+1) for rank, idx in enumerate(indices[0])}

    # Initialize a dictionary to hold combined scores from keyword and vector search results.
    combined_scores = {}
    # Iterate over the union of document IDs from both keyword and vector results.
    for doc_id in set(keyword_results.keys()).union(vector_results.keys()):
        # Calculate combined score for each document using the alpha parameter to balance the influence of both search results.
        combined_scores[doc_id] = alpha * keyword_results.get(doc_id, 0) + (1 - alpha) * vector_results.get(doc_id, 0)

    # Return the dictionary containing combined scores for all relevant documents.
    return combined_scores

# Example usage
query_text = "Python programming"
query_vector = [0.1, 0.25, 0.35]
# Execute the hybrid search function with the specified query text and vector.
results = hybrid_search(query_text, query_vector)
# Print the results of the hybrid search to see the combined scores of documents.
print(results)

