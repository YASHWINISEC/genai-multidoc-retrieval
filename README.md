## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:Design and build a LlamaIndex agent for retrieving information from multiple research papers. The agent should extract and synthesize relevant details to answer user queries. Its performance will be evaluated based on the conciseness, relevance, and accuracy of its responses to diverse questions. The goal is to create an effective tool for knowledge synthesis across a collection of documents.

### DESIGN STEPS:

#### STEP 1:
Install Core Libraries: Install llama-index, llama-hub, transformers, sentence-transformers, faiss-cpu, and numpy to ensure all necessary dependencies for LlamaIndex, text embedding, and FAISS are available.
#### STEP 2:
Upload Initial Documents: Use google.colab.files.upload() to prompt the user to select and upload source text files (e.g., docs.txt, docs1.txt) into the environment.
#### STEP 3:
Import LlamaIndex Document Type: Import the Document class from llama_index.core to enable the creation of LlamaIndex-compatible document objects.
#### STEP 4:
Load Documents into LlamaIndex Format: Iterate through the initially uploaded files, read their content, and encapsulate each file's text into a Document object, storing these objects in a list named documents.
#### STEP 5:
Import Embedding and Indexing Libraries: Import SentenceTransformer for generating text embeddings, faiss for vector indexing, numpy for numerical array operations, and os (though os is imported, its usage isn't shown in the provided snippet for the core algorithm).
#### STEP 6:
Re-upload/Verify Document Files: Potentially re-upload or re-confirm the presence of the source text files, ensuring they are accessible for subsequent processing steps.
#### STEP 7:
Prepare Documents for Embedding: Create a new list of documents by reading the content of the uploaded files directly as raw strings, preparing them for input into the embedding model.
#### STEP 8:
Generate Embeddings: Initialize a SentenceTransformer model (specifically "all-MiniLM-L6-v2") and use it to transform the prepared text documents into numerical vector embeddings, converting the output to a NumPy array.
#### STEP 9:
Initialize and Populate FAISS Index: Determine the dimensionality of the generated embeddings, then initialize a faiss.IndexFlatL2 (a basic FAISS index for L2 distance similarity) with this dimension, and finally add the computed document embeddings to this FAISS index, making them searchable.

### PROGRAM:
```
!pip install llama-index llama-hub transformers sentence-transformers --upgrade --quiet
!pip install llama-index sentence-transformers transformers --quiet
!pip uninstall llama-index -y
!pip install llama-index[embeddings] sentence-transformers --upgrade

from google.colab import files
uploaded = files.upload()

from llama_index.core import Document

documents = []
for filename in uploaded:
    with open(filename, 'r', encoding='utf-8') as f:
        documents.append(Document(text=f.read()))

!pip install git+https://github.com/jerryjliu/llama_index.git

!pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

uploaded = files.upload()
file_names = list(uploaded.keys())

documents = []
for file in file_names:
    with open(file, 'r', encoding='utf-8') as f:
        documents.append(f.read())

embedder = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def ask_question(query, top_k=1):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    for i in indices[0]:
        print(f"\n Matched file: {file_names[i]}")
        print(f"Content snippet: {documents[i][:500]}...\n")

ask_question("Summarize the main points.")
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/eb918c56-f87f-4d06-b38e-d4e9f61482f1)

### RESULT:
Therefore the code is excuted successfully.
