# Importing necessary libraries and modules.
import uuid
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import os
import faiss
import cloudpickle
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnableMap    
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Set the path for the database directory.
DB_PATH="./db/"

# Function to load or create a vector database.
# This database will be used for storing and retrieving document embeddings.
def load_vector_db(DB_PATH="./db/"):
    # Initialize variables for the components of the database.
    db = None
    memoryDocStoreDict = {}
    indexToDocStoreIdDict = {}
    
    # Check if the database already exists. If it does, load its components.
    if os.path.exists(DB_PATH):
        memoryDocStoreDict = cloudpickle.load(open(DB_PATH+"memoryDocStoreDict.pkl", "rb"))
        indexToDocStoreIdDict = cloudpickle.load(open(DB_PATH+"indexToDocStoreIdDict.pkl", "rb"))
        index = faiss.read_index(DB_PATH+"faiss.index")
    else:
        # If the database does not exist, create a new FAISS index.
        index = faiss.IndexFlatL2(384)

    # Create the FAISS vector database with the loaded or new components.
    db = FAISS(
        index=index,
        docstore=InMemoryDocstore(memoryDocStoreDict),
        index_to_docstore_id=indexToDocStoreIdDict,
        embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda:0'})
    )
    return db

# Function to populate the vector database with documents.
# It processes each file in the 'wiki/' directory, splits the content into smaller chunks,
# and stores these chunks along with their metadata in the database.
def populate_vector_db(DB_PATH="./db/"):
    db = load_vector_db(DB_PATH=DB_PATH)

    # Process each file in the 'wiki/' directory.
    for wiki_file in os.listdir("wiki/"):
        texts = []
        metadatas = []
        
        wiki_file_path  = "wiki/"+wiki_file
        wiki_chunks_dir = "wiki_chunks/"+wiki_file
        os.makedirs(wiki_chunks_dir, exist_ok=True)
       
        # Read the content of the file.
        content = open(wiki_file_path, "r").read()
        # Split the content into smaller chunks for better manageability.
        for chunk in TokenTextSplitter(chunk_size=256).split_text(content):
            random_uuid = str(uuid.uuid4())
            texts.append(chunk)
            
            wiki_chunk_file_path = wiki_chunks_dir+"/"+random_uuid+".txt"
            open(wiki_chunk_file_path, "w").write(chunk)
            metadatas.append({
                'wiki_file_path': wiki_file_path,
                'wiki_chunk_file_path': wiki_chunk_file_path
            })

        # Add the text chunks and their metadata to the database.
        db.add_texts(texts, metadatas)
        
    # Save the components of the database if the directory does not exist.
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    
    cloudpickle.dump(db.docstore._dict, open(DB_PATH+"memoryDocStoreDict.pkl", "wb"))
    cloudpickle.dump(db.index_to_docstore_id, open(DB_PATH+"indexToDocStoreIdDict.pkl", "wb"))
    faiss.write_index(db.index, DB_PATH+"faiss.index")
    
    return db

# Function to configure and retrieve a large language model from Hugging Face.
def get_llm():
    # Define the model name and retrieve the necessary token for authentication.
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    token = os.environ['HUGGINGFACE_TOKEN']

    # Configure the model for quantization to reduce memory usage.
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    device_map = {"": 0}

    # Load the model and tokenizer from Hugging Face with the specified configurations.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    # Create a pipeline for text generation using the loaded model and tokenizer.
    llama_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=llama_pipeline, model_kwargs={'temperature':0.7})
    
    return llm

# Function to format a list of documents into a single string.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to ask a question and receive an answer using the large language model and the document database.
def ask(q):
    # Define a template for the prompt to be used with the large language model.
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)

    llm = get_llm()

    # Create a chain of operations to process the question.
    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableMap(
        {"documents": db.as_retriever(), "question": RunnablePassthrough()}
    ) | {
        "documents": lambda input: [doc.metadata for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

    # Invoke the chain of operations with the question.
    response = rag_chain_with_source.invoke(q)
    print(response["answer"])
    for doc in response["documents"]:
        print(doc['wiki_chunk_file_path'])
    

# Main execution block: populate and load the vector database, then use it to answer a sample question.
if __name__=="__main__":
    db = populate_vector_db(DB_PATH=DB_PATH)
    db = load_vector_db(DB_PATH=DB_PATH)
    ask("What is the capital of NJ?")
