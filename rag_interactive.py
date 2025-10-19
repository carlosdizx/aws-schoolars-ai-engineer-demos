"""
Interactive RAG System with Amazon Bedrock
Allows interactive queries and comparison of responses with and without RAG
"""

import boto3
import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Initialize Bedrock client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Model configuration
EMBEDDING_MODEL = "amazon.titan-embed-text-v1"
TEXT_GENERATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


class BedrockEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for Amazon Bedrock"""
    
    def __init__(self):
        pass
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            body = json.dumps({"inputText": text})
            try:
                response = bedrock_runtime.invoke_model(
                    modelId=EMBEDDING_MODEL,
                    body=body,
                    contentType='application/json',
                    accept='application/json'
                )
                response_body = json.loads(response['body'].read())
                embeddings.append(response_body['embedding'])
            except Exception as e:
                print(f"[ERROR] Error getting embedding: {e}")
                raise
        return embeddings


def generate_text(prompt):
    """Generates text using Claude 3 on Amazon Bedrock"""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "top_p": 0.9,
    })
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId=TEXT_GENERATION_MODEL, 
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        print(f"[ERROR] Error generating text: {e}")
        raise


# Initialize ChromaDB
print("Initializing ChromaDB...")
chroma_client = chromadb.Client()

# Create custom embedding function
bedrock_ef = BedrockEmbeddingFunction()

# Create collection
try:
    try:
        chroma_client.delete_collection(name="bedrock_docs")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="bedrock_docs",
        embedding_function=bedrock_ef
    )
    print("[OK] Chroma collection created successfully\n")
except Exception as e:
    print(f"[ERROR] Error creating collection: {e}")
    exit(1)


def add_documents(docs):
    """Adds documents to the Chroma collection"""
    try:
        collection.add(
            documents=docs,
            ids=[f"doc_{i}" for i in range(len(docs))]
        )
        return True
    except Exception as e:
        print(f"[ERROR] Error adding documents: {e}")
        return False


def rag_generate(query, top_k=2, verbose=False):
    """Generates a response using RAG"""
    try:
        # Retrieve relevant documents
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Show retrieved documents if verbose is enabled
        if verbose:
            print("\nRetrieved documents:")
            for i, doc in enumerate(results['documents'][0], 1):
                print(f"  {i}. {doc}")
            print()
        
        # Build prompt with retrieved context
        context = "\n".join(results['documents'][0])
        
        prompt = f"""Given the following context, please answer the question.

Context: {context}

Question: {query}

Based on the provided context, my answer is:"""
        
        # Generate response
        response = generate_text(prompt)
        return response
    except Exception as e:
        print(f"[ERROR] Error in rag_generate: {e}")
        return None


def generate_without_rag(query):
    """Generates a response without using RAG"""
    try:
        return generate_text(query)
    except Exception as e:
        print(f"[ERROR] Error in generate_without_rag: {e}")
        return None


def show_menu():
    """Shows the main menu"""
    print("\n" + "="*80)
    print("INTERACTIVE RAG SYSTEM - AMAZON BEDROCK")
    print("="*80)
    print("\nOptions:")
    print("  1. Make a query with RAG")
    print("  2. Make a query without RAG")
    print("  3. Compare RAG vs Without RAG")
    print("  4. Add new documents")
    print("  5. View current documents")
    print("  6. Exit")
    print("="*80)


def view_documents():
    """Shows all documents in the collection"""
    try:
        # Get all documents
        results = collection.get()
        docs = results.get('documents', [])
        
        if not docs:
            print("\nNo documents in collection.")
            return
        
        print(f"\nDocuments in collection ({len(docs)} total):")
        print("-"*80)
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc}")
        print("-"*80)
    except Exception as e:
        print(f"[ERROR] Error getting documents: {e}")


def main():
    """Main function of the interactive system"""
    
    # Add initial sample documents
    print("Loading sample documents...")
    sample_docs = [
        "Amazon Bedrock is a fully managed service for foundation models.",
        "RAG systems combine retrieval and generation to improve responses.",
        "Embeddings are vector representations of text in high-dimensional spaces.",
        "Chroma is an efficient vector store for building AI applications.",
        "Foundation models can be fine-tuned for specific tasks and domains.",
        "Amazon Bedrock provides access to AI models from leading companies like Anthropic, AI21 Labs, and Amazon.",
        "RAG improves response accuracy by providing relevant context from stored knowledge.",
        "Embeddings enable searching for similar documents using cosine similarity.",
        "Claude is a language model developed by Anthropic available on Amazon Bedrock.",
        "RAG systems are especially useful for applications requiring domain-specific knowledge."
    ]
    
    if add_documents(sample_docs):
        print(f"[OK] {len(sample_docs)} documents loaded successfully\n")
    else:
        print("[ERROR] Error loading initial documents")
        return
    
    # Main loop
    while True:
        show_menu()
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == '1':
            # Query with RAG
            print("\n" + "="*80)
            print("QUERY WITH RAG")
            print("="*80)
            query = input("\nEnter your query: ").strip()
            
            if query:
                print("\nProcessing with RAG...")
                response = rag_generate(query, top_k=3, verbose=True)
                if response:
                    print("Response:")
                    print("-"*80)
                    print(response)
                    print("-"*80)
        
        elif choice == '2':
            # Query without RAG
            print("\n" + "="*80)
            print("QUERY WITHOUT RAG")
            print("="*80)
            query = input("\nEnter your query: ").strip()
            
            if query:
                print("\nProcessing without RAG...")
                response = generate_without_rag(query)
                if response:
                    print("Response:")
                    print("-"*80)
                    print(response)
                    print("-"*80)
        
        elif choice == '3':
            # Compare RAG vs Without RAG
            print("\n" + "="*80)
            print("COMPARISON: RAG vs WITHOUT RAG")
            print("="*80)
            query = input("\nEnter your query: ").strip()
            
            if query:
                print("\nProcessing with RAG...")
                rag_response = rag_generate(query, top_k=3, verbose=True)
                
                print("\nProcessing without RAG...")
                no_rag_response = generate_without_rag(query)
                
                print("\n" + "="*80)
                print("COMPARISON RESULTS")
                print("="*80)
                
                print("\n[RAG] WITH RAG:")
                print("-"*80)
                if rag_response:
                    print(rag_response)
                print("-"*80)
                
                print("\n[WITHOUT RAG] WITHOUT RAG:")
                print("-"*80)
                if no_rag_response:
                    print(no_rag_response)
                print("-"*80)
        
        elif choice == '4':
            # Add new documents
            print("\n" + "="*80)
            print("ADD NEW DOCUMENTS")
            print("="*80)
            print("\nEnter documents (one per line).")
            print("Type 'DONE' when finished:\n")
            
            new_docs = []
            while True:
                doc = input(f"Document {len(new_docs) + 1}: ").strip()
                if doc.upper() == 'DONE':
                    break
                if doc:
                    new_docs.append(doc)
            
            if new_docs:
                print(f"\nAdding {len(new_docs)} documents...")
                if add_documents(new_docs):
                    print(f"[OK] {len(new_docs)} documents added successfully")
            else:
                print("[WARNING] No documents were added")
        
        elif choice == '5':
            # View current documents
            view_documents()
        
        elif choice == '6':
            # Exit
            print("\nThank you for using the RAG System!")
            print("="*80 + "\n")
            break
        
        else:
            print("\n[WARNING] Invalid option. Please select 1-6.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSystem interrupted. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
