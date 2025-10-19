# RAG System with Amazon Bedrock - Complete Implementation

A complete Retrieval-Augmented Generation (RAG) system using Amazon Bedrock, ChromaDB, and state-of-the-art AI models.

## 📁 Project Files

### Core System Files
1. **`rag_system.py`** - Complete RAG system with automatic comparisons
2. **`rag_interactive.py`** - Interactive version with menu interface
3. **`requirements.txt`** - All project dependencies

## 🚀 Quick Start

### 1. Install Dependencies

Activating the environment will allow you to use the project’s specific
libraries. The command varies depending on your operating system and the
shell you are using:

### Windows
```sh
.\.venv\Scripts\Activate
```

### Linux/Mac
```sh
source .venv/bin/activate
```

```sh
pip install chromadb sentence-transformers
```

All other dependencies (boto3, numpy, etc.) should already be installed.

### 2. Configure AWS Credentials

```powershell
aws configure
```

You'll need:
- AWS Access Key ID
- AWS Secret Access Key
- AWS Session Token (Optional)
- Default region: `us-east-1` (or any region with Bedrock enabled)

### 3. Enable Models in AWS Bedrock Console

Go to AWS Bedrock Console and enable access to:
- **amazon.titan-embed-text-v1** (for embeddings)
- **anthropic.claude-3-haiku-20240307-v1:0** (for text generation)

## 💻 Running the System

### Option A: Automatic System (Recommended for Testing)

```powershell
python rag_system.py
```

**What it does:**
- Loads 10 sample documents about AWS Bedrock and RAG
- Runs 3 test queries
- Automatically compares responses WITH RAG vs WITHOUT RAG
- Shows results side by side

**Perfect for:** Understanding how RAG works and its benefits

### Option B: Interactive System

```powershell
python rag_interactive.py
```

**What it does:**
- Presents an interactive menu with 6 options:
  1. Make a query with RAG
  2. Make a query without RAG
  3. Compare RAG vs Without RAG
  4. Add new documents
  5. View current documents
  6. Exit

**Perfect for:** Experimenting with your own queries and documents

## 🎯 Key Features

### ✅ Implemented Components

- **Embeddings**: Amazon Titan Embed Text v1
- **Text Generation**: Claude 3 Haiku (fast and cost-effective)
- **Vector Database**: ChromaDB
- **Semantic Search**: Cosine similarity
- **Sample Documents**: 10 preloaded documents about AWS Bedrock and RAG
- **Automatic Comparison**: Side-by-side RAG vs non-RAG responses

### 🔧 Technical Architecture

```
User Query
    ↓
1. Query Embedding (Titan Embed)
    ↓
2. Search in ChromaDB (Semantic Similarity)
    ↓
3. Retrieve Top-K Relevant Documents
    ↓
4. Build Prompt (Query + Context)
    ↓
5. Generate Response (Claude 3 Haiku)
    ↓
Contextualized Response
```

## 📊 Sample Documents

The system comes preloaded with 10 documents:

1. "Amazon Bedrock is a fully managed service for foundation models."
2. "RAG systems combine retrieval and generation to improve responses."
3. "Embeddings are vector representations of text in high-dimensional spaces."
4. "Chroma is an efficient vector store for building AI applications."
5. "Foundation models can be fine-tuned for specific tasks and domains."
6. "Amazon Bedrock provides access to AI models from leading companies like Anthropic, AI21 Labs, and Amazon."
7. "RAG improves response accuracy by providing relevant context from stored knowledge."
8. "Embeddings enable searching for similar documents using cosine similarity."
9. "Claude is a language model developed by Anthropic available on Amazon Bedrock."
10. "RAG systems are especially useful for applications requiring domain-specific knowledge."


---

**Happy learning with RAG and Amazon Bedrock!** 🎉
