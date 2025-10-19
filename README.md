# RAG System with Amazon Bedrock - Complete Implementation

## ✅ Status: FULLY IMPLEMENTED AND TESTED

A complete Retrieval-Augmented Generation (RAG) system using Amazon Bedrock, ChromaDB, and state-of-the-art AI models.

## 📁 Project Files

### Core System Files
1. **`rag_system.py`** - Complete RAG system with automatic comparisons
2. **`rag_interactive.py`** - Interactive version with menu interface
3. **`requirements.txt`** - All project dependencies

### Documentation Files
4. **`README_RAG.md`** - Complete technical documentation (Spanish)
5. **`README_ENGLISH.md`** - This file (English)
6. **`INSTRUCCIONES_RAG.md`** - Step-by-step usage guide (Spanish)

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
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

## 📊 Sample Documents (English)

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

## 🔍 Example Usage

### Running the Automatic System

```powershell
PS> python rag_system.py

Adding sample documents...
[OK] 10 documents added to collection

================================================================================
RAG SYSTEM WITH AMAZON BEDROCK
================================================================================

Initial RAG system test:

Query: How does Amazon Bedrock relate to RAG systems?

RAG Response: Based on the provided context, Amazon Bedrock is a fully managed 
service that provides access to foundation models...

================================================================================
COMPARISON: RAG vs Without RAG
================================================================================

Query: What are embeddings used for in AI?
================================================================================

[RAG] Response with RAG:
--------------------------------------------------------------------------------
Based on the context provided, embeddings are vector representations of text...
--------------------------------------------------------------------------------

[WITHOUT RAG] Response without RAG:
--------------------------------------------------------------------------------
Embeddings in AI are mathematical representations...
--------------------------------------------------------------------------------
```

## 🛠️ Customization

### Adding Your Own Documents

#### Method 1: Interactive System
1. Run `python rag_interactive.py`
2. Select option 4 (Add new documents)
3. Enter your documents one per line
4. Type 'DONE' when finished

#### Method 2: Edit the Code
Modify `sample_docs` in either file:

```python
sample_docs = [
    "Your custom document 1",
    "Your custom document 2",
    "Your custom document 3",
    # Add more documents here
]
```

### Adjusting Parameters

```python
# Change number of documents to retrieve
rag_generate(query, top_k=5)  # Default is 2

# Adjust generation parameters
"max_tokens": 1000,      # Increase for longer responses
"temperature": 0.5,      # 0.0-1.0 (lower = more deterministic)
"top_p": 0.9,           # 0.0-1.0 (diversity control)
```

### Changing Models

```python
# Use different Claude models
TEXT_GENERATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"   # Fast & cheap
TEXT_GENERATION_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"  # Balanced
TEXT_GENERATION_MODEL = "anthropic.claude-v2:1"                     # Claude v2
```

## 🐛 Troubleshooting

### Error: "Could not connect to Bedrock"
**Solution:**
- Verify AWS credentials: `aws configure`
- Confirm correct region (us-east-1)
- Check Bedrock access in your AWS account

### Error: "Model access denied"
**Solution:**
1. Go to AWS Console → Amazon Bedrock
2. Click "Model access" in sidebar
3. Request access to:
   - Amazon Titan Embed Text v1
   - Anthropic Claude 3 Haiku
4. Wait for approval (usually instant)

### Error: "ValidationException"
**Solution:**
- Model may not be available in your region
- Switch to us-east-1 or us-west-2
- Verify model ID is correct

### Encoding Issues (weird characters)
**Solution:**
- Already fixed in code (no emojis)
- If persists, run: `chcp 65001` in PowerShell

## 📈 Performance Tips

### Document Quality
- Use clear, concise documents
- Avoid duplicates
- Keep documents focused on one topic

### Optimal top_k Values
- **top_k=2**: More focused responses
- **top_k=3-5**: Balance between context and precision
- **top_k>5**: More context but may dilute relevance

### Effective Queries
- Be specific in your questions
- Use terms related to your documents
- Try different formulations

## 🎓 Use Cases

1. **Enterprise Chatbot**
   - Load company manuals and policies
   - Answer employee questions with accurate context

2. **Technical Documentation Assistant**
   - Index product documentation
   - Answer technical questions with exact references

3. **Product Q&A System**
   - Load catalogs and specifications
   - Answer customer queries with updated information

4. **Educational Assistant**
   - Index course materials
   - Answer student questions based on content

## 🔄 Code Translation

**All code is now in English:**
- ✅ Comments translated
- ✅ Variable names in English
- ✅ Function docstrings in English
- ✅ Print messages in English
- ✅ Sample documents in English
- ✅ User prompts in English

## 📚 Key Learnings

1. **RAG significantly improves accuracy** by providing relevant context
2. **Embeddings enable semantic search** beyond keyword matching
3. **ChromaDB simplifies** vector database management
4. **Amazon Bedrock provides easy access** to state-of-the-art AI models
5. **Combining retrieval and generation** creates more robust systems

## 🚀 Next Steps

### Beginner Level
- ✅ Run the system with sample documents
- ✅ Try different queries
- ✅ Compare responses with and without RAG

### Intermediate Level
- Add your own documents
- Experiment with different top_k values
- Adjust generation parameters (temperature, top_p)

### Advanced Level
- Implement ChromaDB persistence
- Add more documents (100+)
- Create web interface with Streamlit
- Implement evaluation metrics
- Add conversation history

## 📖 Resources

- **AWS Bedrock Docs:** https://docs.aws.amazon.com/bedrock/
- **ChromaDB Docs:** https://docs.trychroma.com/
- **Anthropic Claude:** https://docs.anthropic.com/
- **Course Repository:** https://github.com/udacity/cd13926-Building-Apps-Amazon-Bedrock-exercises

## ✨ System Status

**✅ READY TO USE**

Both `rag_system.py` and `rag_interactive.py` are fully functional with:
- All code in English
- No encoding issues
- Compatible with ChromaDB latest version
- Tested and working

Run either file to start using the RAG system immediately!

---

**Happy learning with RAG and Amazon Bedrock!** 🎉
