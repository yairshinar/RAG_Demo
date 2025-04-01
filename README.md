# Retrieval-Augmented Generation (RAG) Demo

This repository contains a self-contained Python notebook demonstrating a basic Retrieval-Augmented Generation (RAG) pipeline using local documents and a language model. 
The goal is to enhance a small LLM with external knowledge through vector search and prompt engineering.

---

## ✨ Demo Summary

I show how to:
1. Embed documents using a sentence transformer
2. Perform similarity search using cosine similarity (scikit-learn)
3. Use a generative model to answer a question based on retrieved documents

This is an end-to-end example of using embeddings and generative models for question-answering.

---

## ⚙️ Prerequisites

- Python 3.8+
- Recommended: 16 GB RAM
- GPU (optional but helpful)

Install the required dependencies:

```bash
pip install transformers torch scikit-learn
```

> Note:  use `cosine_similarity` instead of FAISS for better portability and to avoid system-specific installation issues.

---

## 📂 Documents
I simulate a small knowledge base with a list of documents. In practice, these could be loaded from PDFs, websites, or internal knowledge systems that you have available.

```python
# Sample documents
docs = [
    "ProductX is the latest widget released in 2024. It features improved battery life.",
    "To reset ProductX, hold the power button for 10 seconds until the LED blinks.",
    "Our support plans include Basic, Plus, and Enterprise tiers, offering 24/7 support in higher tiers."
]
```

---

## 🧰 Embed the Documents

I use `sentence-transformers/all-MiniLM-L6-v2` to compute semantic embeddings:

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def mean_pooling(output, mask):
    tokens = output[0]
    mask_exp = mask.unsqueeze(-1).expand(tokens.size()).float()
    return (tokens * mask_exp).sum(1) / mask_exp.sum(1)

def embed(texts, tokenizer, model):
    tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return mean_pooling(output, tokens['attention_mask']).numpy()

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
model = AutoModel.from_pretrained(embedding_model)
doc_embeddings = np.vstack([embed([doc], tokenizer, model)[0] for doc in docs])
```

---

## 🔍 Semantic Search with Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

query = "How do I reset ProductX?"
query_embedding = embed([query], tokenizer, model)[0]
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
top_idx = int(np.argmax(similarities))
retrieved = docs[top_idx]
```

---

## 🤖 Generate Answer using Flan-T5

I use `google/flan-t5-base` for answering the question:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

prompt = f"Context: {retrieved}\nQuestion: {query}\nAnswer:"
result = qa_pipeline(prompt, max_length=100)[0]['generated_text']
print("Answer:", result.strip())
```

---

## 📊 Example Output

```
Query: How do I reset ProductX?
Retrieved: To reset ProductX, hold the power button for 10 seconds until the LED blinks.
Answer: To reset ProductX, press and hold the power button for 10 seconds until the LED blinks.
```

---

## 🚀 Run Locally

```bash
python -m venv rag-demo-env
rag-demo-env\Scripts\activate  # On Windows
pip install transformers torch scikit-learn
```

Launch Jupyter:
```bash
jupyter notebook rag_demo_latest.ipynb
```

Or run the script version:
```bash
python rag_demo.py
```

---

## ⚖️ Customization Ideas

- Replace `docs` with data from files (PDF, DOCX, etc.)
- Swap the LLM with another model (e.g., Mistral, LLaMA)
- Increase `k` to use top-K documents for richer context
- Add a simple web UI with Gradio or Streamlit

---

## 🚨 Notes

- On first run, the models will be downloaded from Hugging Face Hub
- On CPU, expect 2–5 seconds for answer generation
- You can cache models in `.cache/huggingface` for offline use

---

## 🎉 You're ready to go!
Try tweaking the prompt, changing documents, or integrating with your own system.

For any issues, refer to the official Hugging Face docs or raise an issue in this repo.
