# Raag-Based NDIS Assistant

This project is an intelligent AI-powered assistant designed to help users navigate the National Disability Insurance Scheme (NDIS) effectively. Built using the **Raag pipeline**, it utilizes **OpenAI embeddings** to embed domain-specific knowledge into a `.npz` file, enabling semantic search and question answering based on user input and budget constraints.

---

## 🧠 Overview

The system is designed to:

- Answer any NDIS-related question.
- Use a preprocessed and embedded knowledge base stored in an `.npz` file.
- Take user-specific **budget information** as input to personalize responses.
- Deliver fast, context-aware answers using the **Raag pipeline**.
- Ensure relevant, concise, and user-centric information delivery.

---

## ⚙️ How It Works

### 1. Knowledge Base Embedding

- NDIS documents and guidelines are processed and embedded using **OpenAI's text embedding model**.
- These embeddings are stored in a compressed `.npz` file for efficient access and querying.

### 2. User Input Pipeline

- The Raag pipeline captures the user's query and budget data.
- The system combines the query with the budget context to form a **personalized prompt**.
- The embedded knowledge base is searched semantically to find the most relevant chunks of information.

### 3. Answer Generation

- The most relevant sections are fed to a language model (like GPT) via the Raag orchestration.
- The model responds with a natural language answer tailored to the user’s budget and specific needs.

---

## ✨ Features

- ✅ Embedded NDIS knowledge base (offline `.npz` format).
- ✅ Raag pipeline integration.
- ✅ Budget-aware question answering.
- ✅ Fast, contextually relevant responses.
- ✅ Scalable and customizable.

---

## 🧰 Tech Stack

- Python 3.x
- [Raag](https://pypi.org/project/raag/) – AI reasoning pipeline
- OpenAI Embedding API (e.g. `text-embedding-ada-002`)
- NumPy (for `.npz` storage)
- Faiss (optional – for similarity search)
- Flask / Streamlit (optional – for UI or API layer)

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/raag-ndis-assistant.git
   cd raag-ndis-assistant