# 📘 AI-Powered MCQ Generator

This is an end-to-end **MCQ generation app** built with  GenAI ,RAG based app  . Upload any subject PDF, provide an answer keyword, and this app will:

- Retrieve the most relevant context
- Generate a question using a large language model (LLM)
- Create 3 smart distractors using semantic similarity
- Output a full MCQ with 4 options
- Export everything to CSV — ready for quizzes, exams, or learning apps!



## 🚀 Live Demo

**Try the app**: []

**GitHub Repo**: [https://github.com/krunal05/mcq-generator-ai]



## ✨ Features

-  Upload any textbook, notes, or document in **PDF**
-  Provide a **manual answer** for the type of question you want
-  Smart **context retrieval** using **SentenceTransformer + FAISS (MMR strategy)**
-  **Retriever** using **VectorStore FAISS with search_type=mmr**
-  **Question generation** using **FLAN-T5 via LangChain**
-  **Distractor generation** using **KeyBERT + cosine similarity (from the retrieved context)**
-  **Export** all questions, answers, and options as a **CSV**
-  Streamlit-based clean UI


## 🧠 How It Works

1. **Upload PDF**
2. **Input an answer/concept** 
3. The PDF is chunked based on **recursivetextsplitter**
4. Each chunk is embedded using **SentenceTransformer**
5. The best context chunk is retrieved using **FAISS + MMR**
6. **LangChain + FLAN-T5** generates a question based on context + answer
      **Here question generator llm ,retriever , prompt,embedding generation ,text splitter,vector store are all combined just to generate the question , so forming chain and combined called RAG**
7. **KeyBERT** extracts key phrases from context, filters using cosine similarity, and selects 3 distractors
8. **MCQ** is displayed and exported


## 🔧 Tech Stack

| Component             | Tool/Library                              |

| LLM                   |  `google/flan-t5-base` via HuggingFace    |
| Embedding Model       |  `all-MiniLM-L6-v2` (SentenceTransformer) |
| Vector Store          |  FAISS with MMR search strategy           |
| Distractor Engine     |  KeyBERT + cosine similarity filtering    |
| LangChain Components  |  PromptTemplate, LLMChain                 |
| Frontend              |  Streamlit                                |



## 🖥️ Installation


git clone https://github.com/"your username "/mcq-generator-ai.git  |
cd mcq-generator-ai  |
pip install -r requirements.txt  |
streamlit run app.py  |
