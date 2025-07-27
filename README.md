# 🧠 NeuroDoc — AI-Powered Study & Research Assistant

NeuroDoc is a multi-functional GenAI assistant that helps students and professionals extract knowledge from documents, websites, and more. It summarizes content, generates MCQs, and even converts summaries to speech — powered by top-tier LLMs and embeddings.

> "Learn smarter, not harder — with AI on your side."

---

## ✨ Key Features

- 📄 **Document Summarization**  
  Upload PDFs, PPTs, Word, or Text files — get crisp, AI-generated summaries using Gemini and Groq.

- 🌐 **Website Summarizer**  
  Paste any URL — NeuroDoc fetches and summarizes page content using LangChain's WebLoader.

- ❓ **MCQ Generator**  
  Auto-generate multiple-choice questions from text or documents — great for self-evaluation.

- 🧠 **RAG + Embedding Search** *(NEW)*  
  Google Embedding-powered vector store retrieves accurate content chunks before summarization or QA.

- 🔊 **Text-to-Speech (TTS)**  
  Generate audio of summaries using gTTS. Adjust playback speed, download MP3, and learn on the go.

---

## 🔧 Tech Stack

| Layer       | Tech Used                                  |
|-------------|---------------------------------------------|
| **Frontend**| Streamlit                                   |
| **LLMs**    | Groq (Mixtral / LLaMA), Gemini Pro (Google) |
| **Embeddings** | GoogleGenerativeAIEmbeddings via LangChain |
| **Vector Store** | FAISS (in-memory for now)              |
| **Speech**  | gTTS (Google Text-to-Speech)                |
| **Backend** | Python + LangChain + OpenRouter             |

---
