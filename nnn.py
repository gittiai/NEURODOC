import os
import tempfile
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredURLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import create_retrieval_chain, load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import streamlit as st
from fpdf import FPDF
import io

def load_youtube_transcript(url: str):
    try:
        video_id = parse_qs(urlparse(url).query).get("v")
        if not video_id:
            raise ValueError("Invalid YouTube URL format.")
        transcript = YouTubeTranscriptApi.get_transcript(video_id[0])
        text = " ".join([t['text'] for t in transcript])
        return [Document(page_content=text)]
    except Exception as e:
        raise RuntimeError(f"Transcript fetch failed: {e}")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
   #st.image("https://cdn-icons-png.flaticon.com/512/8618/8618942.png", width=150)
   #st.image("/Users/adityashivhare/Downloads/python_project/74e0f2ed-9538-4e93-b0e1-be4b537ba7a5.png",width=200)
   st.image("/Users/adityashivhare/Downloads/python_project/—Pngtree—strongest brain head illustration_4521000.png", width=250)
     
   
    
st.sidebar.markdown(
    "<div style='font-size: 32px; text-align: center;'>🅝🅔🅤🅡🅞🅓🅞🅒</div>",
    unsafe_allow_html=True
)


# --- Sidebar theme toggle ---
Theme = st.sidebar.selectbox("🌗 Select Theme", ["Dark", "Azure", "Tree", "Sunset"])
# Define colors based on theme
if Theme == "Dark":
    bg_color = "#0e1117"
    text_color = "#ffffff"
elif Theme == "Azure":
    bg_color = "#80b6d8"
    text_color = "#408CAB"
elif Theme == "Tree":
    bg_color = "#8cd8a6"
    text_color = "#26A42F"
elif Theme=="Sunset":
   bg_color = "#f1a05e"
   text_color = "#D45A09"


st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# Load environment variables
load_dotenv()



# Sidebar - API Key input
st.sidebar.markdown("## 🔐 Enter API Key")
with st.sidebar:
    groq_api_key = st.text_input("🔑 Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")

st.sidebar.markdown("## 📘 Instructions")
st.sidebar.markdown("""
1. Upload a document (`.pdf`, `.pptx`, `.txt`, `.docx`)
2. Choose an action:
   - Generate Summary
   - Generate MCQs
   - Ask a Question
3. For Q&A, type your question in the box.
4. Use this prompt->(Answer the question using the context provided, and include the file name and page number as the source)  to find file name and page no            
5. Upload Correct URL from Yoytube video 
""")


# LLM and embeddings
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
tab1, tab2 = st.tabs(["DOCUMENTS", "YOUTUBE"])
with tab1:
  st.header("DOCUMENT")
# File upload + option select
  uploaded_file = st.file_uploader("📂 Upload a Document", type=["pdf", "pptx", "txt", "docx"])

  option = st.radio("Choose an option", ["Generate Summary", "Generate MCQs", "Ask a Question","Translate summary"])

from fpdf import FPDF
import io

#aise hi try kr rha tha pdf ka 
def generate_pdf(text: str, title: str = "Generated Report", font: str = "Arial") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font(font, size=12)
    pdf.set_title(title)

    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)

    # Instead of writing directly to BytesIO, get string output
    pdf_output_str = pdf.output(dest='S').encode('latin1')

    # Wrap it in BytesIO so Streamlit can use it
    pdf_stream = io.BytesIO(pdf_output_str)
    return pdf_stream.read()

  
# DOCUMENT TAB 

  
if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            filepath = tmp.name

        # Load the file
        if uploaded_file.name.endswith(".pdf"):
            docs = PyPDFLoader(filepath).load()
        elif uploaded_file.name.endswith(".pptx"):
            docs = UnstructuredPowerPointLoader(filepath).load()
        elif uploaded_file.name.endswith(".txt"):
            docs = TextLoader(filepath).load()
        elif uploaded_file.name.endswith(".docx"):
            docs = UnstructuredWordDocumentLoader(filepath).load()
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Create vector store & retriever
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        # Process user selection
        if option == "Generate Summary":
            joined_text = "\n".join([doc.page_content for doc in chunks])
            prompt = PromptTemplate(
                template="Generate a full-fledged summary on the following content in around 400 words:\n\n{text}",
                input_variables=["text"]
            )
            chain = prompt | llm
            summary = chain.invoke({"text": joined_text})
            st.subheader("📄 Summary")
            st.write(summary.content)
            st.download_button(
             label="📥 Download Summary (TXT)",
             data=summary.content,
             file_name="summary.txt",
             mime="text/plain"
            )
            style=st.selectbox("Select font",["Arial","Courier","Times","Symbol","ZapfDingbats"])
            pdf_bytes = generate_pdf(summary.content, title="Document Summary",font=style)
            st.download_button(
              label="📄 Download Summary (PDF)",
              data=pdf_bytes,
              file_name="summary.pdf",
              mime="application/pdf"
               )



        elif option == "Generate MCQs":
            joined_text = "\n".join([doc.page_content for doc in chunks])
            prompt = PromptTemplate(
                template="Generate 10 multiple choice questions based on the following content. Each question should have 4 options and the correct answer marked.\n\n{text}",
                input_variables=["text"]
            )
            chain = prompt | llm
            mcqs = chain.invoke({"text": joined_text})
            st.subheader("❓ MCQs")
            st.write(mcqs.content)
         
            st.download_button(
             label="📥 Download MCQs (TXT)",
             data=mcqs.content,
             file_name="mcqs_from_pdf.txt",
             mime="text/plain"
             )
           
            st.download_button(
             label="📥 Download MCQs (PDF)",
             data=mcqs.content,
             file_name="mcqs_from_doc.pdf",
             mime="application/pdf"
            )

        elif option == "Ask a Question":
            query = st.text_input("Ask your question here...")
            if query:
                prompt = ChatPromptTemplate.from_template("""
                Answer the question based only on the provided context.
                <context>
                {context}
                </context>
                Question: {input}
                """)
                doc_chain = create_stuff_documents_chain(llm, prompt)
                qa_chain = create_retrieval_chain(retriever, doc_chain)
                response = qa_chain.invoke({"input": query})
                st.subheader("💬 Answer")
                st.write(response["answer"])
                
        elif option=="Translate summary":
            language = st.selectbox("Select Language", ["Hindi", "German", "French", "Chinese","English","Spanish"])
            joined_text = "\n".join([doc.page_content for doc in chunks])
            prompt = PromptTemplate(
                template="Generate a full-fledged summary on the following content in around 400 words:\n\n{text}",
                input_variables=["text"]
            )
            chain = prompt | llm
            summary = chain.invoke({"text": joined_text})

            prompt3=PromptTemplate(
                template="Translate the following summary into {language} \n\n{text}",
                input_variables=["text","language"]
            )
            trans_chain= prompt3|llm
            result=trans_chain.invoke({"text":summary,"language":language})
            st.write(result.content)



# === YOUTUBE TAB ===
with tab2:
    st.header("YOUTUBE")
    st.subheader("Paste a YouTube or Website URL below to get a summary:")
    generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="Enter a YouTube or Website URL")

    prompt_template = """
    Provide a concise and clear summary of the following content in approximately 300 words:
    Content:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

    if st.button("Summarize the content from YT or Website 📝"):
        try:
            with st.spinner("⏳ Loading and summarizing content..."):
                if "youtube.com" in generic_url:
                    docs = load_youtube_transcript(generic_url)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url])
                    docs = loader.load()

                chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt)
                output_summary = chain.run(docs)

                st.success("✅ Summary:")
                st.write(output_summary)

        except Exception as e:
            st.exception(f"⚠️ Exception: {e}")

    prompt_template2 = """
    Generate 10 multiple choice questions using the following content from a video or webpage transcript:
    Content: {text}
    """
    prompt2 = PromptTemplate(template=prompt_template2, input_variables=['text'])

    if st.button("Generate MCQs 🔠"):
        try:
            with st.spinner("⏳ Loading and generating MCQs..."):
                if "youtube.com" in generic_url:
                    docs = load_youtube_transcript(generic_url)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url])
                    docs = loader.load()

                combined_text = "\n\n".join([doc.page_content for doc in docs])
                chain2 = prompt2 | llm
                result = chain2.invoke({"text": combined_text})

                st.success("✅ Multiple Choice Questions:")
                st.write(result.content)
        except Exception as e:
            st.exception(f"⚠️ Exception: {e}")

   # === Setup memory and conversation chain once ===
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)
if "conversation" not in st.session_state:    
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.chat_memory,
        verbose=False
    )

# === Show input box and trigger on button click ===
st.markdown("# 🤖 NEUROBUDDY" )
question = st.text_input("Chat with NeuroBuddy")
if st.button("Submit Question 🚀") and question:
    with st.spinner("Thinking..."):
        response = st.session_state.conversation.run(question)
        st.markdown(f"**Answer:** {response}")

if st.session_state.chat_memory:
    st.markdown("### 🧠 Chat History")
    for msg in st.session_state.chat_memory.chat_memory.messages:
        if msg.type == "human":
            st.markdown(f"🧑‍💻 **You**: {msg.content}")
        elif msg.type == "ai":
            st.markdown(f"🤖 **Bot**: {msg.content}")
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_memory.clear()
    st.success("Chat history cleared.")

