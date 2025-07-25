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
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import create_retrieval_chain, load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
load_dotenv()


gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key
 
st.sidebar.markdown(
    "<div style='font-size: 32px; text-align: center;'>🅝🅔🅤🅡🅞🅓🅞🅒</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-image: url("https://images.unsplash.com/photo-1611325058416-db7794e8e32c?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");

        background-size: cover;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

  

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1611325058416-db7794e8e32c?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar theme toggle ---
Theme = st.sidebar.selectbox("🌗 Select Theme", ["Dark", "Azure", "Tree", "Sunset","Moon"])
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
elif Theme=="Moon":
    bg_color ="#ffffff"
    text_color= "#000000"


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



load_dotenv()


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
5. Upload Correct URL from Youtube video """)



llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
tab1, tab2 = st.tabs(["DOCUMENTS", "WEBSITE"])
with tab1:
  st.header("DOCUMENT")

  uploaded_files = st.file_uploader("📂 Upload a Document", type=["pdf", "pptx", "txt", "docx"],accept_multiple_files=True)

  option = st.radio("Choose an option", ["Generate Summary", "Generate MCQs", "Ask a Question","Translate summary"])


  if uploaded_files:
    all_docs = []
    all_chunks=[]
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            filepath = tmp.name

        
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
        all_docs.extend(docs)


    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    all_chunks = splitter.split_documents(all_docs)
        
     
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    retriever = vectorstore.as_retriever()

        
    if option == "Generate Summary":
            joined_text = "\n".join([doc.page_content for doc in all_chunks])
            prompt = PromptTemplate(
                template="Generate a full-fledged summary on the following content in around 400 words:\n\n{text}",
                input_variables=["text"]
            )
            chain = prompt | llm
            summary = chain.invoke({"text": joined_text})
            st.subheader("📄 Summary")
            st.write(summary.content)
            st.download_button(
             label="📥 Download Summary(TXT)",
             data=summary.content,
             file_name="summary_from_pdf.txt",
             mime="text/plain"
             )
           
            



    elif option == "Generate MCQs":
            joined_text = "\n".join([doc.page_content for doc in all_chunks])
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
             mime="text/plain",
             key=f"download_mcqs_txt_{uploaded_file.name}" 
             )
           
            

    elif option == "Ask a Question":
            query = st.text_input("Ask your question here...", key="ask_question_box")
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
            language = st.selectbox("Select Language", ["Hindi", "German", "French", "Chinese","English","Spanish"], key=f"lang_selectbox_{option}"  )
            joined_text = "\n".join([doc.page_content for doc in all_chunks])
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




with tab2:
    st.header("WEBSITE")
    st.subheader("Paste a Website URL below to get a summary:")
    generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="Enter a  Website URL")
    

    prompt_template = """
    Provide a concise and clear summary of the following content in approximately 300 words:
    Content:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

    if st.button("Summarize the content from Website 📝"):
        try:
            with st.spinner("⏳ Loading and summarizing content..."):
                loader = UnstructuredURLLoader(urls=[generic_url])
                docs = loader.load()

                chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt)
                output_summary = chain.run(docs)

                st.success("✅ Summary:")
                st.write(output_summary)

        except Exception as e:
            st.exception(f"⚠️ Exception: {e}")

    prompt_template2 = """
    Generate 10 multiple choice questions using the following content from a  webpage transcript:
    Content: {text}
    """
    prompt2 = PromptTemplate(template=prompt_template2, input_variables=['text'])

    if st.button("Generate MCQs 🔠"):
        try:
            with st.spinner("⏳ Loading and generating MCQs..."):
                loader = UnstructuredURLLoader(urls=[generic_url])
                docs = loader.load()

                combined_text = "\n\n".join([doc.page_content for doc in docs])
                chain2 = prompt2 | llm
                result = chain2.invoke({"text": combined_text})

                st.success("✅ Multiple Choice Questions:")
                st.write(result.content)
        except Exception as e:
            st.exception(f"⚠️ Exception: {e}")


if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)
if "conversation" not in st.session_state:    
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.chat_memory,
        verbose=False
    )

st.markdown("# 🤖 NEUROBUDDY" )
question = st.text_input("Chat with NeuroBuddy")
if st.button("Submit Question 🚀") and question:
    with st.spinner("Thinking..."):
        response = st.session_state.conversation.run(question)
        st.markdown(f"**Answer:** {response}")
with st.expander("🧠 Chat History", expanded=False):
 if st.session_state.chat_memory:
    for msg in st.session_state.chat_memory.chat_memory.messages:

        if msg.type == "human":
            st.markdown(f"🧑‍💻 You: {msg.content}")
        elif msg.type == "ai":
            st.markdown(f"🤖 Neurobuddy: {msg.content}")
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_memory.clear()
    st.success("Chat history cleared.")

