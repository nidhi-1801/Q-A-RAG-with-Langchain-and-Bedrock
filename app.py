import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


# Streamlit UI Config 

st.set_page_config(page_title="Chat_PDF", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffd6e8, #ffeef6);
}
h1, h2, h3 {
    color: #ff2f92;
    font-family: 'Comic Sans MS', cursive;
}
div.stButton > button {
    background-color: #ff69b4;
    color: white;
    border-radius: 20px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}
div.stButton > button:hover {
    background-color: #ff1493;
}
input {
    border-radius: 12px !important;
    border: 2px solid #ff69b4 !important;
}
section[data-testid="stSidebar"] {
    background-color: #fff0f6;
}
</style>
""", unsafe_allow_html=True)

# Bedrock Client

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

titan_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)


# Data Ingestion

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)
    return docs


# Vector Store

def get_vector_store(docs):
    vectorstore = FAISS.from_documents(docs, titan_embeddings)
    vectorstore.save_local("faiss_index")


# LLMs

def get_claude_llm():
    return BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        model_kwargs={"max_tokens": 512, "temperature": 0.7}
    )

def get_llama3_llm():
    return Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 512}
    )


# Prompt

PROMPT = PromptTemplate(
    template="""
Human: Use the context below to answer the question.
If you don't know the answer, say you don't know.

<context>
{context}
</context>

Question: {question}
Assistant:
""",
    input_variables=["context", "question"]
)


# Retrieval QA

def get_response_llm(llm, vectorstore, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

    response = qa({"query": query})
    return response["result"]

# Streamlit App

def main():
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask something smart")

    # Sidebar for vector store management
    with st.sidebar:
        st.subheader("📚 Vector Store Management")

        if st.button("💾 Update Vectors"):
            with st.spinner("Processing PDFs..."):
                try:
                    docs = data_ingestion()
                    get_vector_store(docs)
                    st.success("✅ Vectors updated 💅")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Main area for Q&A
    if st.button("🤖 Ask Claude"):
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    vectorstore = FAISS.load_local(
                        "faiss_index",
                        titan_embeddings,
                        allow_dangerous_deserialization=True
                    )
                    llm = get_claude_llm()
                    answer = get_response_llm(llm, vectorstore, user_question)
                    
                    st.subheader("💬 Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("💡 Make sure you've updated the vectors first!")
        else:
            st.warning("⚠️ Please enter a question first!")


# Entry Point

if __name__ == "__main__":
    main()