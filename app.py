import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


load_dotenv()


SYSTEM_PROMPT = """
You are a comparative scripture assistant.

Rules:
1. Use ONLY the retrieved text.
2. Never merge teachings.
3. Answer separately for each scripture.
4. If a scripture does not address the question, say:
   \"No direct reference found.\"

Format EXACTLY like this:

Gita says:
- ...

Bible says:
- ...

Quran says:
- ...
""".strip()


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_or_load_vectorstore() -> FAISS:
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_dir = "scripture_faiss"

    if os.path.isdir(index_dir):
        return FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs: list[Document] = []

    for source in ["gita", "bible", "quran"]:
        txt_path = os.path.join("data", source, f"{source}.txt")
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(
                f"Missing {txt_path}. Run the notebook to generate the .txt files first."
            )
        text = _load_text(txt_path)
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": source}))

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_dir)
    return db


@st.cache_resource
def initialize() -> tuple[FAISS, ChatGroq]:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    db = _build_or_load_vectorstore()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return db, llm


def ask(question: str, retriever, llm: ChatGroq) -> str:
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    else:
        docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[{d.metadata.get('source', '').upper()}]\n{d.page_content}" for d in docs
    )

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}
"""

    result = llm.invoke(prompt).content
    if isinstance(result, str):
        return result
    return "\n".join(str(part) for part in result)


def main() -> None:
    st.title("Comparative Scripture Assistant")
    st.caption("Gita • Bible • Quran (RAG)")

    try:
        db, llm = initialize()
    except Exception as e:
        st.error(str(e))
        st.stop()

    retriever = db.as_retriever(search_kwargs={"k": 15})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(question, retriever=retriever, llm=llm)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
