from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.documents import Document

load_dotenv()

MODEL = "llama3.2:latest"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Free local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""

# Load vector database
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# Local LLM
llm = ChatOllama(
    model=MODEL,
    temperature=0
)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all previous user questions with the current question.
    """
    prior = "\n".join(
        m["content"] for m in history if m["role"] == "user"
    )
    return prior + "\n" + question


def answer_question(
    question: str,
    history: list[dict] = []
) -> tuple[str, list[Document]]:
    """
    Answer a question using Retrieval-Augmented Generation (RAG).
    Returns the answer and the retrieved documents.
    """

    combined = combined_question(question, history)

    docs = fetch_context(combined)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )

    messages = [
        SystemMessage(content=system_prompt)
    ]

    messages.extend(convert_to_messages(history))

    messages.append(
        HumanMessage(content=question)
    )

    response = llm.invoke(messages)

    return response.content, docs

answer, docs = answer_question(
    "What insurance plans does Insurellm offer?"
)
print(answer)