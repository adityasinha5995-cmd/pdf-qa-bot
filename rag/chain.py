import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based strictly on the provided document context.

Rules:
- Only use information from the context below to answer.
- If the answer is not in the context, say: "I couldn't find that in the document."
- Keep answers concise and accurate.

Context:
{context}

Question: {question}

Answer:"""


def build_chain(retriever, model_name="llama-3.3-70b-versatile"):
    groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = groq_key

    llm = ChatGroq(
        model_name=model_name,
        temperature=0.1,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain