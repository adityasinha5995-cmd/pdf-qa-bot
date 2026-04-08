from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based strictly on the provided document context.

Rules:
- Only use information from the context below to answer.
- If the answer is not in the context, say: "I couldn't find that in the document."
- Keep answers concise and accurate.
- If asked for a summary, summarise only what is in the context.

Context:
{context}

Question: {question}

Answer:"""


def build_chain(retriever, model_name="mistral"):
    llm = OllamaLLM(
        model=model_name,
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