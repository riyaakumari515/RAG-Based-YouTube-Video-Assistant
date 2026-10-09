import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()


class SimpleRAGChain:
    def __init__(self, llm, vector_store, k=3):
        self.llm = llm
        self.vector_store = vector_store
        self.k = k

    def invoke(self, inputs):
        query = inputs.get("query") if isinstance(inputs, dict) else inputs
        docs = self.vector_store.similarity_search(query, k=self.k)

        # build a compact context from retrieved docs
        context_parts = []
        for d in docs:
            src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
            context_parts.append(f"[source:{src}] {d.page_content}")
        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful assistant analyzing a YouTube video. "
            "Answer the question based strictly on the provided context. "
            "If the answer is not in the context, say 'I couldn't find that in the video.' "
        )

        messages = [
            ("system", system_prompt),
            ("human", f"Context: {context}\n\nQuestion: {query}"),
        ]

        ai_msg = self.llm.invoke(messages)
        answer = getattr(ai_msg, "content", str(ai_msg))

        return {"result": answer, "source_documents": docs}


def get_rag_chain(vector_store):
    # Allow overriding the model via env var to handle deprecations without code edits
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model_name, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    return SimpleRAGChain(llm, vector_store)