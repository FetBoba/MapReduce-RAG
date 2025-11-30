"""LangChain RAG assembly helpers."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional, Tuple

try:
    from langchain.chains import RetrievalQA
except ImportError:  # pragma: no cover - optional dependency in new LangChain releases
    RetrievalQA = None

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

DEFAULT_TEMPLATE = """You are a precise teaching assistant. Use the provided context to answer the question.
Context: {context}
Question: {question}
Answer in complete sentences and cite the most relevant fact."""


class SimpleRAGChain:
    """Minimal chain implementation when RetrievalQA is unavailable."""

    def __init__(
        self, llm: BaseLanguageModel, retriever: VectorStoreRetriever, prompt: PromptTemplate
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt

    @staticmethod
    def _question(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return payload.get("query") or payload.get("question") or ""
        raise ValueError("Payload must be a string or mapping with 'query'.")

    @staticmethod
    def _format_docs(documents) -> str:
        return "\n\n".join(doc.page_content for doc in documents)

    def invoke(self, payload: Any):
        question = self._question(payload)
        documents = self.retriever.invoke(question)
        formatted = self.prompt.format(
            context=self._format_docs(documents), question=question
        )
        answer = self.llm.invoke(formatted)
        return {"result": answer, "source_documents": documents}

    def run_with_metrics(self, payload: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Invoke the chain while measuring retrieval and generation timings."""

        question = self._question(payload)
        timings: Dict[str, float] = {}

        retrieval_start = perf_counter()
        documents = self.retriever.invoke(question)
        timings["retrieval_seconds"] = perf_counter() - retrieval_start

        formatted = self.prompt.format(
            context=self._format_docs(documents), question=question
        )

        generation_start = perf_counter()
        answer = self.llm.invoke(formatted)
        timings["generation_seconds"] = perf_counter() - generation_start
        timings["total_seconds"] = sum(timings.values())

        return {"result": answer, "source_documents": documents}, timings


def build_simple_chain(
    llm: BaseLanguageModel,
    retriever: VectorStoreRetriever,
    prompt_template: Optional[str] = None,
):
    template = prompt_template or DEFAULT_TEMPLATE
    prompt = PromptTemplate.from_template(template)
    return SimpleRAGChain(llm=llm, retriever=retriever, prompt=prompt)


def build_rag_chain(
    llm: BaseLanguageModel,
    retriever: VectorStoreRetriever,
    prompt_template: Optional[str] = None,
):
    if RetrievalQA is not None:
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate.from_template(prompt_template or DEFAULT_TEMPLATE)
            },
        )
    return build_simple_chain(llm=llm, retriever=retriever, prompt_template=prompt_template)
