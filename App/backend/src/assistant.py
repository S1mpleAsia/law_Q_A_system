import logging
import os
import re

import numpy as np
import torch
from FlagEmbedding import FlagReranker
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone
from qdrant_client import models
from search_document_v2.custom_retriever import init_retriever
from search_document_v2.graph_retriever import GraphResult, GraphRetriever


THRESHOLD_SCORE_RERANK = 0.5
# Default for questions that do not choose (the UI toggle sends use_graph per question)
ENABLE_GRAPH_RETRIEVAL = os.getenv("ENABLE_GRAPH_RETRIEVAL", "true").lower() == "true"

logger = logging.getLogger(__name__)


def _text(content):
    """LLM message content as a string (some models return a list of parts)."""
    if isinstance(content, list):
        content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
    return (content or "").strip()


def _article_label(text):
    """First article heading in a chunk, e.g. "Điều 55. Hội đồng thành viên" (for display)."""
    match = re.search(r"Điều \d+\w*\.[^\n]{0,70}", text)
    return match.group(0).strip() if match else text.strip()[:70]


class Singleton(type):
    # Create the instance on the first call, with that call's arguments, then reuse it.
    # (Creating it at class definition ignored the arguments and connected on import.)
    _instance = None

    def __call__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kw)
        return cls._instance


class Retrieval(metaclass=Singleton):

    def __init__(self, top_n=20, top_k=10):
        self.retriever = init_retriever()
        condition = models.FieldCondition(
            key="metadata.status", match=models.MatchValue(value="Hết hiệu lực toàn bộ")
        )
        self.retriever.search_kwargs["filter"] = models.Filter(must_not=[condition])
        self.retriever.search_kwargs["k"] = top_n

        self.top_n = top_n
        self.top_k = top_k

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pc = Pinecone(os.getenv("PINECONE_API_KEY"))
        self.reranker = pc.inference
        # reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, devices=device, batch_size=4)
        # reranker.compute_score(["mode to gpu", "cuda"])
        # self.reranker = reranker

        # Always loaded so each question can turn graph retrieval on or off without a restart;
        # it disables itself if the graph collections are missing
        self.graph = GraphRetriever()

    def search(self, query):
        """Parent documents for the question (kept for callers that do not use the graph result)."""
        docs, _ = self.retrieve(query)
        return docs

    def retrieve(self, query, keywords=None, use_graph=None):
        """Return (parent documents, GraphResult). keywords come from detect_user_intent;
        use_graph=None falls back to ENABLE_GRAPH_RETRIEVAL. Without graph mode the GraphResult is empty."""
        if use_graph is None:
            use_graph = ENABLE_GRAPH_RETRIEVAL
        parent_docs, candidates = self.retriever.invoke(query)

        # Rerank candidates as (parent id, text): hybrid search children ...
        rerank_inputs = [
            (candidate[0].metadata["doc_id"], candidate[0].page_content) for candidate in candidates
        ]
        # ... plus parents found through the knowledge graph that hybrid search missed,
        # reranked on their own text so that noisy graph matches are filtered out
        graph_result = GraphResult()
        graph_ids = []
        if use_graph:
            graph_result = self.graph.search(query, keywords)
            graph_ids = [pid for pid in graph_result.parent_ids if pid not in parent_docs]
            if graph_ids:
                parent_docs.update(self.retriever.docstore.mget(graph_ids))
                rerank_inputs += [(pid, parent_docs[pid].page_content) for pid in graph_ids if pid in parent_docs]
                logger.info(f"Graph added {len(graph_ids)} candidate parents: {graph_ids}")

        if not rerank_inputs:
            return [], graph_result

        rerank_results = self.reranker.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            # Unique ids: several children of the same parent can be candidates
            documents=[{"id": str(i), "text": text} for i, (_, text) in enumerate(rerank_inputs)],
            top_n=len(rerank_inputs),
            return_documents=False,
            parameters={"truncate": "END"},
        )

        top_k_docs = []
        seen_parents = set()
        for result in rerank_results.data:
            if result.score < THRESHOLD_SCORE_RERANK:
                break

            doc_id = rerank_inputs[result.index][0]
            if doc_id in seen_parents or doc_id not in parent_docs:
                continue
            seen_parents.add(doc_id)
            top_k_docs.append(parent_docs[doc_id])

            if len(top_k_docs) == self.top_k:
                break

        # Which articles the graph added that hybrid search missed, and whether they reached the answer
        graph_result.added_parents = [
            {
                "id": pid,
                "title": parent_docs[pid].metadata.get("title", ""),
                "article": _article_label(parent_docs[pid].page_content),
                "used": pid in seen_parents,
            }
            for pid in graph_ids if pid in parent_docs
        ]

        # Verbatim source passages of the best-matching graph relations, unless already among the law texts
        missing = [pid for pid in graph_result.passage_ids if pid not in seen_parents]
        if missing:
            parent_docs.update(self.retriever.docstore.mget([pid for pid in missing if pid not in parent_docs]))
        graph_result.passage_docs = [parent_docs[pid] for pid in missing if pid in parent_docs]
        graph_result.passages = [
            {"id": pid, "title": parent_docs[pid].metadata.get("title", ""), "article": _article_label(parent_docs[pid].page_content)}
            for pid in missing if pid in parent_docs
        ]
        return top_k_docs, graph_result


# Some parents cannot be split (e.g. appendices without article headings) and reach tens of
# thousands of characters; cap each one in the prompt
MAX_DOC_CHARS = 12000
NO_ANSWER_MESSAGE = (
    "Xin lỗi, tôi chưa tạo được câu trả lời cho câu hỏi này. "
    "Bạn có thể diễn đạt lại câu hỏi cụ thể hơn (ví dụ nêu rõ loại hình doanh nghiệp hoặc vấn đề cần hỏi)."
)
PARAPHRASE_INSTRUCTION = (
    " Answer in your own words: summarise the provisions and cite them by article number "
    "(e.g. 'theo khoản 1 Điều 188'); do not copy long passages of the legal text verbatim."
)


class Generation:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        # Retry client for answers blocked as RECITATION (Gemini refuses to reproduce long source
        # passages verbatim); a little randomness helps it paraphrase instead of copying
        self.paraphrase_llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def check_question(self, question, history_summary=None):
        messages = [
            (
                "human",
                f'Cho tóm tắt lịch sử đoạn hội thoại sau: "{history_summary}". \n\n\
                    Câu hỏi tiếp theo:"{question}" có phải là câu hỏi liên quan đến lĩnh vực luật doanh nghiệp Việt Nam không? \
                Chỉ trả lời "yes" hoặc "no".',
            ),
        ]

        messages_explanation = [
            (
                "system",
                f"You are an expert lawyer in Vietnam, tasked with answering only asked \
                    questions from customers about Vietnamese business Law. \
                        You are asked one sentence unrelated to Vietnamese business Law. \
                            Please answer logically and rationally in Vietnamese.\
                            \n\n Below is the history conversation: \n \"{history_summary}\" ",
            ),
            ("human", f"{question}"),
        ]

        answer = self.llm.invoke(messages).content
        result = re.match(r"\W*yes\b", answer.strip().lower()) is not None

        if not result:
            explanation = self.llm.invoke(messages_explanation).content
        else:
            explanation = None

        return result, explanation

    def generate(self, question, docs: Document, history_summary=None, graph_context="", graph_passages=None):
        # Law texts, then (graph mode) the verbatim source passages of the graph relations used below
        all_docs = list(docs) + list(graph_passages or [])
        context = [
            f"- doc {i}: + info:{doc.metadata} \n + content:{doc.page_content[:MAX_DOC_CHARS]}"
            for i, doc in enumerate(all_docs, start=1)
        ]
        context = "\n".join(context)

        # Knowledge-graph facts link concepts across articles / documents (LightRAG-style context).
        # Each line ends with [the document it was extracted from]; the law texts above stay the source to cite
        if graph_context:
            context += (
                "\n\nRelated legal concepts and relations from a knowledge graph built from the same legal texts. "
                "Each line ends with the document it comes from in [brackets]:\n" + graph_context
                + "\n\nRules for these lines: they are short summaries that help you connect provisions across "
                "articles and documents. When you cite, always name the document AND the article, taking the "
                "document from the [bracket] (a line saying only 'Theo Điều 55' refers to Điều 55 of that document). "
                "Prefer quoting provisions from the full legal texts above; do not guess clause or article numbers "
                "that appear in neither the texts nor the lines."
            )

        messages = [
            (
                "system",
                f"You are an expert lawyer in Vietnam, \
                    tasked with answering asked questions from customers about Vietnamese \
                    business Law based on the given information. \
                    Please use, gather, and deduce based on the knowledge in the following \
                    information to answer the user’s question in Vietnamese. \
                    Please respond accurately, fully, clearly citing the law. \n \
                    Relevant legal information: \n {context} ",
            ),
            ("human", f"{question}"),
        ]

        ai_msg = self.llm.invoke(messages)
        answer = _text(ai_msg.content)
        if answer:
            return answer

        # Empty answer: usually finish_reason RECITATION. Retry once asking for a paraphrase
        finish_reason = (ai_msg.response_metadata or {}).get("finish_reason")
        logger.warning(f"Empty answer (finish_reason={finish_reason}); retrying with a paraphrase instruction")
        retry_messages = [(messages[0][0], messages[0][1] + PARAPHRASE_INSTRUCTION), messages[1]]
        retry_msg = self.paraphrase_llm.invoke(retry_messages)
        answer = _text(retry_msg.content)
        if answer:
            return answer

        logger.warning(f"Empty answer after retry (finish_reason={(retry_msg.response_metadata or {}).get('finish_reason')})")
        return NO_ANSWER_MESSAGE

    def generate_for_search_internet(self, question, context, history_summary=None):
        
        messages = [
            (
                "system",
                f"You are an expert lawyer in Vietnam, \
                    tasked with answering asked questions from customers about Vietnamese \
                    business Law based on the searched internet information and history conversation. \
                    Please use, gather, and deduce based on the knowledge in the following internet \
                    information to answer the user’s question in Vietnamese. Please respond accurately, full. \n \
                    - Internet informationn: \n {context} \n\
                    - History conversation: \n \"{history_summary}\"",
            ),
            ("human", f"{question}"),
        ]

        ai_msg = self.llm.invoke(messages)

        return ai_msg.content


class Assistant(metaclass=Singleton):
    def __init__(self, top_n=20, top_k=10):
        self.retrieval = Retrieval(top_n, top_k)
        self.generation = Generation()
        self.llm = self.generation.llm

    def ask(self, question, use_graph=None):
        result, explanation = self.generation.check_question(question)
        if not result:
            return explanation

        docs, graph = self.retrieval.retrieve(question, use_graph=use_graph)
        answer = self.generation.generate(question, docs, graph_context=graph.context,
                                          graph_passages=getattr(graph, "passage_docs", None))

        return answer
    
    def get_history_summary(self, history_summary, question_answer):
        messages = [
            (
                "system",
                "You are an expert at summarizing legal conversations. \
                    Please summarize the provided Vietnamese conversation by user, \
                        limit from 500 to 1000 words (do not skip any questions).",
            ),
            ("human", 
             f"- Cho đoạn tóm tắt hội thoại sau: \"{history_summary}\" \n\
                - Cho đoạn hội thoại kế tiếp: \"{question_answer}\" \n\
                    \n Hãy tóm tắt tiếp đoạn hội thoại."
            )
        ]
        
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content
    
    def router(self, history, question):
        return self.generation.check_question(question, history)


def init_assistant():
    assistant = Assistant(top_n=16, top_k=5)
    return assistant
