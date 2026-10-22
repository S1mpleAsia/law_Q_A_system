"""Graph retrieval over the knowledge graph

Finds parent chunks related to a question through the graph:
1. nodes and edges whose descriptions are closest to the question (Qdrant)
2. edges one hop away from the matched nodes (MongoDB)
3. every match votes for the parent chunks it was extracted from
"""
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import pymongo
from qdrant_client import QdrantClient

from .custom_embedding import M3EmbeddingsAPI
from .graph_builder import ENTITY_COLLECTION, RELATION_COLLECTION
from .vector_store import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
)

logger = logging.getLogger(__name__)

RELATION_WEIGHT = 1.5  # an edge is more specific than a node
MAX_GRAPH_PASSAGES = 2  # full text of the source passages of the best-matching edges, given to the LLM
HOP_WEIGHT = 0.5  # edges reached by following a matched node, not matched directly


@dataclass
class GraphResult:
    parent_ids: list = field(default_factory=list)  # parents ranked by graph evidence
    context: str = ""  # entity / relation descriptions to show the LLM next to the law texts
    # Structured view of the same matches, for the UI
    keywords: dict = field(default_factory=dict)  # {"low_level": [...], "high_level": [...]} actually used
    entities: list = field(default_factory=list)  # [{"name", "type", "description", "score"}]
    relations: list = field(default_factory=list)  # [{"source", "target", "keywords", "description", "score"}]
    neighbors: list = field(default_factory=list)  # one-hop edges of the matched nodes: [{"source", "target", "keywords"}]
    neighbor_count: int = 0  # all one-hop edges used for scoring (neighbors only lists the most specific)
    source_chunks: list = field(default_factory=list)  # parents the graph points to: [{"id", "title", "article", "score"}]
    passage_ids: list = field(default_factory=list)  # source parents of the best-matching edges, shown verbatim to the LLM
    passages: list = field(default_factory=list)  # filled by Retrieval: [{"id", "title", "article"}] actually added to the prompt
    added_parents: list = field(default_factory=list)  # filled by Retrieval: [{"id", "title", "article", "used"}]


def _clean_keywords(keywords):
    """Merged keywords may contain a stringified list (some extractions returned a list); flatten them."""
    cleaned = []
    for keyword in keywords:
        parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", keyword) if keyword.startswith("[") else []
        for part in ([a or b for a, b in parts] if parts else [keyword]):
            part = part.strip(" []'\"")
            if part and part not in cleaned:
                cleaned.append(part)
    return cleaned


class GraphRetriever:
    def __init__(self, top_entities=6, top_relations=6, max_parents=8, context_entities=6, context_relations=6):
        self.top_entities = top_entities
        self.top_relations = top_relations
        self.max_parents = max_parents
        self.context_entities = context_entities
        self.context_relations = context_relations

        self.db = pymongo.MongoClient(os.getenv("MONGODB_URL"))[os.getenv("LAW_DOC_DB", "law")]
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        self.embedder = M3EmbeddingsAPI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL, model=EMBEDDING_MODEL)

        existing = {c.name for c in self.qdrant.get_collections().collections}
        self.enabled = ENTITY_COLLECTION in existing and RELATION_COLLECTION in existing
        if not self.enabled:
            logger.warning("Graph collections not found in Qdrant; graph retrieval disabled (run Core/build_graph.ipynb)")

    def search(self, query, keywords=None):
        """Return a GraphResult for the question. Never raises: on any error the chatbot
        falls back to hybrid search only.

        keywords: {"low_level": [...], "high_level": [...]} as extracted by the LLM (LightRAG):
        low-level keywords (specific entities) search nodes, high-level keywords (themes,
        kinds of relation) search edges. Without keywords the whole question is used for both.
        """
        if not self.enabled:
            return GraphResult()
        try:
            return self._search(query, keywords or {})
        except Exception as e:
            logger.warning(f"Graph retrieval failed, continuing without it: {type(e).__name__}: {e}")
            return GraphResult()

    def _search(self, query, keywords):
        low_level = ", ".join(keywords.get("low_level") or []) or query
        high_level = ", ".join(keywords.get("high_level") or []) or query
        entity_vector, relation_vector = self.embedder.embed_documents([low_level, high_level])
        entity_hits = self.qdrant.query_points(ENTITY_COLLECTION, query=entity_vector, limit=self.top_entities, with_payload=True).points
        relation_hits = self.qdrant.query_points(RELATION_COLLECTION, query=relation_vector, limit=self.top_relations, with_payload=True).points

        entity_keys = [h.payload["key"] for h in entity_hits]
        relation_keys = [h.payload["key"] for h in relation_hits]
        entities = {d["_id"]: d for d in self.db["graph_entities"].find({"_id": {"$in": entity_keys}})}
        relations = {d["_id"]: d for d in self.db["graph_relations"].find({"_id": {"$in": relation_keys}})}
        hop_edges = list(self.db["graph_relations"].find(
            {"$or": [{"source": {"$in": entity_keys}}, {"target": {"$in": entity_keys}}]},
            {"source": 1, "target": 1, "keywords": 1, "parent_ids": 1},
        ))

        # A vote is divided by sqrt(number of parents of the match), so a generic node such as
        # "doanh nghiệp" (hundreds of parents) weighs much less than an edge found in two articles
        scores = defaultdict(float)

        def vote(parent_ids, weight):
            if parent_ids:
                share = weight / math.sqrt(len(parent_ids))
                for pid in parent_ids:
                    scores[pid] += share

        for h in entity_hits:
            if h.payload["key"] in entities:
                vote(entities[h.payload["key"]]["parent_ids"], h.score)
        for h in relation_hits:
            if h.payload["key"] in relations:
                vote(relations[h.payload["key"]]["parent_ids"], RELATION_WEIGHT * h.score)
        for edge in hop_edges:
            vote(edge["parent_ids"], HOP_WEIGHT)

        ranked = sorted(scores, key=scores.get, reverse=True)[:self.max_parents]
        matched_entities, matched_relations = self._describe(
            [(entities[h.payload["key"]], h.score) for h in entity_hits if h.payload["key"] in entities],
            [(relations[h.payload["key"]], h.score) for h in relation_hits if h.payload["key"] in relations],
        )
        logger.info(f"Graph retrieval: keywords low={low_level!r} high={high_level!r} -> "
                    f"{len(entity_hits)} nodes, {len(relation_hits)} edges, {len(ranked)} parents")
        # Verbatim source passages: where the best-matching relations were stated (LightRAG gives the
        # LLM the source chunks next to the graph facts, so citations can be checked against the law text)
        passage_ids = []
        for h in relation_hits:
            r = relations.get(h.payload["key"])
            if not r:
                continue
            pid = (r.get("description_sources") or r["parent_ids"])[0]
            if pid not in passage_ids:
                passage_ids.append(pid)
            if len(passage_ids) == MAX_GRAPH_PASSAGES:
                break

        return GraphResult(
            parent_ids=ranked,
            passage_ids=passage_ids,
            context=self._format_context(matched_entities, matched_relations),
            keywords={"low_level": keywords.get("low_level") or [], "high_level": keywords.get("high_level") or []},
            entities=matched_entities,
            relations=matched_relations,
            neighbors=self._neighbors(hop_edges, set(relation_keys)),
            neighbor_count=len(hop_edges),
            source_chunks=self._source_chunks(ranked, scores),
        )

    def _neighbors(self, hop_edges, matched_relation_keys, limit=10):
        """Most specific one-hop edges (fewest source parents) that were not matched directly, for display."""
        edges = sorted((e for e in hop_edges if e["_id"] not in matched_relation_keys), key=lambda e: len(e["parent_ids"]))[:limit]
        keys = {e["source"] for e in edges} | {e["target"] for e in edges}
        names = {d["_id"]: d["name"] for d in self.db["graph_entities"].find({"_id": {"$in": list(keys)}}, {"name": 1})}
        return [
            {"source": names.get(e["source"], e["source"]), "target": names.get(e["target"], e["target"]),
             "keywords": _clean_keywords(e["keywords"])}
            for e in edges
        ]

    def _source_chunks(self, ranked, scores):
        """Title and first article heading of the parents the graph points to, for display."""
        docs = {d["id"]: d for d in self.db["documents"].find({"id": {"$in": ranked}}, {"id": 1, "content": 1, "metadata.title": 1})}
        chunks = []
        for pid in ranked:
            if pid in docs:
                match = re.search(r"Điều \d+\w*\.[^\n]{0,70}", docs[pid]["content"])
                chunks.append({
                    "id": pid,
                    "title": docs[pid]["metadata"].get("title", ""),
                    "article": match.group(0).strip() if match else docs[pid]["content"].strip()[:70],
                    "score": round(scores[pid], 2),
                })
        return chunks

    def _describe(self, entity_matches, relation_matches):
        """Matched nodes / edges as plain dicts: display names, descriptions labelled with the document
        they were written from (descriptions often say only "Theo Điều 55"), similarity scores."""
        endpoint_keys = {r["source"] for r, _ in relation_matches} | {r["target"] for r, _ in relation_matches}
        names = {d["_id"]: d["name"] for d in self.db["graph_entities"].find({"_id": {"$in": list(endpoint_keys)}}, {"name": 1})}

        def sourced(doc, n):
            texts = doc["descriptions"][:n]
            sources = (doc.get("description_sources") or [])[:n]
            return [{"text": t, "parent_id": sources[i] if i < len(sources) else None} for i, t in enumerate(texts)]

        entity_descs = [sourced(e, 2) for e, _ in entity_matches[:self.context_entities]]
        relation_descs = [sourced(r, 1) for r, _ in relation_matches[:self.context_relations]]
        parent_ids = {d["parent_id"] for ds in entity_descs + relation_descs for d in ds if d["parent_id"]}
        titles = {d["id"]: d["metadata"].get("title", "") for d in
                  self.db["documents"].find({"id": {"$in": list(parent_ids)}}, {"id": 1, "metadata.title": 1})}
        for ds in entity_descs + relation_descs:
            for d in ds:
                d["source"] = titles.get(d["parent_id"], "")

        def labelled(ds, limit):
            return " ".join(f"{d['text']} [{d['source']}]" if d["source"] else d["text"] for d in ds)[:limit]

        entities = [
            {
                "name": e["name"],
                "type": e.get("type") or "khái niệm",
                "descriptions": ds,
                "description": labelled(ds, 500),
                "score": round(score, 3),
            }
            for (e, score), ds in zip(entity_matches[:self.context_entities], entity_descs)
        ]
        relations = [
            {
                "source": names.get(r["source"], r["source"]),
                "target": names.get(r["target"], r["target"]),
                "keywords": _clean_keywords(r["keywords"]),
                "descriptions": ds,
                "description": labelled(ds, 400),
                "score": round(score, 3),
            }
            for (r, score), ds in zip(relation_matches[:self.context_relations], relation_descs)
        ]
        return entities, relations

    @staticmethod
    def _format_context(entities, relations):
        """Short text for the prompt: matched nodes and edges with their descriptions (which cite articles)."""
        lines = [f"- {e['name']} ({e['type']}): {e['description']}" for e in entities]
        lines += [
            f"- {r['source']} → {r['target']} [{', '.join(r['keywords'])}]: {r['description']}" for r in relations
        ]
        return "\n".join(lines)
