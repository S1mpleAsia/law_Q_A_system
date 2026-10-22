"""Build LightRAG knowledge graph over the parent chunks stored in MongoDB.
"""

import os
import json
import re
import string
import time
import unicodedata
import uuid
from collections import Counter
import datetime
import pymongo
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient, models
from .custom_embedding import M3EmbeddingsAPI
from .vector_store import (
    COLLECTION_NAME,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
)


EXPIRED_STATUS = "Hết hiệu lực toàn bộ"
ENTITY_COLLECTION = f"{COLLECTION_NAME}_entities"
RELATION_COLLECTION = f"{COLLECTION_NAME}_relations"

EMBEDDING_SIZE = 1024  # BGE-M3 dense vector size
MAX_DESCRIPTIONS = 8  # descriptions kept per node / edge after merging

ENTITY_TYPES = [
    "loại hình doanh nghiệp",
    "cơ quan nhà nước",
    "chủ thể",
    "bộ máy quản lý",
    "thủ tục",
    "hồ sơ giấy tờ",
    "quyền",
    "nghĩa vụ",
    "điều kiện",
    "chế tài",
    "khái niệm pháp lý",
    "văn bản pháp luật",
]

# Fill with EXTRACTION_PROMPT.format(entity_types=..., texts=...).
# Each text block should start with a header line such as "--- parent_id: 142881_11 (Luật 59/2020/QH14) ---".
EXTRACTION_PROMPT = """You build a knowledge graph from Vietnamese legal texts about enterprises.

For EACH text below (identified by its parent_id), extract:
- entities: legal concepts that matter for answering questions, with
  name (Vietnamese, canonical full form, e.g. "công ty trách nhiệm hữu hạn hai thành viên trở lên", not an abbreviation),
  type (one of: {entity_types}),
  description (1-2 Vietnamese sentences on what THIS text says about the entity, citing the article, e.g. "Theo Điều 46 ...").
- relations between entities from the same text, with
  source, target (entity names exactly as in your entities list),
  keywords (a few Vietnamese words for the kind of relation, e.g. "chuyển đổi", "nộp hồ sơ tại", "có quyền"),
  description (1 Vietnamese sentence explaining the relation, citing the article).

Rules:
- Every entity and relation must carry the parent_id of the text it comes from.
- Skip generic words with no legal meaning (e.g. "quy định", "trường hợp", "Luật này").
- Do not invent facts that are not in the text.

Return only JSON:
{{"entities": [{{"parent_id": "...", "name": "...", "type": "...", "description": "..."}}],
  "relations": [{{"parent_id": "...", "source": "...", "target": "...", "keywords": "...", "description": "..."}}]}}

Texts:
{texts}"""

# Common abbreviations
ABBREVIATIONS = {
    "tnhh": "trách nhiệm hữu hạn",
    "cp": "cổ phần",
    "dn": "doanh nghiệp",
    "dntn": "doanh nghiệp tư nhân",
    "hđtv": "hội đồng thành viên",
    "hđqt": "hội đồng quản trị",
    "đhđcđ": "đại hội đồng cổ đông",
    "bks": "ban kiểm soát",
    "gcn": "giấy chứng nhận",
    "đkdn": "đăng ký doanh nghiệp",
    "ubnd": "ủy ban nhân dân",
}


class DailyQuotaExceeded(Exception):
    """The LLM daily request quota is used up; rerun extract_graph() after it resets."""


def entity_key(name: str) -> str:
    """Normalise an entity name"""
    processed_name = unicodedata.normalize('NFC', name).lower()
    processed_name = re.sub(r"[\"'[\]“”‘’(){}]", " ", processed_name).strip().strip(string.punctuation)

    return " ".join(ABBREVIATIONS.get(word, word) for word in processed_name.split())   


def _graph_db():
    url = os.getenv("MONGODB_URL")
    law_db = os.getenv("LAW_DOC_DB", "law")
    return pymongo.MongoClient(url)[law_db]


def _llm(requests_per_minute: float, thinking_budget):
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60,
    )

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        temperature=0,
        rate_limiter=rate_limiter,
        response_mime_type="application/json",
        thinking_budget=thinking_budget,
        max_retries=2,
    )


def _parse_extraction(raw: str | list, parent_ids: list) -> dict:
    """Turn the LLM output into {parent_id: {"entities": [...], "relations": [...]}} for every id in parent_ids."""
    
    if isinstance(raw, list):
        raw = "".join(part if isinstance(part, str) else part.get("text", "") for part in raw)

    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(raw)

    required_fields = {
        "entities": ("name", "type", "description"),
        "relations": ("source", "target", "description"),
    }
    result = {pid: {"entities": [], "relations": []} for pid in parent_ids}

    for kind, required in required_fields.items():
        for item in data.get(kind) or []:
            if not isinstance(item, dict):
                continue
            pid = item.get("parent_id")
            if pid not in result and len(parent_ids) == 1:
                pid = parent_ids[0]
            if pid not in result:
                continue
            if not all(str(item.get(field) or "").strip() for field in required):
                continue
            result[pid][kind].append(
                {key: str(value).strip() for key, value in item.items() if key != "parent_id"}
            )
    return result


def _is_daily_quota_error(error: Exception) -> bool:
    error_msg = str(error)
    return "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in error_msg


def extract_graph(
    title=None,
    limit=None,
    parents_per_call=1,
    requests_per_minute=12,
    thinking_budget=1024,
    include_expired=False,
    max_requests=None,
):
    """Extract entities / relations for parents that have no cached extraction yet. Returns the number of LLM calls."""
    db = _graph_db()
    documents, extractions = db["documents"], db["graph_extractions"]

    query = {}
    if title:
        query["metadata.title"] = title
    if not include_expired:
        query["metadata.status"] = {"$ne": EXPIRED_STATUS}

    done = {d["_id"] for d in extractions.find({}, {"_id": 1})}
    parents = documents.find(query, {"id": 1, "content": 1, "metadata.title": 1}).sort("id", 1)
    pending = [p for p in parents if p["id"] not in done]

    if limit is not None:
        pending = pending[:limit]
    print(f"{len(pending)} parents to extract ({len(done)} already cached)")
    if not pending:
        return 0

    llm = _llm(requests_per_minute, thinking_budget)
    model = os.getenv("GEMINI_MODEL")
    calls = 0
    for start in range(0, len(pending), parents_per_call):
        if max_requests is not None and calls >= max_requests:
            print(f"Stopped after {calls} requests (max_requests)")
            break

        group = pending[start:start + parents_per_call]
        group_ids = [p["id"] for p in group]
        texts = "\n\n".join(
            f"--- parent_id: {p['id']} ({p['metadata']['title']}) ---\n{p['content']}" for p in group
        )
        prompt = EXTRACTION_PROMPT.format(entity_types=", ".join(ENTITY_TYPES), texts=texts)

        try:
            response = llm.invoke(prompt)
            calls += 1
            parsed = _parse_extraction(response.content, group_ids)
        except json.JSONDecodeError:
            # Not cached, so the group is retried on the next run
            print(f"Invalid JSON for {group_ids}; skipped, will retry on the next run")
            continue
        except Exception as e:
            if _is_daily_quota_error(e):
                raise DailyQuotaExceeded(f"Daily quota reached after {calls} requests this run") from e
            raise

        now = datetime.datetime.now(datetime.timezone.utc)
        extractions.bulk_write([
            pymongo.ReplaceOne(
                {"_id": pid},
                {"_id": pid, **parsed[pid], "model": model, "thinking_budget": thinking_budget, "extracted_at": now},
                upsert=True,
            )
            for pid in group_ids
        ])

        n_entities = sum(len(parsed[pid]["entities"]) for pid in group_ids)
        n_relations = sum(len(parsed[pid]["relations"]) for pid in group_ids)
        print(f"{start + len(group)} / {len(pending)} parents | +{n_entities} entities, +{n_relations} relations")

    return calls


def merge_graph():
    """Rebuild `graph_entities` / `graph_relations` from all cached extractions"""
    db = _graph_db()
    nodes = {}  # node key -> accumulated data
    edges = {}  # "source_key|target_key" -> accumulated data

    def add_entity(name, entity_type, description, parent_id):
        key = entity_key(name)
        if not key:
            return None
        node = nodes.setdefault(key, {
            "names": Counter(), "types": Counter(), "descriptions": [], "description_sources": [], "parent_ids": set(),
        })
        node["names"][name] += 1
        if entity_type:
            node["types"][entity_type] += 1
        if description and description not in node["descriptions"]:
            node["descriptions"].append(description)
            node["description_sources"].append(parent_id)  # aligned with descriptions: where each was written
        node["parent_ids"].add(parent_id)
        return key

    for extraction in db["graph_extractions"].find():
        parent_id = extraction["_id"]
        for entity in extraction.get("entities", []):
            add_entity(entity["name"], entity.get("type"), entity.get("description"), parent_id)

        for relation in extraction.get("relations", []):
            # Endpoints become nodes too, even if the model did not list them as entities
            source = add_entity(relation["source"], None, None, parent_id)
            target = add_entity(relation["target"], None, None, parent_id)
            if not source or not target or source == target:
                continue
            edge = edges.setdefault(f"{source}|{target}", {
                "source": source, "target": target,
                "keywords": Counter(), "descriptions": [], "description_sources": [], "parent_ids": set(),
            })
            for keyword in re.split(r"[,;]", relation.get("keywords", "")):
                if keyword.strip():
                    edge["keywords"][keyword.strip().lower()] += 1
            if relation["description"] not in edge["descriptions"]:
                edge["descriptions"].append(relation["description"])
                edge["description_sources"].append(parent_id)
            edge["parent_ids"].add(parent_id)

    degree = Counter()
    for edge in edges.values():
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1

    entity_docs = [
        {
            "_id": key,
            "name": node["names"].most_common(1)[0][0],
            "type": node["types"].most_common(1)[0][0] if node["types"] else None,
            "descriptions": node["descriptions"][:MAX_DESCRIPTIONS],
            # Parent each description came from, so an answer can cite the document, not just "Điều 55"
            "description_sources": node["description_sources"][:MAX_DESCRIPTIONS],
            "parent_ids": sorted(node["parent_ids"]),
            "degree": degree[key],
        }
        for key, node in nodes.items()
    ]
    relation_docs = [
        {
            "_id": key,
            "source": edge["source"],
            "target": edge["target"],
            "keywords": [keyword for keyword, _ in edge["keywords"].most_common(5)],
            "descriptions": edge["descriptions"][:MAX_DESCRIPTIONS],
            "description_sources": edge["description_sources"][:MAX_DESCRIPTIONS],
            "parent_ids": sorted(edge["parent_ids"]),
        }
        for key, edge in edges.items()
    ]

    for collection, docs in (("graph_entities", entity_docs), ("graph_relations", relation_docs)):
        db[collection].delete_many({})
        if docs:
            db[collection].insert_many(docs)
    db["graph_relations"].create_index("source")
    db["graph_relations"].create_index("target")

    print(f"merged: {len(entity_docs)} entities, {len(relation_docs)} relations")
    return len(entity_docs), len(relation_docs)


def _entity_text(doc) -> str:
    """Text embedded for a node, e.g. "Phòng Đăng ký kinh doanh (cơ quan nhà nước): <first 3 descriptions>"."""
    entity_type = doc.get("type") or "khái niệm"
    return f"{doc['name']} ({entity_type}): " + " ".join(doc["descriptions"][:3])


def _relation_text(doc, names) -> str:
    source = names.get(doc["source"], doc["source"])
    target = names.get(doc["target"], doc["target"])
    return f"{source} - {target}: {', '.join(doc['keywords'])}. " + " ".join(doc["descriptions"][:2])


def index_graph(upsert_batch=16, start=0, max_attempts=5):
    """Embed merged nodes / edges into Qdrant"""
    db = _graph_db()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    embedder = M3EmbeddingsAPI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL, model=EMBEDDING_MODEL)

    existing = {c.name for c in client.get_collections().collections}
    for name in (ENTITY_COLLECTION, RELATION_COLLECTION):
        if name not in existing:
            client.create_collection(
                name,
                vectors_config=models.VectorParams(size=EMBEDDING_SIZE, distance=models.Distance.COSINE),
            )

    # (key, text to embed, payload) for every node, then every edge; sorted so `start` is stable between runs
    names = {d["_id"]: d["name"] for d in db["graph_entities"].find({}, {"name": 1})}
    entity_items = [
        (d["_id"], _entity_text(d), {"key": d["_id"], "name": d["name"], "type": d.get("type")})
        for d in db["graph_entities"].find().sort("_id", 1)
    ]
    relation_items = [
        (d["_id"], _relation_text(d, names), {"key": d["_id"], "source": d["source"], "target": d["target"]})
        for d in db["graph_relations"].find().sort("_id", 1)
    ]
    jobs = [(ENTITY_COLLECTION, "entity", entity_items), (RELATION_COLLECTION, "relation", relation_items)]

    position = 0  # index over nodes then edges, the unit of `start`
    for collection, kind, items in jobs:
        for i in range(0, len(items), upsert_batch):
            batch_position = position + i
            if batch_position + upsert_batch <= start:
                continue  # whole batch already indexed in a previous run
            batch = items[i:i + upsert_batch]

            for attempt in range(1, max_attempts + 1):
                try:
                    vectors = embedder.embed_documents([text for _, text, _ in batch])
                    client.upsert(collection, points=[
                        # Stable id from the node / edge key: a rerun overwrites instead of duplicating
                        models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{kind}:{key}")),
                                           vector=vector, payload=payload)
                        for (key, _, payload), vector in zip(batch, vectors)
                    ])
                    break
                except Exception as e:
                    if attempt == max_attempts:
                        print(f"Failed at position {batch_position}; rerun index_graph(start={batch_position})")
                        raise
                    wait = 10 * attempt
                    print(f"{kind} batch at {batch_position}: {type(e).__name__}, retry {attempt}/{max_attempts - 1} in {wait}s")
                    time.sleep(wait)

            print(f"{kind}: {min(i + upsert_batch, len(items))} / {len(items)}")
        position += len(items)

    return len(entity_items), len(relation_items)
