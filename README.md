# 🚀 Vietnamese Enterprise Law Q&A System

A chatbot that answers questions about Vietnamese enterprise law (Luật Doanh nghiệp and related decrees and circulars) with Retrieval-Augmented Generation. The answers cite the law articles they are based on.

- **Corpus:** 1,367 legal documents crawled from [vbpl.vn](https://vbpl.vn). They are split into 5,317 parent chunks (MongoDB) and 29,452 child chunks (Qdrant).
- **Retrieval:** hybrid search (BGE-M3 dense + BM25 sparse, RRF fusion), then reranking.
- **Knowledge graph (LightRAG style):** 4,715 entities and 6,504 relations extracted from the law texts. A toggle in the UI turns graph retrieval on or off for each question.

## Architecture

```
Streamlit UI (question + knowledge-graph toggle)
  → FastAPI → Redis → Celery worker
  → ① Gemini: rewrite the question (+ low-level / high-level keywords when the graph is on)
  → ②a Hybrid search                          ②b Knowledge-graph retrieval
       dense + sparse on child chunks (Qdrant)      low-level keywords → 6 entities
       → their parent chunks (MongoDB)              high-level keywords → 6 relations
                                                    + 1-hop edges → vote for ≤ 8 parents
  → ③ Rerank (Pinecone, bge-reranker-v2-m3): candidates from ②a + ②b → top 5 parents, score ≥ 0.5
  → ④ Gemini answers from the 5 parents (+ graph context and up to 2 source passages)
```

| Component | Service |
| --- | --- |
| LLM (rewrite, answer, graph extraction) | Gemini `gemini-3.5-flash-lite` |
| Embedding | BGE-M3 through Cloudflare Workers AI (OpenAI-compatible API) |
| Sparse vectors | FastEmbed `Qdrant/bm25` (local) |
| Reranker | Pinecone inference `bge-reranker-v2-m3` |
| Vector store | Qdrant Cloud |
| Parent chunks, knowledge graph, chat history | MongoDB Atlas |
| Task queue | Celery + Redis |

### Chunking
- **Parent chunks** are split on Phần / Chương / Mục / Điều, up to 1024 words, and stored verbatim in MongoDB `law.documents`. The LLM reads these. Their ids are `<document id>_<index>`.
- **Child chunks** are split on Điều / khoản / điểm, up to 255 words, and stored in Qdrant with dense and sparse vectors. Search runs on these, and each child points to its parent through `metadata.doc_id`.

### Knowledge graph
Built offline by `Core/build_graph.ipynb` (code in `search_document_v2/graph_builder.py`) from the 2,880 parent chunks that are still in force.

1. **Extract:** Gemini reads one parent chunk at a time and returns JSON with:
   - entities: name, one of 12 legal types, and a 1–2 sentence description citing the article;
   - relations: source, target, keywords and a description.

   Results are cached in `graph_extractions`, so the extraction can resume after an interruption or a quota limit.
2. **Merge:** entities with the same normalized name are merged; abbreviations such as TNHH and HĐQT are expanded first. Relations are merged on (source, target). Each entity and relation keeps up to 8 descriptions, the parent of each description (`description_sources`) and all its source parents (`parent_ids`). The output is MongoDB `graph_entities` and `graph_relations`.
3. **Index:** the text `"name (type): descriptions"` of each entity and `"source - target: keywords. descriptions"` of each relation are embedded with BGE-M3. They are stored in Qdrant `<COLLECTION_NAME>_entities` and `<COLLECTION_NAME>_relations`.

At query time, `search_document_v2/graph_retriever.py` works as follows:
1. It searches entities with the low-level keywords and relations with the high-level keywords.
2. It adds every edge touching the matched entities (1 hop).
3. Each match votes for its `parent_ids`, with a weight divided by √(number of parents).
4. It passes the top 8 parents to the reranker together with the hybrid-search candidates.
5. It gives Gemini the matched descriptions, labelled with their source document, and up to 2 verbatim source passages.

## Project structure

```
├── App
│   ├── backend/
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── .env.example                 # environment variables (copy to .env)
│   │       ├── app.py                       # FastAPI: /chat/complete, /conversations, auth
│   │       ├── tasks.py                     # Celery tasks: route the question, retrieve, answer
│   │       ├── brain_v2.py                  # question rewrite + keyword extraction, routing, web search agent
│   │       ├── assistant.py                 # Retrieval (hybrid + graph + rerank) and Generation (Gemini)
│   │       ├── models.py, cache.py          # chat history (MongoDB)
│   │       ├── auth.py, config.py, database.py, utils.py
│   │       └── search_document_v2/
│   │           ├── custom_retriever.py      # parent/child chunking, hybrid search
│   │           ├── vector_store.py          # Qdrant collection, BGE-M3 + BM25 embeddings
│   │           ├── custom_embedding.py      # embedding through an OpenAI-compatible API
│   │           ├── mongodb_store.py         # parent chunk store
│   │           ├── graph_builder.py         # knowledge graph: extract, merge, index
│   │           ├── graph_retriever.py       # knowledge graph retrieval
│   │           ├── tavily_search.py         # web search for questions outside the corpus
│   │           └── BGE_M3/modeling.py       # local BGE-M3 (used when no embedding API key is set)
│   └── chatbot-ui/
│       ├── requirements.txt
│       ├── .streamlit/                      # pages_sections.toml, secrets.toml (Google login)
│       ├── main.py, menu.py, service.py
│       └── pages/                           # home.py, chat.py (chat + LightRAG panel), new_chat.py
├── Core
│   ├── crawl_law_docs.ipynb                 # crawl the corpus from vbpl.vn → Data/doc/law_corpus.json
│   ├── crawl_data.ipynb                     # crawler for the original (old) vbpl.vn site
│   ├── deploy.ipynb                         # index the corpus into MongoDB + Qdrant, test the assistant
│   └── build_graph.ipynb                    # build the knowledge graph, query examples
├── BGE_M3/                                  # BGE-M3 fine-tuning code
├── Data/                                    # corpus, train/test data (download from Releases)
├── bge-m3-output/, bge-m3-reranker-output/  # fine-tuned checkpoints (download separately)
├── finetune_eval.ipynb                      # fine-tune and evaluate BGE-M3 and the reranker
├── preprocessing.ipynb                      # build the training / test data
└── test.ipynb                               # demo: chunking, indexing and hybrid search on one document
```

## Setup

### 1. Python environment
Python 3.11. Install everything in one virtual environment at the repo root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r App/backend/requirements.txt -r App/chatbot-ui/requirements.txt ipykernel
python -m ipykernel install --user --name law-qa --display-name "law-qa (.venv)"
```

### 2. Environment variables
```bash
cp App/backend/src/.env.example App/backend/src/.env
```
Fill in the keys. Every variable is described in `.env.example`.

### 3. Streamlit (Google login)
`App/chatbot-ui/.streamlit/secrets.toml`:
```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "<random string>"
client_id = "<Google OAuth client id>"
client_secret = "<Google OAuth client secret>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

### 4. Redis
```bash
brew install redis && brew services start redis
# or: docker run -d --name redis -p 6379:6379 redis:7-alpine
```

## Build the data (once)

Use the `law-qa (.venv)` kernel for the notebooks.

1. **Corpus:** download `Data.zip` from Releases and extract it into `Data/`, or crawl it again with `Core/crawl_law_docs.ipynb`.
2. **Index:** run `Core/deploy.ipynb`. It indexes `Data/doc/law_corpus.json` into MongoDB and Qdrant (5,317 parents, 29,452 children). Use `START` / `LIMIT` to index in several runs; re-indexing a document overwrites it.
3. **Knowledge graph:** run `Core/build_graph.ipynb` in the order Extract → Merge → Index.
   - Extraction makes one Gemini call per parent chunk (about 2,880 calls). Rerun the cell to continue after a quota limit.
   - Merge and Index do not call the LLM.

## Run the app

Run each command in its own terminal, from the repo root, with `.venv` activated:

```bash
# 1. Backend API (http://localhost:8002)
cd App/backend/src && python app.py

# 2. Celery worker
cd App/backend/src && celery -A tasks.celery_app worker --pool=solo --loglevel=info

# 3. UI (http://localhost:8501)
cd App/chatbot-ui && BACKEND_URL=http://localhost:8002 streamlit run main.py --server.port 8501
```

Restart the Celery worker after changing backend code, and make sure only one worker is running.

In the chat page, the **🔗 Knowledge graph (LightRAG)** toggle applies to the next question. When it is on, each answer has a panel that shows the LightRAG steps:
1. extracted keywords;
2. graph diagram;
3. matched entities (local);
4. matched relations (global);
5. 1-hop neighbors;
6. source parent chunks;
7. the context sent to Gemini.

## Results

### Retrieval models (Bảng 4.1 of the report)
2,048 test questions. The positives are the answer passages of each question, mixed into the corpus. This measures the embedding / rerank models offline, on the fine-tuned BGE-M3 (`finetune_eval.ipynb`), not the deployed pipeline.

|Model                    | recall@3  | recall@5  | recall@10 | acc@3     | acc@5     | acc@10    |
|-------------------------|-----------|-----------|-----------|-----------|-----------|-----------|
|BGE-M3 fine-tuned        | 0.6264    | 0.7116    | 0.8007    | 0.7856    | 0.8589    | 0.917     |
|BGE-M3 + rerank          | 0.6409    | 0.7259    | 0.8077    | 0.8047    | 0.875     | 0.9199    |
|BGE-M3 + BM25 + rerank   | 0.6506    | 0.7299    | 0.814     | 0.8101    | 0.8735    | 0.9209    |

### Knowledge graph
21 multi-hop questions about Luật Doanh nghiệp 2020, with 38 target articles, run through the full pipeline once with the graph off and once with it on:

| | Graph off | Graph on |
| --- | --- | --- |
| Target articles in the texts given to Gemini | 26 / 38 | 29 / 38 |
| Target articles cited in the answer | 24 / 38 | 28 / 38 |

All the differences come from 6 of the 21 questions, so the sample is too small for a firm conclusion.

### Demo
![image](https://github.com/user-attachments/assets/8a3ee0a7-93e9-4439-b7c8-831e83864a1b)
