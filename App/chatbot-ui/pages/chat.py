import json
import logging
import time
import os

import requests
import streamlit as st
from menu import menu_with_redirect
from service import fetch_conversation_details
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from streamlit_star_rating import st_star_rating

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

ctx = get_script_run_ctx()
all_app_pages = ctx.pages_manager.get_pages()
try:
    current_page = all_app_pages[ctx.page_script_hash]
except KeyError:
    current_page = [
        p
        for p in all_app_pages.values()
        if p["relative_page_hash"] == ctx.page_script_hash
    ][0]

header = st.container()
header.title(f"🤖 AI Legal assistant")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

### Custom CSS for the sticky header
st.markdown("""
    <style>
    .dark-container {
        padding: 20px;
        border-radius: 10px;
        color: white;
    }

    @media (prefers-color-scheme: dark) {
        .dark-container {
            background-color: #1e1e1e;
        }
    }

    @media (prefers-color-scheme: light) {
        .dark-container {
            background-color: #f0f0f0;
            color: black;
        }
    }
    </style>
""", unsafe_allow_html=True)


# Input bot
bot_id = "botLegal"
user_id = st.experimental_user.email
BACKEND_URL = os.getenv("BACKEND_URL", "http://fastapi_app:8002")
converation_id = st.query_params.get("id", current_page["url_pathname"])



def send_user_request(text, chat_id):
    url = f"{BACKEND_URL}/chat/complete"

    payload = json.dumps(
        {
            "user_message": text,
            "user_id": str(user_id),
            "bot_id": bot_id,
            "conversation_id": st.query_params.get("id", chat_id),
            "use_graph": use_graph,
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
    if response.status_code != 200:
        raise TimeoutError(f"Request to bot fail: {response.text}")
    return json.loads(response.text)


# The backend waits up to 60s for the answer; wait a bit longer so its own timeout reply arrives
BOT_RESPONSE_TIMEOUT = 75


class BotError(Exception):
    """The backend could not produce an answer (task failed or timed out)."""


def get_bot_response(request_id):
    url = f"{BACKEND_URL}/chat/complete_v2/{request_id}"

    response = requests.request("GET", url, headers={}, data="", timeout=BOT_RESPONSE_TIMEOUT)
    if response.status_code != 200:
        raise TimeoutError(f"Get bot response fail: {response.text}")
    return response.status_code, json.loads(response.text)


def get_chat_complete(text, chat_id):
    user_request = send_user_request(text, chat_id)
    request_id = user_request["task_id"]
    _, chat_response = get_bot_response(request_id)
    result = chat_response.get("task_result")
    # A failed or timed-out task has no answer; its reason is in error_message
    if chat_response.get("task_status") != "SUCCESS" or not isinstance(result, dict) or "content" not in result:
        raise BotError(chat_response.get("error_message") or f"task {chat_response.get('task_status')}")
    return result["content"], result["chat_id"], result.get("use_graph"), result.get("graph")


# Streamed response
def response_generator(user_message, chat_id):
    def generate_stream(res_txt):
        for line in res_txt.split("\n\n"):
            logging.info(f"Line: {line}")
            for sen in line.split("\n"):
                yield sen + "\n\n"
                time.sleep(0.05)
            yield "\n"

    res, chat_id, used_graph, graph = get_chat_complete(user_message, chat_id)
    return generate_stream(res), chat_id, used_graph, graph


def _dot_label(text, width=28):
    """Graphviz label: escaped and wrapped so long legal names stay readable."""
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\\n".join(lines)


DIAGRAM_NEIGHBORS = 6  # one-hop edges drawn in the diagram (the tab lists more), keeps it readable


def _graph_dot(graph):
    """Subgraph the query touched: matched concepts (local), matched relations (global), one-hop neighbours."""
    nodes, lines = {}, []

    def node(name, style):
        # First style wins: a matched concept stays highlighted even if it is also a neighbour endpoint
        nodes.setdefault(name, style)

    for e in graph.get("entities") or []:
        node(e["name"], "matched")
    for r in graph.get("relations") or []:
        node(r["source"], "plain")
        node(r["target"], "plain")
        label = _dot_label(", ".join(r["keywords"][:2]), width=22)
        lines.append(f'"{_dot_label(r["source"])}" -> "{_dot_label(r["target"])}" '
                     f'[label="{label}", color="#d9822b", fontcolor="#a35f12", penwidth=2.2];')
    for n in (graph.get("neighbors") or [])[:DIAGRAM_NEIGHBORS]:
        node(n["source"], "plain")
        node(n["target"], "plain")
        label = _dot_label(", ".join(n["keywords"][:1]), width=22)
        lines.append(f'"{_dot_label(n["source"])}" -> "{_dot_label(n["target"])}" '
                     f'[label="{label}", color="#9aa5b1", fontcolor="#7b8794", style=dashed];')
    styles = {
        "matched": 'fillcolor="#d6e9fb", color="#2f6fb0", penwidth=1.6',
        "plain": 'fillcolor="#ffffff", color="#9aa5b1"',
    }
    header = ['digraph G { rankdir=LR; nodesep=0.25; ranksep=0.5; '
              'node [shape=box, style="rounded,filled", fontsize=10, fontname="Helvetica"]; '
              'edge [fontsize=8, fontname="Helvetica"];']
    body = [f'"{_dot_label(name)}" [{styles[style]}];' for name, style in nodes.items()]
    return "\n".join(header + body + lines + ["}"])


def _sourced_text(item):
    """Descriptions of a concept / relation, each followed by the document it was extracted from."""
    descriptions = item.get("descriptions")
    if not descriptions:  # answers saved before descriptions carried their source
        return item.get("description", "")
    return "  \n".join(f"{d['text']} — _{d['source']}_" if d.get("source") else d["text"] for d in descriptions)


def render_graph(graph):
    """How LightRAG answered this question, step by step: keywords, local and global retrieval
    on the knowledge graph, neighbour expansion, source chunks and the context given to the LLM."""
    entities, relations = graph.get("entities") or [], graph.get("relations") or []
    neighbors, chunks = graph.get("neighbors") or [], graph.get("source_chunks") or []
    keywords = graph.get("keywords") or {}

    with st.expander("LightRAG: truy vấn knowledge graph cho câu hỏi này"):
        st.markdown("**① Tách từ khoá** (cùng lần gọi LLM viết lại câu hỏi)")
        col_low, col_high = st.columns(2)
        col_low.markdown("**Low-level** · thực thể cụ thể → tìm *khái niệm*  \n"
                         + " · ".join(f"`{k}`" for k in keywords.get("low_level") or ["(cả câu hỏi)"]))
        col_high.markdown("**High-level** · chủ đề, quan hệ → tìm *quan hệ*  \n"
                          + " · ".join(f"`{k}`" for k in keywords.get("high_level") or ["(cả câu hỏi)"]))

        st.markdown("**② Đồ thị con được truy vấn**")
        if entities or relations:
            st.graphviz_chart(_graph_dot(graph), use_container_width=True)
            st.caption("Ô xanh: khái niệm tìm bằng low-level (local). Mũi tên cam: quan hệ tìm bằng high-level (global). "
                       "Mũi tên xám đứt: quan hệ lân cận 1 bước của các khái niệm tìm được.")
        else:
            st.info("Graph không tìm được khái niệm hay quan hệ nào cho câu hỏi này.")

        tab_local, tab_global, tab_hop, tab_chunks, tab_context = st.tabs([
            f"③ Khái niệm · local ({len(entities)})",
            f"④ Quan hệ · global ({len(relations)})",
            f"⑤ Lân cận 1 bước ({graph.get('neighbor_count', len(neighbors))})",
            f"⑥ Đoạn luật nguồn ({len(chunks)})",
            "⑦ Ngữ cảnh gửi LLM",
        ])
        with tab_local:
            st.caption("Khái niệm có mô tả gần nhất với từ khoá low-level (độ tương đồng cosine).")
            for e in entities:
                st.markdown(f"`{e['score']:.2f}` **{e['name']}** · _{e['type']}_  \n{_sourced_text(e)}")
        with tab_global:
            st.caption("Quan hệ có mô tả gần nhất với từ khoá high-level (độ tương đồng cosine).")
            for r in relations:
                st.markdown(f"`{r['score']:.2f}` **{r['source']}** → **{r['target']}** · "
                            f"_{', '.join(r['keywords'])}_  \n{_sourced_text(r)}")
        with tab_hop:
            st.caption(f"Từ các khái niệm ở bước ③, đi theo cạnh sang khái niệm kề bên: "
                       f"{graph.get('neighbor_count', len(neighbors))} quan hệ. Dưới đây là {len(neighbors)} quan hệ cụ thể nhất.")
            for n in neighbors:
                st.markdown(f"**{n['source']}** → **{n['target']}** · _{', '.join(n['keywords'])}_")
        with tab_chunks:
            st.caption("Mỗi khái niệm và quan hệ ghi lại đoạn luật nó được trích ra. Điểm càng cao, "
                       "đoạn luật càng được nhiều khái niệm / quan hệ cụ thể trỏ tới. Các đoạn này được "
                       "xếp hạng lại cùng kết quả hybrid search trước khi đưa cho LLM.")
            for c in chunks:
                st.markdown(f"`{c['score']:.2f}` **{c['title']}** — {c['article']}")
        with tab_context:
            passages = graph.get("passages") or []
            st.caption("Phần graph được ghép vào prompt, cạnh các điều luật đã tìm được. "
                       "Mỗi dòng ghi văn bản nguồn trong [ngoặc vuông].")
            if passages:
                st.markdown("**Nguyên văn đoạn luật nguồn của các quan hệ khớp nhất** (thêm vào prompt):  \n"
                            + "  \n".join(f"• **{p['title']}** — {p['article']}" for p in passages))
            st.code(graph.get("context") or "(trống)", language=None, wrap_lines=True)

if (
    converation_id
    and converation_id != "new_chat"
    and st.experimental_user.is_logged_in
):
    # Initialize chat history
    history = fetch_conversation_details(converation_id, st.experimental_user.email)
    st.session_state.messages = history["result"]

if "messages" not in st.session_state or converation_id == "new_chat":
    st.session_state.messages = []

def show_answer_mode(used_graph, graph):
    """Under an answer: which mode produced it and, in graph mode, the LightRAG panel."""
    if used_graph:
        st.caption("🔗 Knowledge graph context")
        if graph:
            render_graph(graph)
    elif used_graph is False:
        st.caption("Hybrid search")


# Display chat messages from history on app rerun (the graph panel is stored with each answer,
# so switching mode or reloading the page keeps it)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_answer_mode(message.get("use_graph"), message.get("graph"))

# Answer mode for the next question, like choosing "deep research" before sending a message
use_graph = st.toggle(
    "🔗 Knowledge graph cho câu hỏi tiếp theo",
    value=True,
    key="use_graph",
    help="Bật: tìm thêm khái niệm và quan hệ trong knowledge graph. Tắt: chỉ dùng hybrid search. "
         "Mỗi câu trả lời ghi lại chế độ đã dùng.",
)

# Accept user input
if prompt := st.chat_input("Bạn muốn hỏi gì về luật liên quan đến doanh nghiệp?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        searching = st.markdown(" Searching...")
        try:
            resp_message, chat_id, used_graph, graph = response_generator(prompt, converation_id)
        except (BotError, TimeoutError, requests.RequestException) as e:
            searching.empty()
            st.error(f"Không trả lời được câu hỏi này, vui lòng thử lại. Chi tiết: {e}")
            st.stop()
        searching.empty()
        response = st.write_stream(resp_message)
        show_answer_mode(used_graph, graph)
        stars = st_star_rating("Bạn có hài lòng với câu trả lời không?",
                               maxValue=5, defaultValue=3, size=21, key="rating",
                               customCSS="h3 {font-size: 15px}")
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "use_graph": used_graph, "graph": graph}
    )
    if converation_id == "new_chat":
        st.query_params["id"] = chat_id
    # st.rerun()
