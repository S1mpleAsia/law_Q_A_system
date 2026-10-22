from dotenv import load_dotenv

load_dotenv()

import logging
from dataclasses import asdict


# from agent import react_agent_handle
from brain_v2 import (
    assistant,
    detect_user_intent,
    get_legal_agent_anwer,
    generate_conversation_text
)
from assistant import ENABLE_GRAPH_RETRIEVAL
from celery import shared_task
from database import get_celery_app
from models import get_conversation_messages, update_chat_conversation
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logging.getLogger("celery.redirected").setLevel(logging.ERROR)

celery_app = get_celery_app("celery_app")
celery_app.autodiscover_tasks()


@shared_task()
def bot_answer_message(history, question, keywords=None, use_graph=False):
    """Returns (answer, GraphResult or None)."""
    # Retrieval: law texts (hybrid search, plus graph candidates in graph mode, reranked) and graph matches
    top_docs, graph = assistant.retrieval.retrieve(question, keywords, use_graph=use_graph)

    if len(top_docs) > 0:
        logger.info(f"Only call RAG (graph mode: {use_graph}, graph context: {len(graph.context.splitlines())} lines)")
        assistant_answer = assistant.generation.generate(question, top_docs, graph_context=graph.context,
                                                         graph_passages=getattr(graph, "passage_docs", None))
        return assistant_answer, graph

    logger.warning(f"call Tavily search")
    agent_answer = get_legal_agent_anwer(history, question)

    return agent_answer, None


@shared_task()
def bot_route_answer_message(history, question, use_graph=False):
    """Returns (answer, GraphResult or None)."""
    history_messages = generate_conversation_text(history)

    # Graph mode extracts LightRAG keywords in the same rewrite call
    new_question, keywords = detect_user_intent(history_messages, question, extract_keywords=use_graph)
    logger.warning(f"User intent: {new_question} | keywords: {keywords}")

    # detect the route
    result, explanation = assistant.router(history_messages, new_question)
    if not result:
        logger.warning(f"Router to chitchat")
        return explanation, None
    else:
        return bot_answer_message(history_messages, new_question, keywords, use_graph)

@shared_task()
def llm_handle_message(bot_id, user_id, conv_id, question, use_graph=None):
    logger.info("Start handle message")
    # The UI toggle chooses per question; requests without a choice use ENABLE_GRAPH_RETRIEVAL
    if use_graph is None:
        use_graph = ENABLE_GRAPH_RETRIEVAL
    # Update chat conversation for new user_question
    conversation_id = update_chat_conversation(bot_id, user_id, conv_id, question, True)
    logger.info("Conversation id: %s", conversation_id)

    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[-5:-1]
    # Use bot route to handle message
    response, graph = bot_route_answer_message(history, question, use_graph)
    logger.info(f"Chatbot response: {response}")
    # Save response to history
    graph_data = asdict(graph) if use_graph and graph is not None else None
    # Stored with the answer so reloading the conversation still shows the mode and the graph panel
    update_chat_conversation(bot_id, user_id, conversation_id, response, False,
                             extra={"use_graph": use_graph, "graph": graph_data})
    # Return response; "graph" lets the UI show what the knowledge graph contributed
    return {
        "role": "assistant",
        "content": response,
        "chat_id": conversation_id,
        "use_graph": use_graph,
        "graph": graph_data,
    }
