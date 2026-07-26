from typing import Annotated
from uuid import uuid4
import os
import sqlite3
from dotenv import load_dotenv

from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langgraph.graph import MessagesState, START, END, StateGraph

from dataset_store import DatasetStore, DEFAULT_DB_PATH

load_dotenv()
db = SQLDatabase.from_uri("sqlite:///finetune.db")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))

class State(MessagesState):
    episodic: list
    semantic: list
    composite: list

def generate_episodic_memory(state: State):
    ep_system_prompt = """
    Extract all the episodic memories from the conversation history.
    """
    pass

def load_conversations() -> list[dict]:
    """Group every stored message into per-thread, chronologically ordered turns."""
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT thread_id, role, content FROM messages ORDER BY thread_id, id"
        ).fetchall()
    finally:
        conn.close()

    conversations: dict[str, list[dict]] = {}
    for thread_id, role, content in rows:
        conversations.setdefault(thread_id, []).append({"role": role, "content": content})

    return [{"thread_id": tid, "turns": turns} for tid, turns in conversations.items()]


if __name__ == "__main__":
    for convo in load_conversations():
        print(f"Thread {convo['thread_id']}")
        for turn in convo["turns"]:
            print(f"  {turn['role']}: {turn['content']}")
        print("-" * 40)