from typing import Annotated
from uuid import uuid4
import os
import sqlite3
from dotenv import load_dotenv

from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from pydantic import BaseModel, Field

from dataset_store import DEFAULT_DB_PATH, get_watermark, set_watermark

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))

def query_db(watermark: int):
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    rows = conn.execute(
        """
        SELECT thread_id,
               GROUP_CONCAT(role || ': ' || content, char(10) || char(10) ORDER BY id) AS conversation
        FROM messages
        WHERE id > ?
        GROUP BY thread_id
        """,
        (watermark,),
    ).fetchall()
    new_max = conn.execute(
        "SELECT MAX(id) FROM messages WHERE id > ?", (watermark,)
    ).fetchone()[0]
    conn.close()
    return rows, new_max or watermark

class State(MessagesState):
    all_messages: list
    ep_memory: list
    sem_memory: list
    composite_memory: list

class MemoryTag(BaseModel):
    context: str = Field(description="The context of the memory.")
    memory: str = Field(description="The memory itself.")
    topics: list = Field(description="Tag the memory with rellevant keywords or topics.")

class MemoryList(BaseModel):
    memories: list[MemoryTag] = Field(description="All memories extracted from the conversation.")

def gen_episodic_memory(state: State):
    ep_system = SystemMessage(content="""
    Extract EPISODIC memories — specific things that happened in this conversation:
    what the user said, asked, or did, framed as an event. Focus on the user, not yourself.
    """)
    ep_memory = []
    for msg in state["all_messages"]:
        ep_prompt = [ep_system] + [HumanMessage(content=f"""Here is a transcript between a user and an AI assistant. Analyze it; do not continue it.
        
        <transcript>
        {msg}
        </transcript>""")]
        response = llm.with_structured_output(MemoryList).invoke(ep_prompt)
        ep_memory.extend(response.memories)
    return {"ep_memory": ep_memory}

def gen_semantic_memory(state: State):
    sem_system = SystemMessage(content="""
    Extract SEMANTIC memories — durable standing facts about the user that stay true
    after this conversation ends (profile, preferences, stable traits). Not events, not
    facts about yourself. Example: "the user is 12" is semantic; "the user told me their
    age today" is episodic.
    """)
    sem_memory = []
    for msg in state["all_messages"]:
        sem_prompt = [sem_system] + [HumanMessage(content=f"""
        
        <transcript>
        {msg}
        </transcript>""")]
        response = llm.with_structured_output(MemoryList).invoke(sem_prompt)
        sem_memory.extend(response.memories)
    return {"sem_memory": sem_memory}

class DedupeMemory(BaseModel):
    ep_mem_idx: list[int] = Field(description="The index of the episodic memory to delete. 0 idx starting from the fist episodic memory.")
    sem_mem_idx: list[int] = Field(description="The index of the semantic memory to delete. 0 idx starting from the fist semantic memory.")

def dedupe_memory(state: State):
    check_dedupes = """"""
    dedupe_system = [SystemMessage(content="Return the index of memories that need to be deleted because they are duplicative")]
    for ep_memory in state["ep_memory"]:
        check_dedupes += f"{ep_memory}\n"
    for sem_memory in state["sem_memory"]:
        check_dedupes += f"{sem_memory}\n"
    response = llm.with_structured_output(DedupeMemory).invoke(dedupe_system + [HumanMessage(content=f"""
    Memories:

    {check_dedupes}""")])
    ep_drop = set(response.ep_mem_idx)
    sem_drop = set(response.sem_mem_idx)
    ep_kept = [m for i, m in enumerate(state["ep_memory"]) if i not in ep_drop]
    sem_kept = [m for i, m in enumerate(state["sem_memory"]) if i not in sem_drop]
    return {"ep_memory": ep_kept, "sem_memory": sem_kept}

class CompositeMemory(BaseModel):
    context: str = Field(description="The context of the composite memory.")
    composite_memory: str = Field(description="The composite memory.")
    topics: list = Field(description="Tag the composite memory with rellevant keywords or topics.")

class CompositeMemoryList(BaseModel):
    memories: list[CompositeMemory] = Field(description="All composite memories extracted from the conversation.")

def composite_memory(state: State):
    check_composite = """"""
    composite_system = [SystemMessage(content="""
    Rigorously analyze these episodic and semantic memories and form composite memories that build on top and provide useful new information from these stand-alone memories. 
    Don't add what is not necessary and what does not add additional value.""")]
    for ep_memory in state["ep_memory"]:
        check_composite += f"{ep_memory}\n"
    for sem_memory in state["sem_memory"]:
        check_composite += f"{sem_memory}\n"
    response = llm.with_structured_output(CompositeMemoryList).invoke(composite_system + [HumanMessage(content=f"""
    Memories:

    {check_composite}""")])
    return {"composite_memory": response.memories}

graph = StateGraph(State)
graph.add_node("ep_memory", gen_episodic_memory)
graph.add_node("sem_memory", gen_semantic_memory)
graph.add_node("dedupe_memory", dedupe_memory)
graph.add_node("composite_memory", composite_memory)

graph.add_edge(START, "ep_memory")
graph.add_edge(START, "sem_memory")
graph.add_edge("ep_memory", "dedupe_memory")
graph.add_edge("sem_memory", "dedupe_memory")
graph.add_edge("ep_memory", "composite_memory")
graph.add_edge("sem_memory", "composite_memory")
graph.add_edge("dedupe_memory", END)
graph.add_edge("composite_memory", END)

mem_extract = graph.compile()

if __name__ == "__main__":
    with open("memory_graph.png", "wb") as f:
        f.write(mem_extract.get_graph().draw_mermaid_png())

    watermark = get_watermark("extract_watermark")
    rows, new_max = query_db(watermark)
    messages = [msg for _thread_id, msg in rows]
    if not messages:
        print("No new messages to process.")
    else:
        result = mem_extract.invoke({"all_messages": messages})
        for ep_memory in result["ep_memory"]:
            print(f"Episodic memory: {ep_memory}\n\n")
        for sem_memory in result["sem_memory"]:
            print(f"Semantic memory: {sem_memory}\n\n")
        for composite_memory in result["composite_memory"]:
            print(f"Composite memory: {composite_memory}\n\n")
        set_watermark("extract_watermark", new_max)
