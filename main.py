from typing import Annotated
from uuid import uuid4

from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from dataset_store import DatasetStore

llm = ChatOllama(model="huggingface.co/ShallowLearning/autotune-gemma:latest", reasoning=False)


class State(MessagesState):
    messages: Annotated[list, add_messages]


def conversation_agent(state: State) -> State:
    system_message = SystemMessage(
        content="You are a helpful assistant that can answer questions and help with tasks."
    )
    response = llm.invoke([system_message] + state["messages"])
    return {"messages": [response]}


graph = StateGraph(State)
graph.add_node("agent", conversation_agent)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

checkpointer = InMemorySaver()
agent = graph.compile(checkpointer=checkpointer)

def run_chat() -> None:
    thread_id = str(uuid4())
    store = DatasetStore()

    while True:
        user_input = input("Type your message: ")
        if user_input == "exit":
            break

        store.save_message(thread_id, "user", user_input)
        msg = [HumanMessage(content=user_input)]

        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke({"messages": msg}, config)
        assistant_message = result["messages"][-1]
        store.save_message(thread_id, "assistant", assistant_message.content)
        assistant_message.pretty_print()
    
if __name__ == "__main__":
    run_chat()
