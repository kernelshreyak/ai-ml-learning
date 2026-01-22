import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-4.1-mini"
MAX_ROUNDS = 3
UNDERSTANDING_THRESHOLD = 0.6


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.environ["OPENAI_API_KEY"],
    )


student_llm = make_llm()
teacher_llm = make_llm()


class StudentAssessment(BaseModel):
    comment: str = Field(
        description="Student's response describing understanding and any follow-up questions."
    )
    understanding_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the student understood the teacher's last answer.",
    )


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    attempts: int
    score: float


def student_node(state: ConversationState) -> ConversationState:
    messages = [
        SystemMessage(
            content=(
                "You are a student willing to learn but are naive. Provide a clear, "
                "short response describing what you understood and any follow-up question. "
                "Return structured output with a comment and an understanding_score."
            )
        )
    ] + state["messages"]
    assessment = student_llm.with_structured_output(StudentAssessment).invoke(messages)

    reply = AIMessage(content=assessment.comment, name="Student")
    return {
        "messages": [reply],
        "attempts": state["attempts"] + 1,
        "score": assessment.understanding_score,
    }


def teacher_node(state: ConversationState) -> ConversationState:
    messages = [
        SystemMessage(
            content="You are an expert professor and can only answer at PhD level regardless of question asked. Also you are bit sarcastic always"
        )
    ] + state["messages"]
    raw_reply = teacher_llm.invoke(messages)
    reply = AIMessage(content=raw_reply.content, name="Teacher")
    return {"messages": [reply]}


def route_after_student(state: ConversationState) -> str:
    # Always allow at least 2 student assessments before considering early stop.
    if state["attempts"] < 2:
        return "teacher"
    if state["score"] < UNDERSTANDING_THRESHOLD and state["attempts"] < MAX_ROUNDS:
        return "teacher"
    return END


graph = StateGraph(ConversationState)
graph.add_node("student", student_node)
graph.add_node("teacher", teacher_node)
graph.add_edge(START, "teacher")
graph.add_edge("teacher", "student")
graph.add_conditional_edges(
    "student", route_after_student, {"teacher": "teacher", END: END}
)

app = graph.compile()

initial_question = "What are banach spaces and how does it relate to topology?"

initial_input = {
    "messages": [HumanMessage(content=initial_question)],
    "attempts": 0,
    "score": 0.0,
}

print(f"[Student] {initial_question}")
seen_messages = 1
final_state = None
for state in app.stream(initial_input, stream_mode="values"):
    final_state = state
    messages = state["messages"]
    if len(messages) > seen_messages:
        for message in messages[seen_messages:]:
            if isinstance(message, AIMessage):
                speaker = message.name or "Teacher"
                print(f"[{speaker}] {message.content}")
        seen_messages = len(messages)
