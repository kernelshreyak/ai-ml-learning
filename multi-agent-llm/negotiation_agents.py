import os
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-4.1-mini"

# Negotiation knobs
LIST_PRICE = 12000.0
MIN_PRICE = 850.0
MAX_CYCLES = 6
MIN_BUYER_DECLINE_ROUNDS = 1
DELAY_SECONDS = 2.0
# LANGUAGE = "normal desi hindi north-indian dialect (english font)"
# LANGUAGE = "southern american"
LANGUAGE = "english"

# Prompt knobs
MARKET_CONTEXT = (
    "Product: lightly used piano (200 years old, okay-ish condition).\n"
    "Market: New models just launched, so used prices are softening.\n"
    "Demand: low; several similar listings exist locally.\n"
    "Urgency: seller wants to close within a day; buyer has other options."
)
SELLER_PROMPT = (
    "You are a professional seller. Your list price is ${list_price}. "
    "You may lower your price, but never below ${min_price}. "
    "Be firm but polite; always provide a concrete price offer each turn."
    "get more agressive and much more angry as the turns increase"
    f"Always prefer to use language: {LANGUAGE}"
    "\nContext:\n{market_context}"
)
BUYER_PROMPT = (
    "You are a value-focused buyer. Your goal is to negotiate the lowest possible price. "
    "Be persuasive, concise, and propose a concrete price - at times be agressive though not outright rude."
    "Dont use same leverage and try to escalate moving forward in turns to convince the seller to re-evaluate their price"
    f"Always prefer to use language: {LANGUAGE}"
    "\nContext:\n{market_context}"
)


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    cycles: int
    last_offer: float
    agreed: bool
    buyer_walked: bool
    final_price: Optional[float]


class SellerOffer(BaseModel):
    message: str = Field(description="Seller response to the buyer.")
    offer_price: float = Field(ge=0.0, description="Current price offer from seller.")
    agreed: bool = Field(description="Whether the seller accepts the buyer's proposal.")


class BuyerResponse(BaseModel):
    message: str = Field(description="Buyer response to the seller offer.")
    proposed_price: float = Field(
        ge=0.0, description="Buyer counter-offer or accepted price."
    )
    accept: bool = Field(description="Whether the buyer accepts the seller's offer.")
    disagree: bool = Field(
        description="Whether the buyer rejects the negotiation and walks away."
    )


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.environ["OPENAI_API_KEY"],
    )


seller_llm = make_llm()
buyer_llm = make_llm()


def seller_node(state: ConversationState) -> ConversationState:
    messages = [
        SystemMessage(
            content=SELLER_PROMPT.format(
                list_price=LIST_PRICE,
                min_price=MIN_PRICE,
                market_context=MARKET_CONTEXT,
            )
        )
    ] + state["messages"]
    offer = seller_llm.with_structured_output(SellerOffer).invoke(messages)
    offer_price = max(offer.offer_price, MIN_PRICE)
    reply = AIMessage(
        content=f"{offer.message}\nOffer: ${offer_price:.2f}",
        name="Seller",
    )
    return {
        "messages": [reply],
        "last_offer": offer_price,
        "agreed": offer.agreed,
        "final_price": offer_price if offer.agreed else None,
    }


def buyer_node(state: ConversationState) -> ConversationState:
    if state["agreed"]:
        reply = AIMessage(
            content=f"Accepted: ${state['last_offer']:.2f}",
            name="Buyer",
        )
        return {
            "messages": [reply],
            "cycles": state["cycles"] + 1,
            "agreed": True,
            "buyer_walked": False,
            "final_price": state["last_offer"],
        }
    messages = [
        SystemMessage(content=BUYER_PROMPT.format(market_context=MARKET_CONTEXT))
    ] + state["messages"]
    response = buyer_llm.with_structured_output(BuyerResponse).invoke(messages)
    accept = response.accept
    buyer_walked = response.disagree
    if state["cycles"] < MIN_BUYER_DECLINE_ROUNDS and accept:
        accept = False
        buyer_walked = False
    proposed_price = response.proposed_price
    if proposed_price <= 0:
        proposed_price = max(MIN_PRICE, state["last_offer"] - 50.0)
    reply_text = response.message
    if buyer_walked:
        reply_text += "\nDecision: Walk away"
    elif accept:
        reply_text += f"\nAccepted: ${state['last_offer']:.2f}"
    else:
        reply_text += f"\nCounter: ${proposed_price:.2f}"
    reply = AIMessage(content=reply_text, name="Buyer")
    return {
        "messages": [reply],
        "cycles": state["cycles"] + 1,
        "agreed": accept,
        "buyer_walked": buyer_walked,
        "final_price": state["last_offer"] if accept else None,
    }


def route_after_buyer(state: ConversationState) -> str:
    if state["agreed"]:
        return END
    if state["buyer_walked"]:
        return END
    if state["cycles"] >= MAX_CYCLES:
        return END
    return "seller"


graph = StateGraph(ConversationState)
graph.add_node("seller", seller_node)
graph.add_node("buyer", buyer_node)
graph.add_edge(START, "seller")
graph.add_conditional_edges(
    "seller",
    lambda state: END if state["agreed"] else "buyer",
    {"buyer": "buyer", END: END},
)
graph.add_conditional_edges("buyer", route_after_buyer, {"seller": "seller", END: END})

app = graph.compile()

initial_state: ConversationState = {
    "messages": [],
    "cycles": 0,
    "last_offer": LIST_PRICE,
    "agreed": False,
    "buyer_walked": False,
    "final_price": None,
}

print("[System] Negotiation started")
seen_messages = 0
final_state = None
for state in app.stream(initial_state, stream_mode="values"):
    final_state = state
    messages = state["messages"]
    if len(messages) > seen_messages:
        for message in messages[seen_messages:]:
            if isinstance(message, AIMessage):
                speaker = message.name or "Assistant"
                print(f"[{speaker}] {message.content}")
        seen_messages = len(messages)
        time.sleep(DELAY_SECONDS)

if final_state and final_state["final_price"] is not None:
    print(f"[System] Final price agreed: ${final_state['final_price']:.2f}")
elif final_state and final_state["buyer_walked"]:
    print("[System] Negotiation ended: buyer walked away.")
else:
    last_offer = (
        final_state["last_offer"] if final_state else initial_state["last_offer"]
    )
    print(
        f"[System] Negotiation ended without agreement. Last offer: ${last_offer:.2f}"
    )
