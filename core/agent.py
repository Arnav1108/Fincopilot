#agent.py

import os 
from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage,BaseMessage
from langgraph.graph import StateGraph, START , END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from core.tools import TOOLS
from langchain_core.messages import ToolMessage
from core.tools.finance_tools import calculate_metrics
from core.tools.news_tools import fetch_news
from core.rag.rag import ingest
from core.tools.rag_tools import search_documents
from typing import Literal
import tiktoken
from core.router import classify_intent
from core.rag.rag import ingest, list_indexes
from core.memory.db import get_conversation_summary, update_conversation_summary
from core.memory.db import (
    save_message,
    load_messages,
    update_conversation_title,
)
import re
import json

load_dotenv()


SYSTEM_PROMPT = """You are FinCopilot, an expert AI financial analyst assistant.

Your capabilities:
  📄 Search documents - use search_documents tool
  📰 Fetch news - use fetch_news tool
  📊 Analyze sentiment - use analyze_sentiment tool
  🧮 Calculate metrics - use calculate_metrics tool
  📄 Fetch SEC filings - use fetch_sec_10k tool
  
Rules:
  - ALWAYS use search_documents before answering document questions
  -AlWAYS use fetch_news when asked for news
  - Every factual claim derived from documents MUST include a citation.
  - Give concise, synthesized answers, not raw excerpts
  - If you don't know, say so clearly
  - Use fetch_sec_10k when user explicitly requests a filing
  
  When presenting financial figures:
- Always format numbers with commas
- Use consistent units (e.g., $416.2B)
- Present multi-year data in bullet format


When answering document-based financial questions:

You MUST format your response exactly as follows:

Title: <Short Title>

Analysis:
<2–4 complete sentences>

Observations / Risks:
- Bullet point
- Bullet point
- Bullet point

Source:
(filename – p.X)

Rules:
- Each section must start on a new line.
- Use "-" for bullets (not •).
- Leave one blank line between sections.
- Never merge sections into a single paragraph.
- Never repeat numeric values.
- Always include "$" before monetary values.


"""


PLANNER_PROMPT = """
You are a financial research planner.

Break the user query into structured execution steps.

Available tools:
- search_documents
- fetch_news
- calculate_metrics

Return ONLY valid JSON in this format:

{
  "steps": [
    {"tool": "search_documents", "query": "..."},
    {"tool": "fetch_news", "query": "..."},
    {"action": "synthesize"}
  ]
}

Rules:
- Do not explain.
- Do not add text outside JSON.
- Always end with an action step.
"""


def count_tokens(messages, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0

    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            total_tokens += len(encoding.encode(msg.content))

    return total_tokens


def summarize_messages(messages: list[BaseMessage]) -> str:
    """
    Summarizes older conversation history into compressed memory.
    """

    text_block = "\n".join(
        f"{type(msg).__name__}: {msg.content}"
        for msg in messages
        if hasattr(msg, "content") and msg.content
    )

    prompt = f"""
You are summarizing a financial conversation.

Summarize the key financial facts, decisions, and context.
Preserve:
- Companies mentioned
- Financial figures
- Decisions taken
- Important analytical context

Keep it concise but information-dense.
"""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=text_block)
    ])

    return response.content



class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    plan: dict
    tool_results: list
    pending_sec_path: str | None
    pending_sec_meta: dict | None
    analysis: dict
    risk_diff: dict
    comparison_results: list | None



def get_llm() -> ChatOpenAI:

    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=1000,
        presence_penalty=0,
        frequency_penalty=0,
        streaming=False
    ).bind_tools(TOOLS)


def parse_financial_inputs(user_text: str) -> dict | None:
    """
    Uses small LLM call to extract structured financial inputs.
    Returns validated dict or None if parsing fails.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    schema_prompt = """
Extract the following financial values from the user text.
Return ONLY valid JSON with these exact keys:

{
  "revenue": float,
  "net_income": float,
  "total_assets": float,
  "total_equity": float,
  "shares_outstanding": float,
  "current_price": float
}

Rules:
- Convert B or billion to full number (e.g., 416B → 416000000000)
- Convert M or million accordingly
- If any value is missing, return null for that field.
- Do NOT explain.
- Output JSON only.
"""

    response = llm.invoke([
        SystemMessage(content=schema_prompt),
        HumanMessage(content=user_text)
    ])

    try:
        parsed = json.loads(response.content)

        required_keys = {
            "revenue",
            "net_income",
            "total_assets",
            "total_equity",
            "shares_outstanding",
            "current_price",
        }

        if not required_keys.issubset(parsed.keys()):
            return None

        # Ensure no missing values
        if any(parsed[k] is None for k in required_keys):
            return None

        return parsed

    except Exception:
        return None


def AgentNode(state:AgentState)->dict:

    llm = get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    responce = llm.invoke(messages)
    return {"messages":[responce]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def RouterNode(state: AgentState) -> dict:
    user_message = state["messages"][-1].content
    intent = classify_intent(user_message)

    # 🔥 Deterministic override for simple document queries
    if re.search(r"(20\d{2})", user_message):
        if any(word in user_message.lower() for word in ["revenue", "net income", "cash flow"]):
            if "compare" not in user_message.lower():
                intent = "DOCUMENT"

    print("\n================ ROUTER =================")
    print("User Query:", user_message)
    print("Detected Intent:", intent)
    print("=========================================\n")

    return {"intent": intent}


def DeterministicExecutionNode(state: AgentState) -> dict:
    intent = state.get("intent")
    user_message = state["messages"][-1].content

    try:
        if intent == "CALCULATION":
            parsed = parse_financial_inputs(user_message)

            if not parsed:
                return {
                    "messages": [
                        AIMessage(content="⚠️ Could not parse financial inputs. Please provide all required values.")
                    ]
                }

            result = calculate_metrics.invoke(parsed)

            return {
                "messages": [
                    AIMessage(content=result)
                ]
            }

        elif intent == "NEWS":
            result = fetch_news.invoke({"query": user_message})

            return {
                "messages": [
                    AIMessage(content=result)
                ]
            }

    except Exception as e:
        return {
            "messages": [
                AIMessage(content=f"⚠️ Deterministic execution failed: {str(e)}")
            ]
        }

    return {
        "messages": [
            AIMessage(content="⚠️ Could not handle this request deterministically.")
        ]
    }


def PlannerNode(state: AgentState) -> dict:
    user_message = state["messages"][-1].content

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages_for_planner = [
    SystemMessage(content=PLANNER_PROMPT)
    ]

    messages_for_planner.extend(state["messages"][-6:])  # last 6 messages

    response = llm.invoke(messages_for_planner)

    try:
        plan = json.loads(response.content)

        if "steps" not in plan or not isinstance(plan["steps"], list):
            raise ValueError("Invalid planner format")

    except Exception:
        plan = {"steps": [{"action": "synthesize"}]}

    # 🔥 IMPORTANT: Reset messages to only latest user message
    print("\n================ PLANNER =================")
    print("Generated Plan:")
    print(json.dumps(plan, indent=2))
    print("==========================================\n")
    return {
        "plan": plan,
        "tool_results": [],
        "messages": state["messages"]
    }


def PlannerExecutorNode(state: AgentState) -> dict:
    plan = state.get("plan", {})
    steps = plan.get("steps", [])
    tool_results = []

    result = None

    for step in steps:
        index_name = None

        if "tool" not in step:
            if step.get("action") == "synthesize":
                break
            continue

        tool_name = step["tool"]
        query = step.get("query", "")

        try:
            if tool_name == "search_documents":

                # Boost revenue search
                if any(word in query.lower() for word in ["revenue", "net sales"]):
                    query += " total net sales consolidated statements of operations"

                # Extract year
                year_match = re.search(r"(20\d{2})", query)
                year = year_match.group(1) if year_match else None

                available_indexes = list_indexes()

                for idx in available_indexes:
                    parts = idx.split("_")
                    if len(parts) < 3:
                        continue

                    company_part = parts[0]
                    year_part = parts[-1]

                    if year and year_part == year and company_part in query.lower():
                        index_name = idx
                        break

                result = search_documents.invoke({
                    "query": query,
                    "index_name": index_name
                })

            elif tool_name == "fetch_news":
                result = fetch_news.invoke({"query": query})

            elif tool_name == "calculate_metrics":
                parsed = parse_financial_inputs(query)
                if not parsed:
                    raise ValueError("Invalid financial inputs")
                result = calculate_metrics.invoke(parsed)

            tool_results.append({
                "tool": tool_name,
                "status": "success",
                "data": result
            })

        except Exception as e:
            tool_results.append({
                "tool": tool_name,
                "status": "failed",
                "error": str(e)
            })

        print("\n================ EXECUTOR =================")
        print("Tool:", tool_name)
        print("Query:", query)
        print("Resolved Index:", index_name)

    print("Retrieved Result Length:", len(str(result)))
    print("==========================================\n")
    return {
        "tool_results": tool_results
    }


def route_decision(state):
    return state["intent"]


def CustomToolNode(state : AgentState) -> dict:
    last_message = state["messages"][-1]

    if not hasattr(last_message,"tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_map = {t.name: t for t in TOOLS}
    tool = tool_map[tool_name]

    result = tool.invoke(tool_args)

    return {
        "messages": [
            ToolMessage(
                content=str(result),
                tool_call_id = tool_call["id"]
            )
        ],
        "tool_results": state.get("tool_results",[]) + [result]
    }


def ToolResultRouter(state: AgentState):
    if state.get("tool_results"):
        last_tool = state["tool_results"][-1]

        if isinstance(last_tool, dict) and last_tool.get("status") == "success":
            if "local_path" in last_tool:
                return "confirm_sec"

    return "agent"


def ConfirmSecNode(state: AgentState) -> dict:
    last_tool = state["tool_results"][-1]

    return {
        "messages": [
            AIMessage(
                content=(
                    f"📄 10-K Found\n\n"
                    f"Company: {last_tool['company']}\n"
                    f"Year: {last_tool['year']}\n"
                    f"Filing ID: {last_tool['filing_date']}\n\n"
                    f"Preview:\n{last_tool['preview']}\n\n"
                    f"Would you like to ingest this document into your knowledge base? (yes/no)"
                )
            )
        ],
        "pending_sec_path": last_tool["local_path"],
        "pending_sec_meta": {
            "company": last_tool["company"],
            "year": last_tool["year"]
        },
        "tool_results": state.get("tool_results", [])
    }


def IngestSecNode(state: AgentState) -> dict:
    path = state.get("pending_sec_path")
    meta = state.get("pending_sec_meta")

    if not path or not meta:
        return {
            "messages": [
                AIMessage(content=(
                    "⚠️ There is no pending SEC filing to ingest.\n\n"
                    "To fetch a filing, ask me something like:\n"
                    "\"Fetch Apple 2023 10-K\" or \"Get Microsoft 2022 annual report\""
                ))
            ]
        }

    result = ingest(
        file_paths=[path],
        metadata={
            "company": meta["company"],
            "doc_type": "10-K",
            "year": meta["year"],
        }
    )
    if result["status"] == "success" and result.get("chunks", 0) < 20:
        message = "⚠️ Ingestion suspicious: very low chunk count."

    if result["status"] == "success":
        message = "✅ Document successfully ingested into knowledge base."
    else:
        message = f"⚠️ Ingestion failed: {result.get('reason')}"

    return {
        "messages": [AIMessage(content=message)],
        "pending_sec_path": None,
        "pending_sec_meta": None
    }       
    

def SynthesisNode(state: AgentState) -> dict:
    user_query = state["messages"][-1].content
    tool_results = state.get("tool_results", [])

    # Flatten tool results into readable structured blocks
    structured_data = ""
    for i, result in enumerate(tool_results, 1):
        structured_data += f"\n--- TOOL RESULT {i} ---\n"
        structured_data += str(result.get("data", result))
        structured_data += "\n"

    synthesis_prompt = """
You are a senior financial analyst.

Instructions:
1. Use ONLY the provided tool results.
2. If multiple companies are present, compare them explicitly.
3. Mention each company separately before concluding.
4. Present financial figures clearly.
5. Do not ignore any tool result.
6. Do not hallucinate numbers.

If only one company exists, provide single-company analysis.
If two companies exist, provide structured comparison.
"""

    response = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    ).invoke([
        SystemMessage(content=synthesis_prompt),
        HumanMessage(content=f"User Query:\n{user_query}"),
        HumanMessage(content=f"Tool Results:\n{structured_data}")
    ])

    return {
        "messages": [AIMessage(content=response.content)]
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent",AgentNode)
    graph.add_node("tools",CustomToolNode)
    graph.add_node("router",RouterNode)
    graph.add_node("deterministic", DeterministicExecutionNode)
    graph.add_node("planner",PlannerNode)
    graph.add_node("executor", PlannerExecutorNode)
    graph.add_node("synthesis", SynthesisNode)
    graph.add_node("confirm_sec", ConfirmSecNode)
    graph.add_node("ingest_sec", IngestSecNode)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "CALCULATION": "deterministic",
            "NEWS": "deterministic",
            "DOCUMENT": "agent",
            "GENERAL": "agent",
            "COMPLEX": "planner",
            "FETCH_SEC": "agent",
            "CONFIRM_SEC": "ingest_sec",
            "CANCEL_SEC": END
        }
    )
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    graph.add_conditional_edges(
    "tools",
    ToolResultRouter,   
    {
        "confirm_sec": "confirm_sec",
        "agent": "agent"
    }
    )
    graph.add_edge("deterministic", END)
    graph.add_edge("confirm_sec", END)
    graph.add_edge("ingest_sec", END)

    # Planner execution flow
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "synthesis")
    graph.add_edge("synthesis", END)

    checkpointer = MemorySaver()    

    return graph.compile(checkpointer=checkpointer)

def chat(user_input: str, session_id: str, graph) -> str:
    try:
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        # 🔹 Load persistent memory from SQLite
        stored = load_messages(session_id)

        summary = get_conversation_summary(session_id)

        previous_messages = []

        # Inject summary first if exists
        if summary:
            previous_messages.append(
                SystemMessage(content=f"Conversation summary:\n{summary}")
            )

        for role, content in stored:
            if role == "user":
                previous_messages.append(HumanMessage(content=content))
            else:
                previous_messages.append(AIMessage(content=content))

        # Add new user message to history
        current_messages = previous_messages + [HumanMessage(content=user_input)]

        # Token-aware trimming
        MAX_TOKENS = 7000  # Safe limit under model max

        if count_tokens(current_messages) > MAX_TOKENS:

        # Take oldest half (excluding latest message)
            old_messages = current_messages[:-5]

            if old_messages:
                summary_text = summarize_messages(old_messages)
                update_conversation_summary(session_id, summary_text)

                # Keep only last few messages
                current_messages = current_messages[-5:]

        inputs = {
            "messages": current_messages
        }

        final_state = None

        # 🔹 Run graph execution
        for state_update in graph.stream(
            inputs,
            config=config,
            stream_mode="values",
        ):
            final_state = state_update

        if not final_state:
            return "⚠️ No response generated."

        messages = final_state.get("messages", [])

        # 🔹 Extract last AI message
        final_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_ai_message = msg.content
                break

        if not final_ai_message:
            return "⚠️ No response generated."

        # 🔹 Save both user + AI to database
        save_message(session_id, "user", user_input)

        # Auto-generate title if first message
        existing_messages = load_messages(session_id)
        if len(existing_messages) == 1:
            clean_title = user_input.strip().split("\n")[0]
            title = clean_title[:50]
            update_conversation_title(session_id, title)

        save_message(session_id, "ai", final_ai_message)

        return final_ai_message

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return "⚠️ The AI service is temporarily unavailable."
