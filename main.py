from fastapi import FastAPI, Request
import os
import json
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client
from sentence_transformers import SentenceTransformer
import httpx

load_dotenv()

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPS_TEST_WEBHOOK_URL = os.getenv("OPS_TEST_WEBHOOK_URL")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

supabase_a = create_client(
    os.getenv("SUPABASE_A_URL"),
    os.getenv("SUPABASE_A_KEY")
)

supabase_b = create_client(
    os.getenv("SUPABASE_B_URL"),
    os.getenv("SUPABASE_B_KEY")
)

rag_model = SentenceTransformer('all-MiniLM-L6-v2')


async def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": chat_id, "text": text})

# Tool definitions — this tells Groq what tools exist and when to use them
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_leads_count",
            "description": "Get the count and status of leads sourced by the lead-gen pipeline",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_triage_status",
            "description": "Get the count of open or escalated support tickets from TriageGate",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_rag_docs",
            "description": "Search the RAG document knowledge base to answer a question from stored documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query or question"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_workflow",
            "description": "Re-run or trigger a specific n8n automation workflow by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_name": {"type": "string", "description": "Name of the workflow to trigger"}
                },
                "required": ["workflow_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "Get a health check summary across all connected systems (Supabase tables, last run times)",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    message = data.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    print(f"Received message from chat_id {chat_id}: {text}")

    # Ask Groq which tool to use, if any
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an AI ops assistant. Decide which tool to call based on the user's request."},
            {"role": "user", "content": text}
        ],
        tools=tools,
        tool_choice="auto"
    )

    choice = response.choices[0].message

    if choice.tool_calls:
        for tool_call in choice.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(
                tool_call.function.arguments) if tool_call.function.arguments else {}
            print(f"Groq decided to call: {tool_name} with args: {tool_args}")

            if tool_name == "query_leads_count":
                result = supabase_a.table("processed_leads").select(
                    "qualified", count="exact").execute()
                total = result.count
                qualified = sum(1 for row in result.data if row["qualified"])
                reply_text = f"You have {total} total leads, {qualified} of them qualified."

            elif tool_name == "query_triage_status":
                result = supabase_b.table("decisions").select(
                    "decision_type", count="exact").execute()
                total = result.count
                escalated = sum(
                    1 for row in result.data if row["decision_type"] == "escalated")
                auto_resolved = total - escalated
                reply_text = f"You have {total} total tickets: {escalated} escalated, {auto_resolved} auto-resolved."

            elif tool_name == "search_rag_docs":
                query_text = tool_args.get("query", text)
                question_embedding = rag_model.encode([query_text])[0].tolist()

                results = supabase_a.rpc("match_documents", {
                    "query_embedding": question_embedding,
                    "match_count": 3
                }).execute()

                context = "\n\n".join([match["content"]
                                      for match in results.data])

                rag_response = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{
                        "role": "user",
                        "content": f"Answer the question using only the context below. If the answer isn't in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {query_text}"
                    }]
                )
                reply_text = rag_response.choices[0].message.content

            elif tool_name == "trigger_workflow":
                workflow_name = tool_args.get("workflow_name", "unknown")
                async with httpx.AsyncClient() as client:
                    workflow_response = await client.post(OPS_TEST_WEBHOOK_URL)
                if workflow_response.status_code == 200:
                    reply_text = f"Workflow '{workflow_name}' triggered successfully."
                else:
                    reply_text = f"Tried to trigger '{workflow_name}' but got an error (status {workflow_response.status_code})."

            else:
                reply_text = f"Tool '{tool_name}' isn't wired up yet."

            await send_telegram_message(chat_id, reply_text)
    else:
        await send_telegram_message(chat_id, choice.content)

    return {"ok": True}
