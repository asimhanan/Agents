import json
import os
import re
import urllib.request
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

client = InferenceClient(token=os.environ["HF_TOKEN"])
MODEL = "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = """You are an assistant with access to ONE tool.

Tool: read_google_doc
Description: Fetches the full text content of a publicly accessible Google Doc link.
Input: {"url": "<google_doc_url>"}

Rules for how you must reply — follow EXACTLY, no other text:
- If you need to read a Google Doc link provided by the user, reply with ONLY this, nothing else:
  ACTION: read_google_doc
  INPUT: {"url": "..."}
- If you already have the Google Doc contents (from a TOOL RESULT) or the tool isn't relevant, reply with ONLY:
  FINAL: <your answer to the user based on the document's instructions>
"""


# ---------------------------------------------------------------------------
# Step A: Define the Google Doc Fetcher Tool
# ---------------------------------------------------------------------------

def read_google_doc(url: str) -> dict:
    """Extracts the document ID from a Google Doc URL and exports its content as plain text."""
    try:
        # Works for /edit, /view, /export style links — grabs whatever sits between "/d/" and the next "/"
        match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
        if not match:
            return {"error": "Invalid Google Doc URL format."}

        doc_id = match.group(1)
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"

        req = urllib.request.Request(export_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            content = response.read().decode("utf-8")

        return {"document_text": content}
    except Exception as e:
        return {"error": f"Failed to fetch document: {str(e)}"}


TOOLS = {"read_google_doc": read_google_doc}


# ---------------------------------------------------------------------------
# Step B: Parse the model's strict-format reply
# ---------------------------------------------------------------------------

def parse_reply(reply: str):
    reply = reply.strip()
    if reply.startswith("FINAL:"):
        return {"type": "final", "content": reply[len("FINAL:"):].strip()}

    if reply.startswith("ACTION:"):
        action_match = re.search(r"ACTION:\s*(\w+)", reply)
        input_match = re.search(r"INPUT:\s*(\{.*\})", reply, re.DOTALL)
        if action_match and input_match:
            tool_name = action_match.group(1)
            tool_input = json.loads(input_match.group(1))
            return {"type": "action", "tool": tool_name, "input": tool_input}

    # Model didn't follow the format — treat whatever it said as the final answer
    return {"type": "final", "content": reply}


# ---------------------------------------------------------------------------
# Step C: The agent loop
# ---------------------------------------------------------------------------

def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat_completion(model=MODEL, messages=messages, max_tokens=512)
        reply = response.choices[0].message.content

        parsed = parse_reply(reply)

        if parsed["type"] == "final":
            return parsed["content"]

        tool_fn = TOOLS.get(parsed["tool"])
        result = tool_fn(**parsed["input"]) if tool_fn else {"error": "unknown tool"}

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"TOOL RESULT: {json.dumps(result)}"})


# ---------------------------------------------------------------------------
# Execution Example (only runs when you do `python google_doc_agent.py`,
# NOT when app.py imports run_agent — that's what keeps this safe to import)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure your Google Doc sharing settings are set to "Anyone with the link can view"
    doc_url = "https://docs.google.com/document/d/1OtdfbUpZ9NXLcdKKq4-uWsdcqziJyzM5Ejnvx0j27L4/edit?usp=sharing"
    question = f"Read the instructions in this Google Doc and answer my question: What is important points of my app.? Link: {doc_url}"
    print("User:", question, "\n")
    answer = run_agent(question)
    print("Agent:", answer)
