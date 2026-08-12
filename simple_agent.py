"""
Phase 1 project: your first single-tool agent — Hugging Face version.

Why this version looks different from a "tool calling API" version:
Not all models (especially the free/open ones you'll run locally later)
have built-in structured tool-calling. So instead of relying on an SDK
feature, this teaches the agent loop the way it actually works underneath:

  1. Tell the model what tools exist, and ask it to reply in a strict format
  2. Read its reply. Is it asking for a tool, or giving a final answer?
  3. If a tool was requested: run the real Python function
  4. Feed the result back into the conversation as a new message
  5. Repeat until the model gives a final answer

This exact loop works unchanged with a hosted Hugging Face model now, and
later with a fully local model (llama.cpp, Ollama, transformers) — only
the "call the model" line changes. That's the point of building it this way.
"""

import os
import re
import json
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Free Hugging Face account -> Settings -> Access Tokens -> create a "read" token
# setx HF_TOKEN "your-token-here"   (PowerShell, then reopen the terminal)
client = InferenceClient(token=os.environ["HF_TOKEN"])
MODEL = "Qwen/Qwen2.5-7B-Instruct"  # small, free-tier friendly, good at following formats

SYSTEM_PROMPT = """You are an assistant with access to ONE tool.

Tool: text_stats
Description: analyzes a block of text and returns word count, sentence count,
and estimated reading time in minutes.
Input: {"text": "<the text to analyze>"}

Rules for how you must reply — follow EXACTLY, no other text:
- If you need the tool, reply with ONLY this, nothing else:
  ACTION: text_stats
  INPUT: {"text": "..."}
- If you already have what you need (or the tool isn't relevant), reply with ONLY:
  FINAL: <your answer to the user>
"""


# ---------------------------------------------------------------------------
# Step A: define the real Python function the tool will run
# ---------------------------------------------------------------------------

def text_stats(text: str) -> dict:
    """The actual work. The model never runs this itself — it only asks us to."""
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    word_count = len(words)
    sentence_count = len(sentences)
    reading_time_min = round(word_count / 200, 1)  # ~200 wpm average
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "estimated_reading_time_minutes": reading_time_min,
    }


TOOLS = {"text_stats": text_stats}


# ---------------------------------------------------------------------------
# Step B: parse the model's strict-format reply
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

    # model didn't follow the format — treat whatever it said as the final answer
    return {"type": "final", "content": reply}


# ---------------------------------------------------------------------------
# Step C: the agent loop itself
# ---------------------------------------------------------------------------

def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        # --- 1. send conversation to the model ---
        response = client.chat_completion(model=MODEL, messages=messages, max_tokens=512)
        reply = response.choices[0].message.content

        # --- 2. did the model answer, or ask for a tool? ---
        parsed = parse_reply(reply)

        if parsed["type"] == "final":
            return parsed["content"]

        # --- 3. run the real function ---
        tool_fn = TOOLS.get(parsed["tool"])
        result = tool_fn(**parsed["input"]) if tool_fn else {"error": "unknown tool"}

        # --- 4. feed the result back into the conversation ---
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"TOOL RESULT: {json.dumps(result)}"})

        # --- 5. loop back to step 1 ---


# ---------------------------------------------------------------------------
# Try it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Al-Asim Welfare Foundation works to improve access to quality education "
        "in rural communities. Our programs focus on teacher training, learning "
        "materials, and basic digital literacy. We believe every child deserves "
        "a fair start, regardless of where they were born."
    )

    question = f"How long would it take an average reader to read this, and how many sentences is it?\n\n{sample}"

    print("User:", question, "\n")
    answer = run_agent(question)
    print("Agent:", answer)
