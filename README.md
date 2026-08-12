#To run it: 

pip install -r requirements.txt
#
 get a free token from huggingface.co, set HF_TOKEN, then python simple_agent.py.

# Phase 1: Foundations (Weeks 1–3)

Goal: understand how LLM agents actually work under the hood, then build your first single-tool agent in Python.

---

## 1. LLM Basics (what you need to actually know, not theory overload)

- An LLM is a text-in, text-out function. You send it a conversation (a list of messages), it returns the next message.
- It has **no memory** between calls — every call must include the full conversation history you want it to "remember."
- It **can't do anything by itself** — no internet, no file access, no calculator. That's what "tools" are for.
- A "tool" is just a function you describe to the model (name, description, input schema). The model can't run it — it can only *ask* you to run it, by replying with a tool-use request instead of plain text.

## 2. The Agent Loop

This is the one concept the whole roadmap builds on. Every agent, no matter how complex, is this loop:

```
1. Send conversation + available tools to the model
2. Model replies with either:
     a) a normal text answer  -> done, show it to the user
     b) a tool-use request    -> go to step 3
3. Run the actual Python function the model asked for
4. Add the tool's result to the conversation
5. Go back to step 1
```

Phase 2 (multiple tools), Phase 3 (memory/RAG), and Phase 4 (multi-agent) are all this same loop with more pieces added around it. Understanding this loop cold is the entire point of Phase 1.

## 3. What `simple_agent.py` does

A single-tool agent that gives the model one capability: `text_stats` — analyzes a piece of text and returns word count, sentence count, and estimated reading time.

Why this tool and not a generic calculator: it's a small first building block toward the Phase 2 project (an education content assistant for AWF) — you'll extend this exact loop with more content-focused tools.

**Note on this version:** it uses a free Hugging Face model (`Qwen/Qwen2.5-7B-Instruct`) instead of a paid API. Free/open models often don't have built-in "tool calling" the way commercial APIs do, so instead the code teaches the agent loop the way it actually works underneath — the model is instructed to reply in a strict `ACTION:` / `FINAL:` format, and our Python code parses that reply itself. This is more honest about what's really happening, and it's the exact same technique you'll use when you move to a fully local model later — only the "call the model" line changes, the loop stays identical.

Walk through the code with the loop above open side by side — you'll see steps 1–5 as actual lines of Python.

## 4. Setup (VS Code, Windows)

1. Install the library:
   ```
   pip install huggingface_hub
   ```
2. Create a free account at huggingface.co, then go to Settings -> Access Tokens -> create a new "read" token.
3. Set it as an environment variable (PowerShell):
   ```
   setx HF_TOKEN "your_key"
   ```
   (close and reopen VS Code's terminal after this so it picks up the new variable)
4. Open this folder in VS Code, run:
   ```
   python simple_agent.py
   ```

Free-tier note: Hugging Face's hosted Inference API is rate-limited (fine for learning, not for production). If a model is "cold" it may take a few seconds to spin up on the first call — that's normal, not a bug.

## 5. Looking ahead: local models

You mentioned wanting to eventually download and run a model locally instead of depending on any hosted API. That fits perfectly with how this loop is built — because we're not using a provider-specific "tool calling" feature, the exact same `parse_reply` / loop logic will work with a model running through Ollama or llama.cpp on your own machine. When you're ready for that, the only change is swapping the `client.chat_completion(...)` line for a local call — everything else in this file stays the same. We'll do that switch together once you're comfortable with this version.

## 6. Try this yourself (don't skip this)

Once it runs, make these changes yourself before moving to Phase 2:
1. Change the tool to count paragraphs instead of sentences.
2. Add a `print()` inside the loop that shows *why* the loop is looping (tool call vs final answer) — makes step 2 above concrete.
3. Ask it a question that doesn't need the tool at all, and watch it skip straight to a text answer — proves the model decides, not your code.
