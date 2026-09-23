"""One function, one config change: this is the abstraction TECHNICAL_PLAN.md
Section 6 describes so the final-demo swap from free/local (Ollama) to paid
cloud (OpenAI/Claude) is a one-line change, not a rewrite.

Development: backend="ollama" (free, local, already installed on this machine
with llama3:latest / mistral:latest pulled).
Final demo only: backend="openai" or backend="claude" (paid, needs an API key
in the environment -- OPENAI_API_KEY / ANTHROPIC_API_KEY).
"""

import os

# Read fresh from the environment on every call (NOT bound as a stale
# function-default at import time) -- callers like agents/chat_agent.py and
# app.py set os.environ["BTP_LLM_BACKEND"] right before invoking the graph,
# expecting that to take effect. A `backend: str = os.environ.get(...)`
# default parameter would bake in whatever the env var was when this module
# was FIRST imported and silently ignore every later change, which would
# quietly break the paid-API demo swap this whole abstraction exists for.


def call_llm(prompt: str, system: str | None = None, backend: str | None = None, **kwargs) -> str:
    backend = backend or os.environ.get("BTP_LLM_BACKEND", "ollama")
    if backend == "ollama":
        model = kwargs.get("model") or os.environ.get("BTP_OLLAMA_MODEL", "llama3:latest")
        return _call_ollama(prompt, system=system, model=model)
    if backend == "openai":
        return _call_openai(prompt, system=system, model=kwargs.get("model", "gpt-4o-mini"))
    if backend == "claude":
        return _call_claude(prompt, system=system, model=kwargs.get("model", "claude-sonnet-5"))
    raise ValueError(f"Unknown LLM backend '{backend}'. Use 'ollama' (free/local), 'openai', or 'claude'.")


def _call_ollama(prompt: str, system: str | None, model: str) -> str:
    import ollama
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def _call_openai(prompt: str, system: str | None, model: str) -> str:
    # Paid -- final-demo only (TECHNICAL_PLAN.md Week 11 / PROFESSOR_PROPOSAL.md Sec.4).
    # Requires: pip install openai ; export OPENAI_API_KEY=...
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def _call_claude(prompt: str, system: str | None, model: str) -> str:
    # Paid -- final-demo only. Requires: pip install anthropic ; export ANTHROPIC_API_KEY=...
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model, max_tokens=1024,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    import time

    start = time.time()
    reply = call_llm(
        "In one sentence, what does a reconstruction-error anomaly detector do?",
        system="You are a concise technical assistant.",
    )
    elapsed = time.time() - start
    used_backend = os.environ.get("BTP_LLM_BACKEND", "ollama")
    used_model = os.environ.get("BTP_OLLAMA_MODEL", "llama3:latest")
    print(f"[{used_backend} / {used_model}] ({elapsed:.1f}s)\n{reply}")
