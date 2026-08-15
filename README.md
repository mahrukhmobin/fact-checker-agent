# Fact-Checker Agent

A multi-step AI agent that fact-checks claims using **LangGraph**. Given any claim, it searches the web for supporting and contradicting evidence, and — if the evidence conflicts — automatically triggers a deeper verification step using more authoritative sources before delivering a final verdict.

## What It Does

1. **Searches the web** for evidence both supporting and contradicting the claim
2. **Analyzes the evidence** using an LLM, separating it into two categories
3. **Checks for conflict** — if evidence disagrees, it branches into deeper verification
4. **Deep verification** (conditional) — searches expert/scientific sources for a more accurate summary
5. **Generates a final verdict** — True / False / Misleading / Unverifiable — with a confidence score and reasoning
6. **Saves a structured report** as a `.txt` file

Built using **LangGraph's** `StateGraph`, custom **Nodes**, and **Conditional Edges** — so the agent adapts its path based on what it finds, instead of following one fixed sequence.

## Tech Stack

- **LangGraph** — StateGraph, Nodes, Conditional Edges
- **Groq (Llama 3.3 70B)** — LLM for evidence analysis and verdict generation
- **DDGS (DuckDuckGo Search)** — real-time web search

## Setup

1. Clone the repo
```bash
git clone https://github.com/mahrukhmobin/fact-checker-agent.git
cd fact-checker-agent
```

2. Install dependencies
```bash
pip install langgraph groq ddgs python-dotenv
```

3. Create a `.env` file in the project root and add your Groq API key
```
GROQ_API_KEY=your_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com)

4. Run it
```bash
python fact_checker.py
```

## Author

Built by [Mahrukh Mobin](https://www.linkedin.com/in/mahrukh-mobin-) as part of a hands-on journey into building AI agents.
