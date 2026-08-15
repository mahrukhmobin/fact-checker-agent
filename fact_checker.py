from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from groq import Groq
from ddgs import DDGS
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY= os.getenv("GROQ_API_KEY")
client= Groq(api_key=GROQ_API_KEY)

class FactCheckState(TypedDict):
    claim: str
    search_results: str
    supporting_evidence: str
    contradicting_evidence: str
    has_conflict: bool
    verdict: str
    confidence: str
    final_report: str

def search_evidence(state: FactCheckState) -> dict:
    claim= state["claim"]
    results_text= ""
    with DDGS() as ddgs:
        supporting_search= list(ddgs.text(f"{claim} evidence proof true", max_results=4))
        for r in supporting_search:
            results_text += f"- {r['title']}: {r['body']}\n"
        against_search= list(ddgs.text(f"{claim} debunked false myth", max_results=4))
        for r in against_search:
            results_text += f"- {r['title']}: {r['body']}\n"
    return {"search_results": results_text}

def analyze_evidence(state: FactCheckState) -> dict:
    claim= state["claim"]
    search_results= state["search_results"]
    prompt = f"""You are a strict fact-checking assistant. Do not add any greeting, preamble, or explanation outside the requested format.

Claim: "{claim}"

Search results related to this claim:
{search_results}

Task: Analyze the search results and separate them into two categories.

Respond in EXACTLY this format, with no extra text before or after:
SUPPORTING:
- point 1
- point 2

CONTRADICTING:
- point 1
- point 2

Rules:
- If the search results provide no relevant information for a category, write exactly "None found" under that heading.
- If the search results are empty or irrelevant to the claim, write "None found" under both headings.
- Keep each point short (one sentence).
- Do not include any text other than the two headings and their points.
"""
    response= client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages= [{"role": "user", "content": prompt}]
    )
    answer= response.choices[0].message.content
    if "CONTRADICTING:" in answer:
        parts= answer.split("CONTRADICTING:") 
        supporting= parts[0].replace("SUPPORTING:", "").strip()
        contradicting= parts[1].strip()
    else:
        supporting= answer
        contradicting= "None Found"
    return{
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting
    }


def check_conflict(state: FactCheckState) -> dict:
    contradicting= state["contradicting_evidence"]
    if "none found" in contradicting.lower():
        return {"has_conflict": False}
    else:
        return {"has_conflict": True}

def deep_verification(state: FactCheckState) -> dict:
    claim= state["claim"]
    extra_text= ""
    with DDGS() as ddgs:
        expert_search= list(ddgs.text(f"{claim} scientific study research fact check", max_results=5))
        for r in expert_search:
            extra_text += f"- {r['title']}: {r['body']}\n"
    prompt = f"""You are a strict fact-checking assistant. The initial research found conflicting evidence for this claim, so deeper verification was done.

Claim: "{claim}"

Original supporting evidence:
{state["supporting_evidence"]}

Original contradicting evidence:
{state["contradicting_evidence"]}

Additional expert/scientific search results:
{extra_text}

Task: Based on ALL the evidence above (original + additional), give an updated, more accurate summary.

Respond in EXACTLY this format, with no extra text before or after:
SUPPORTING:
- point 1

CONTRADICTING:
- point 1

Rules:
- Prioritize scientific/expert sources over general web results.
- Keep each point short (one sentence).
- If a category truly has no evidence, write "None found".
"""
    response= client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages= [{"role": "user", "content": prompt}]
    )
    answer= response.choices[0].message.content
    if "CONTRADICTING:" in answer:
        parts= answer.split("CONTRADICTING:") 
        supporting= parts[0].replace("SUPPORTING:", "").strip()
        contradicting= parts[1].strip()
    else:
        supporting= answer
        contradicting= "None Found"
    return{
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting
    }

def route_after_check(state: FactCheckState) -> str:
    if state["has_conflict"]:
        return "deep_verification"
    else:
        return "generate_verdict"

def generate_verdict(state: FactCheckState) -> dict:
    claim= state["claim"]
    supporting= state["supporting_evidence"]
    contradicting= state["contradicting_evidence"]
    prompt = f"""You are a strict fact-checking assistant. Based on the evidence below, give a final verdict.

Claim: "{claim}"

Supporting evidence:
{supporting}

Contradicting evidence:
{contradicting}

Task: Give a final verdict on this claim.

Respond in EXACTLY this format, with no extra text before or after:
VERDICT: [one word: True, False, Misleading, or Unverifiable]
CONFIDENCE: [a percentage, like 75%]
REASON: [one or two sentences explaining why]
"""
    response= client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages= [{"role": "user", "content": prompt}]
    )
    answer= response.choices[0].message.content
    verdict_line= ""
    confidence_line= ""
    reason_line= ""
    for line in answer.split("\n"):
        if line.startswith("VERDICT:"):
            verdict_line= line.replace("VERDICT:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            confidence_line= line.replace("CONFIDENCE:", "").strip()
        elif line.startswith("REASON:"):
            reason_line= line.replace("REASON:", "").strip()
    return{
        "verdict": f"{verdict_line} ({reason_line})",
        "confidence": confidence_line
    }

def generate_report(state: FactCheckState) -> dict:
    report = f"""
========================================
FACT-CHECK REPORT
========================================
Claim: {state["claim"]}

Verdict: {state["verdict"]}
Confidence: {state["confidence"]}

Supporting Evidence:
{state["supporting_evidence"]}

Contradicting Evidence:
{state["contradicting_evidence"]}
========================================
"""
    return {"final_report": report}

graph= StateGraph(FactCheckState)
graph.add_node("search_evidence", search_evidence)
graph.add_node("analyze_evidence", analyze_evidence)
graph.add_node("check_conflict", check_conflict)
graph.add_node("deep_verification", deep_verification)
graph.add_node("generate_verdict", generate_verdict)
graph.add_node("generate_report", generate_report)
graph.set_entry_point("search_evidence")
graph.add_edge("search_evidence", "analyze_evidence")
graph.add_edge("analyze_evidence", "check_conflict")
graph.add_conditional_edges(
    "check_conflict",
    route_after_check,
    {
        "deep_verification": "deep_verification",
        "generate_verdict": "generate_verdict"
    }
)
graph.add_edge("deep_verification", "generate_verdict")
graph.add_edge("generate_verdict", "generate_report")
graph.add_edge("generate_report", END)
app= graph.compile()

def run_fact_check(claim):
    initial_state= {
        "claim": claim,
        "search_results": "",
        "supporting_evidence": "",
        "contradicting_evidence": "",
        "has_conflict": False,
        "verdict": "",
        "confidence": "",
        "final_report": ""
    }
    result= app.invoke(initial_state)
    return result["final_report"]

if __name__ == "__main__":
    user_claim= input("Enter a claim to fact-check: ")
    print("\nFact-checking in progress...\n")
    report= run_fact_check(user_claim)
    print(report)
    filename= f"{user_claim[:30].replace(' ', '_')}_factcheck.txt"
    with open(filename, "w", encoding= "utf-8") as file:
        file.write(report)
    print(f"\nReport saved to {filename}")