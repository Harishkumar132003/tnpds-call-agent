"""Per-step test evals, pushed to Langfuse as datasets + scored runs.

One dataset per pipeline step, each item an expected behaviour; a run executes
the live step (live prompts from the portal, live models) and scores it:

- step-topic-check   : NEW / FOLLOW_UP decisions          score `correct` (exact)
- step-llm1-decide   : search tool vs direct reply        score `correct` + `silent_on_tool_call`
- step-query-rewrite : one English query, intent kept
                       score `well_formed` (rules) + `intent_kept` (judge)
- step-llm2-answer   : grounded answer from given sheets  score `grounded` (judge)

Usage:  .venv/bin/python scripts/step_evals.py [--run NAME]
Needs the portal on :8090 (for the composed prompts) and backend/.env loaded.
Results appear in Langfuse under Datasets -> <dataset> -> Runs -> NAME.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langfuse import Langfuse  # noqa: E402

from voice_agent.config import load_settings  # noqa: E402
from voice_agent.topics import TopicTracker  # noqa: E402

PORTAL = "http://127.0.0.1:8090"
JUDGE_MODEL = "gpt-5.5"
SARVAM_CHAT = "https://api.sarvam.ai/v2/chat/completions"

settings = load_settings()
SARVAM_KEY = settings.sarvam_api_key.get_secret_value()
OPENAI_KEY = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None

cfg = httpx.get(f"{PORTAL}/api/config", timeout=30).json()
pv = httpx.post(f"{PORTAL}/api/config/preview", json={"language": "ta-IN"}, timeout=30).json()
ROUTER_PROMPT, ANSWER_PROMPT = pv["prompt"], pv["answer_prompt"] or pv["prompt"]

lf = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key.get_secret_value(),
    host=settings.langfuse_host,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the FAQ and procedure documents.",
            "parameters": {
                "type": "object",
                "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
                "required": ["questions"],
            },
        },
    }
]


def sarvam(messages, tools=None):
    body = {
        "model": settings.llm_model,
        "reasoning_effort": "none",
        "temperature": cfg.get("temperature") or 0.3,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
    for _ in range(3):
        try:
            r = httpx.post(
                SARVAM_CHAT,
                timeout=90,
                headers={"Authorization": f"Bearer {SARVAM_KEY}"},
                json=body,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]
        except Exception:
            time.sleep(2)
    raise RuntimeError("sarvam chat failed")


def judge(instruction: str, payload: str) -> tuple[bool, str]:
    """gpt-5.5 as a pass/fail judge; returns (passed, reason)."""
    for _ in range(3):
        try:
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                timeout=120,
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={
                    "model": JUDGE_MODEL,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": instruction
                            + '\nReturn JSON: {"pass": true|false, "reason": "<one line>"}',
                        },
                        {"role": "user", "content": payload},
                    ],
                },
            )
            v = json.loads(r.json()["choices"][0]["message"]["content"])
            return bool(v.get("pass")), str(v.get("reason") or "")
        except Exception as exc:  # noqa: PERF203
            err = exc
            time.sleep(3)
    return False, f"judge failed: {err}"


# --------------------------------------------------------------------------- cases

LOST_CARD_CTX = [
    "Caller: என்னோட ஸ்மார்ட் கார்டு தொலைஞ்சு போச்சு",
    "Agent: கவலை படாதீங்க, tnpds.gov.in-ல Reprint Smart Card-ல அப்ளை பண்ணலாம்.",
]
ADD_CHILD_CTX = [
    "Caller: என் பையன் பேரை கார்டுல சேர்க்கணும்",
    "Agent: சரி, tnpds.gov.in-ல Add Member-ல அப்ளை பண்ணலாம்.",
]

TOPIC_CASES = [
    ("fee follow-up", LOST_CARD_CTX, "அதுக்கு எவ்வளவு காசு கட்டணும்?", "FOLLOW_UP"),
    ("topic switch", LOST_CARD_CTX, "என் பொண்ணு பேரை கார்டுல சேர்க்கணும்", "NEW"),
    ("documents follow-up", ADD_CHILD_CTX, "அதுக்கு என்ன டாக்குமெண்ட்ஸ் வேணும்?", "FOLLOW_UP"),
    ("thanks", LOST_CARD_CTX, "சரி நன்றி", "FOLLOW_UP"),
    ("new subject", ADD_CHILD_CTX, "ரேஷன் கடை எத்தனை மணிக்கு திறக்கும்?", "NEW"),
    ("online follow-up", LOST_CARD_CTX, "ஆன்லைன்ல பண்ண முடியுமா?", "FOLLOW_UP"),
    ("repeat request", LOST_CARD_CTX, "புரியல, திரும்ப சொல்லுங்க", "FOLLOW_UP"),
    ("different request", ADD_CHILD_CTX, "என் மொபைல் நம்பர மாத்தணும்", "NEW"),
]

LLM1_CASES = [
    ("lost card", "என்னோட ஸ்மார்ட் கார்டு தொலைஞ்சு போச்சு, என்ன பண்ணணும்?", True),
    ("thanks", "சரி, நன்றி.", False),
    ("greeting", "ஹலோ, வணக்கம்.", False),
    ("rice quantity", "எங்க வீட்டுல மூணு பேர், எவ்வளவு அரிசி கிடைக்கும்?", True),
    ("address change", "வீடு மாறிட்டோம், கார்டுல அட்ரஸ் மாத்தணும்.", True),
    ("misheard word", "ஓட்டோ மட்டும் இல்லையா?", False),
    ("ok", "சரி.", False),
    ("new card documents", "புது ஸ்மார்ட் கார்டுக்கு என்ன டாக்குமெண்ட் வேணும்?", True),
]

REWRITE_CASES = [
    (
        "lost card",
        ["என்னோட ஸ்மார்ட் கார்டு தொலைஞ்சு போச்சு, என்ன பண்ணணும்?"],
        "ask how to get a reprint for a lost smart card",
    ),
    (
        "fee follow-up",
        ["என்னோட ஸ்மார்ட் கார்டு தொலைஞ்சு போச்சு", "அதுக்கு எவ்வளவு காசு கட்டணும்?"],
        "ask the fee for the lost-card reprint",
    ),
    (
        "add son (verb!)",
        ["என் பையன் பேரை கார்டுல சேர்க்கணும்"],
        "ask how to ADD the son's name (adding, never removing)",
    ),
    (
        "docs follow-up",
        ["என் பையன் பேரை கார்டுல சேர்க்கணும்", "அதுக்கு என்ன டாக்குமெண்ட்ஸ் வேணும்?"],
        "ask the documents needed to add the son's name",
    ),
    (
        "address change",
        ["வீடு மாறிட்டோம்", "கார்டுல அட்ரஸ் மாத்தணும்"],
        "ask how to change the address on the card",
    ),
    (
        "remove deceased",
        ["எங்க அப்பா இறந்துட்டாரு", "அவர் பேரை கார்டுல இருந்து நீக்கணும்"],
        "ask how to REMOVE the deceased father's name",
    ),
]

FEE_SHEET = (
    "[013 - RP-Lost-MisPlaced]\nReprint Smart Card: visit www.tnpds.gov.in, login with the "
    "registered mobile number and OTP, choose Reprint Smart Card, reason lost/misplaced. "
    "Pay Rs. 50 plus service charges (SBI card, net banking or UPI). An 18-digit reference "
    "number is sent by SMS; another SMS follows when the card is printed. Processed within "
    "15 to 20 working days from the date of application."
)
QTY_SHEET = (
    "[102 - Commodity Quantity]\nRice entitlement depends on the card type: AAY cards 35 kg "
    "per month. PHH/NPHH rice cards: 3 adults 20 kg; 2 adults with children 16 kg; each "
    "additional adult 4 kg. Sugar cards get sugar instead of rice."
)
SCC_SHEET = (
    "[052 - SCC - Name Correction]\nFor name correction: Chennai residents visit the Civil "
    "Supplies Office (Assistant Commissioner). Other areas: contact the Taluk Supply Officer. "
    "Take the Smart Card, family head's Aadhaar and the member's Aadhaar."
)
ADD_SHEET = (
    "[026 - Add member - Child]\nGo to www.tnpds.gov.in, Smart card related services, Add "
    "Member. Login with registered mobile number and OTP. Enter the child's Aadhaar number "
    "and upload the child's Aadhaar card (PDF). A reference number is sent by SMS after "
    "submission."
)

LLM2_CASES = [
    (
        "fee with qualifier",
        "ஸ்மார்ட் கார்டு ரீபிரிண்ட்டுக்கு எவ்வளவு காசு கட்டணும்?",
        {"found": True, "passages": [{"source": "013", "score": 0.66, "text": FEE_SHEET}]},
        "The answer must state the fee as Rs. 50 PLUS service charges (not just Rs. 50) "
        "and say nothing the sheet does not say.",
    ),
    (
        "quantity needs card type",
        "எங்க வீட்டுல மூணு பேர் இருக்கோம், எவ்வளவு அரிசி கிடைக்கும்?",
        {"found": True, "passages": [{"source": "102", "score": 0.62, "text": QTY_SHEET}]},
        "The sheet's quantity depends on the card type. The answer must either ask the "
        "caller's card type or state the per-card-type conditions; a single "
        "unconditional number like '20 kg' fails.",
    ),
    (
        "nothing found -> no info",
        "பிஎம் கிசான் பணம் எப்போ வரும்?",
        {"found": False, "reason": "no relevant information"},
        "The answer must say it does not have that information (and may offer a human "
        "agent). Any factual claim about PM Kisan fails.",
    ),
    (
        "area qualifier kept",
        "என் பேரு கார்டுல தப்பா இருக்கு, எங்க போய் திருத்தணும்?",
        {"found": True, "passages": [{"source": "052", "score": 0.6, "text": SCC_SHEET}]},
        "The answer must keep the area split: Civil Supplies Office for Chennai "
        "residents, Taluk Supply Officer for other areas. Offering the Civil Supplies "
        "Office to everyone fails.",
    ),
    (
        "no invented documents",
        "பையன் பேரை சேர்க்க என்ன டாக்குமெண்ட் வேணும்?",
        {"found": True, "passages": [{"source": "026", "score": 0.61, "text": ADD_SHEET}]},
        "The sheet requires only the child's Aadhaar (uploaded as PDF). Naming documents "
        "the sheet does not mention (ration card, parent's Aadhaar, photo, birth "
        "certificate) fails.",
    ),
]


# --------------------------------------------------------------------------- runners


def ensure_dataset(name: str, description: str, items: list[tuple[str, dict, dict]]) -> None:
    lf.create_dataset(name=name, description=description)  # idempotent in v2 (upsert by name)
    for item_id, item_input, expected in items:
        lf.create_dataset_item(
            dataset_name=name, id=item_id, input=item_input, expected_output=expected
        )


def record(dataset: str, run: str, item_id: str, trace_name: str, inp, out, scores: dict) -> None:
    """One executed item: a trace, linked into the dataset run, with its scores."""
    trace = lf.trace(name=trace_name, input=inp, output=out, tags=["step-eval", dataset])
    for item in _items[dataset]:
        if item.id == item_id:
            item.link(trace, run_name=run)
            break
    for score_name, (value, comment) in scores.items():
        lf.score(trace_id=trace.id, name=score_name, value=value, comment=comment[:500])


async def run_topic(run: str):
    tracker = TopicTracker(
        SARVAM_KEY, model=settings.llm_model, prompt=cfg.get("topic_prompt") or None
    )
    results = []
    for name, ctx, latest, expected in TOPIC_CASES:
        decision, reason = await tracker._classify(list(ctx), latest)
        ok = decision == expected
        record(
            "step-topic-check",
            run,
            f"topic-{name}",
            f"topic check · {name}",
            {"context": ctx, "latest": latest},
            {"decision": decision, "reason": reason},
            {"correct": (1.0 if ok else 0.0, f"expected {expected}, got {decision}: {reason}")},
        )
        results.append((name, ok, f"{decision} ({reason[:60]})"))
    return results


async def run_llm1(run: str):
    results = []
    for name, text, expect_tool in LLM1_CASES:
        msg = await asyncio.to_thread(
            sarvam,
            [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "assistant", "content": cfg.get("greeting") or "வணக்கம்!"},
                {"role": "user", "content": text},
            ],
            TOOLS,
        )
        called = bool(msg.get("tool_calls"))
        spoke = bool((msg.get("content") or "").strip())
        ok = called == expect_tool
        scores = {
            "correct": (
                1.0 if ok else 0.0,
                f"expected {'tool call' if expect_tool else 'direct reply'}, "
                f"got {'tool call' if called else 'direct reply'}",
            )
        }
        if called:
            scores["silent_on_tool_call"] = (
                0.0 if spoke else 1.0,
                "wrote text alongside the tool call" if spoke else "no text, as instructed",
            )
        record(
            "step-llm1-decide",
            run,
            f"llm1-{name}",
            f"llm1 decide · {name}",
            {"caller": text},
            {
                "tool_called": called,
                "text": msg.get("content") or "",
                "questions": [
                    json.loads(c["function"]["arguments"]).get("questions")
                    for c in msg.get("tool_calls") or []
                ],
            },
            scores,
        )
        results.append((name, ok, "tool" if called else "direct"))
    return results


async def run_rewrite(run: str):
    from voice_agent.knowledge.rewrite import QueryRewriter

    rw = QueryRewriter(SARVAM_KEY, model=settings.llm_model, timeout=10)
    results = []
    for name, history, intent in REWRITE_CASES:
        out = await rw.rewrite(history, prompt=cfg.get("rewrite_prompt") or None)
        query = out.query if out else ""
        well_formed = bool(query) and "\n" not in query and query.isascii()
        passed, reason = (False, "rewrite returned nothing")
        if query:
            passed, reason = await asyncio.to_thread(
                judge,
                "You check one rewritten FAQ search query for a Tamil ration-card helpline. "
                f"The caller's messages (oldest first): {history}. The query must be one "
                f"standalone English question that would {intent}, filling the subject from "
                "the earlier messages and never changing the caller's action.",
                f"Rewritten query: {query}",
            )
        record(
            "step-query-rewrite",
            run,
            f"rewrite-{name}",
            f"query rewrite · {name}",
            {"caller_messages": history},
            {"query": query},
            {
                "well_formed": (
                    1.0 if well_formed else 0.0,
                    "one English line" if well_formed else "empty, multi-line or not English",
                ),
                "intent_kept": (1.0 if passed else 0.0, reason),
            },
        )
        results.append((name, passed and well_formed, query[:70]))
    return results


async def run_llm2(run: str):
    results = []
    for name, question, tool_result, contract in LLM2_CASES:
        messages = [
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "assistant", "content": cfg.get("greeting") or "வணக்கம்!"},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_knowledge_base",
                            "arguments": json.dumps({"questions": [question]}, ensure_ascii=False),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(tool_result, ensure_ascii=False),
            },
        ]
        msg = await asyncio.to_thread(sarvam, messages, TOOLS)
        answer = (msg.get("content") or "").strip()
        passed, reason = await asyncio.to_thread(
            judge,
            "You audit one Tamil voice-agent answer for groundedness. The agent may only use "
            f"the search results given. Pass criteria for this case: {contract}",
            f"SEARCH RESULTS GIVEN:\n{json.dumps(tool_result, ensure_ascii=False)}\n\n"
            f"CALLER QUESTION:\n{question}\n\nAGENT ANSWER:\n{answer}",
        )
        record(
            "step-llm2-answer",
            run,
            f"llm2-{name}",
            f"llm2 answer · {name}",
            {"question": question, "tool_result": tool_result},
            {"answer": answer},
            {"grounded": (1.0 if passed else 0.0, reason)},
        )
        results.append((name, passed, answer[:70]))
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=f"run-{datetime.now(UTC):%Y%m%d-%H%M}")
    args = parser.parse_args()

    ensure_dataset(
        "step-topic-check",
        "NEW/FOLLOW_UP decisions of the topic classifier",
        [
            (f"topic-{n}", {"context": c, "latest": t}, {"decision": e})
            for n, c, t, e in TOPIC_CASES
        ],
    )
    ensure_dataset(
        "step-llm1-decide",
        "LLM 1: search tool vs direct reply",
        [(f"llm1-{n}", {"caller": t}, {"tool_call": e}) for n, t, e in LLM1_CASES],
    )
    ensure_dataset(
        "step-query-rewrite",
        "Caller's messages -> one English search query",
        [(f"rewrite-{n}", {"caller_messages": h}, {"intent": i}) for n, h, i in REWRITE_CASES],
    )
    ensure_dataset(
        "step-llm2-answer",
        "Grounded answers from given sheets",
        [
            (f"llm2-{n}", {"question": q, "tool_result": r}, {"contract": c})
            for n, q, r, c in LLM2_CASES
        ],
    )

    global _items
    _items = {
        name: lf.get_dataset(name).items
        for name in (
            "step-topic-check",
            "step-llm1-decide",
            "step-query-rewrite",
            "step-llm2-answer",
        )
    }

    all_results = {}
    for label, runner in (
        ("topic-check", run_topic),
        ("llm1-decide", run_llm1),
        ("query-rewrite", run_rewrite),
        ("llm2-answer", run_llm2),
    ):
        rows = await runner(args.run)
        all_results[label] = rows
        for name, ok, detail in rows:
            print(f"[{'PASS' if ok else 'FAIL'}] {label:<14} {name:<22} {detail}", flush=True)

    lf.flush()
    total = sum(len(r) for r in all_results.values())
    passed = sum(ok for r in all_results.values() for _, ok, _ in r)
    print(
        f"\n{passed}/{total} passed · run '{args.run}' pushed to {settings.langfuse_host} "
        f"(Datasets -> step-* -> Runs)"
    )


if __name__ == "__main__":
    asyncio.run(main())
