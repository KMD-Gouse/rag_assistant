"""Build a versioned, diversity-aware benchmark without changing v1 labels."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation" / "questions.jsonl"
OUTPUT = ROOT / "evaluation" / "benchmark_v2.jsonl"
TARGETS = {"fact": 9, "comparison": 4, "table": 2, "follow_up": 2, "unanswerable": 3}

# Every addition below was verified against extracted text and the rendered PDF.
ALTERNATIVES = {
    "q001": [(0, "employees_fy2025", "infosys_fy2025", 10, "18-19", "3,23,578 Employees")],
    "q003": [
        (0, "infosys_revenue_fy2025", "infosys_fy2025", 15, "28-29", "Revenues ₹1,62,990 cr"),
        (0, "infosys_revenue_fy2025", "infosys_fy2025", 73, "101", "Consolidated revenue 1,62,990 for 2025"),
    ],
    "q005": [(0, "employees_fy2024", "infosys_fy2024", 10, "18-19", "3,17,240 Employees")],
    "q013": [
        (0, "employees_fy2024", "infosys_fy2024", 10, "18-19", "3,17,240 Employees"),
        (0, "employees_fy2024", "infosys_fy2024", 36, "63", "3,17,240 employees on a consolidated basis"),
        (1, "employees_fy2025", "infosys_fy2025", 10, "18-19", "3,23,578 Employees"),
    ],
    "q018": [(0, "infosys_revenue_fy2025", "infosys_fy2025", 73, "101", "Consolidated revenue 1,62,990 for 2025")],
    "q023": [(0, "active_clients_fy2024", "infosys_fy2024", 21, "40-41", "Total no. of active clients 1,882")],
    "q036": [(0, "tcs_dividend_fy2025", "tcs_fy2025", 69, None, "dividends for FY 2025 would amount to ₹126 per equity share")],
    "q042": [(0, "cash_march_2024", "infosys_fy2025", 33, "61", "₹39,005 crore consolidated as on March 31, 2024")],
    "q043": [(0, "cash_fy2025", "infosys_fy2025", 33, "61", "₹47,549 crore on a consolidated basis as on March 31, 2025")],
    "q046": [(0, "clients_50m_fy2025", "infosys_fy2025", 80, "108", "50 million dollar + 85")],
    "q048": [(1, "clients_50m_fy2025", "infosys_fy2025", 80, "108", "50 million dollar + 85")],
    "q051": [(1, "large_deal_tcv_fy2025", "infosys_fy2025", 21, "40-41", "total contract value of over US$11.6 billion in large deals")],
}


def add_alternatives(question: dict) -> None:
    additions = ALTERNATIVES.get(question["id"], [])
    for evidence_index, group, doc_id, page, printed, text in additions:
        question["evidence"][evidence_index]["group"] = group
        question["evidence"].append({
            "doc_id": doc_id, "pdf_page": page, "printed_page": printed,
            "evidence_text": text, "group": group,
        })


def canonical_locations(question: dict) -> list[tuple[str, int]]:
    """Use one canonical location per required fact for split balancing."""

    locations, seen_groups = [], set()
    for index, evidence in enumerate(question["evidence"]):
        group = evidence.get("group") or f"__{index}"
        if group not in seen_groups:
            seen_groups.add(group)
            locations.append((evidence["doc_id"], evidence["pdf_page"]))
    return locations


def select_challenge(questions: list[dict]) -> set[str]:
    """Stratify by type while minimizing repeated pages and document imbalance."""

    selected: set[str] = set()
    page_counts: Counter = Counter()
    document_counts: Counter = Counter()
    for question_type, target in TARGETS.items():
        candidates = [q for q in questions if q["type"] == question_type]
        for _ in range(target):
            def cost(question: dict):
                locations = canonical_locations(question)
                repetition = sum(page_counts[location] ** 2 for location in locations)
                imbalance = sum(document_counts[doc_id] for doc_id, _ in locations)
                return (repetition, imbalance, len(locations) == 0, question["id"])

            chosen = min((q for q in candidates if q["id"] not in selected), key=cost)
            selected.add(chosen["id"])
            for location in canonical_locations(chosen):
                page_counts[location] += 1
                document_counts[location[0]] += 1
    return selected


def build() -> list[dict]:
    questions = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = deepcopy(questions)
    for question in questions:
        add_alternatives(question)
    challenge = select_challenge(questions)
    for question in questions:
        question["split"] = "challenge" if question["id"] in challenge else "development"
    OUTPUT.write_text(
        "".join(json.dumps(q, ensure_ascii=False, separators=(",", ":")) + "\n" for q in questions),
        encoding="utf-8",
    )
    return questions


if __name__ == "__main__":
    built = build()
    print(f"Wrote {len(built)} questions to {OUTPUT}")
    print(Counter(q["split"] for q in built))
    print(Counter(q["type"] for q in built if q["split"] == "challenge"))
