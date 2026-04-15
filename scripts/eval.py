"""
eval.py — RAG pipeline evaluator
Usage: python eval.py --api_key YOUR_MISTRAL_KEY [--top_k 5]

Edit the three paths in the CONFIG block below before each run.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import httpx

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — change these three lines between runs
# ══════════════════════════════════════════════════════════════════════════════

PDFS_DIR     = Path("data/eval/pdfs_sample_200")          # folder of PDFs to ingest
EVAL_CSV     = Path("data/eval/200_random.csv") # queries + ground-truth answers
RESULTS_FILE = Path("data/results/eval_200_results.csv")  # output

# ══════════════════════════════════════════════════════════════════════════════

API_BASE   = "http://localhost:8000"
INGEST_URL = f"{API_BASE}/api/v1/ingest"
QUERY_URL  = f"{API_BASE}/api/v1/query"
CLEAR_URL  = f"{API_BASE}/api/v1/ingest"


# ── helpers ──────────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """Call Mistral embedding API directly for eval scoring."""
    from mistralai.client import Mistral
    client = Mistral(api_key=api_key)
    resp   = client.embeddings.create(model="mistral-embed", inputs=texts)
    return [list(row.embedding) for row in resp.data]


def llm_judge(query: str, ground_truth: str, agent_answer: str, api_key: str) -> float:
    """Ask Mistral to score the agent answer 0.0–10.0 against the ground truth."""
    from mistralai.client import Mistral
    from mistralai.client.models.systemmessage import SystemMessage
    from mistralai.client.models.usermessage   import UserMessage

    client = Mistral(api_key=api_key)
    prompt = (
        f"Question: {query}\n\n"
        f"Ground truth answer: {ground_truth}\n\n"
        f"Agent answer: {agent_answer}\n\n"
        "Score the agent answer on a scale of 0.0 to 10.0.\n"
        "10 = fully correct and faithful to the ground truth.\n"
        "0  = wrong, hallucinated, or completely missing the point.\n"
        "Reply with ONLY a single number. No explanation."
    )
    resp = client.chat.complete(
        model="mistral-small-latest",
        temperature=0.0,
        max_tokens=8,
        messages=[
            SystemMessage(content="You are a strict RAG evaluator."),
            UserMessage(content=prompt),
        ],
    )
    try:
        return float(resp.choices[0].message.content.strip())
    except (ValueError, AttributeError):
        return -1.0


# ── ingestion ─────────────────────────────────────────────────────────────────

def clear_kb():
    r = httpx.delete(CLEAR_URL, timeout=30)
    r.raise_for_status()
    print("Knowledge base cleared.")


MAX_PDF_SIZE_MB = 20   # must match server's max_file_size_mb setting


def ingest_pdfs(pdf_paths: list[Path], batch_size: int = 20) -> list[str]:
    """
    Upload PDFs in batches of `batch_size` to respect the server's
    max_files_per_request limit.  Files exceeding MAX_PDF_SIZE_MB are skipped
    with a warning instead of crashing the whole run.

    Returns the list of filenames that were successfully ingested.
    """
    max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024

    # Pre-filter: separate valid files from oversized ones.
    valid, skipped = [], []
    for p in pdf_paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        if p.stat().st_size > max_bytes:
            skipped.append((p.name, size_mb))
        else:
            valid.append(p)

    if skipped:
        print(f"  ⚠ Skipping {len(skipped)} file(s) that exceed {MAX_PDF_SIZE_MB} MB:")
        for name, mb in skipped:
            print(f"      {name}  ({mb:.1f} MB)")

    if not valid:
        raise RuntimeError("No valid PDFs to ingest after size filtering.")

    total_chunks = 0
    batches = [valid[i:i + batch_size] for i in range(0, len(valid), batch_size)]
    for b_idx, batch in enumerate(batches, 1):
        files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in batch]
        r = httpx.post(INGEST_URL, files=files, timeout=300)
        r.raise_for_status()
        data = r.json()
        total_chunks += data["total_chunks"]
        print(f"  Batch {b_idx}/{len(batches)}: ingested {len(batch)} files "
              f"→ {data['total_chunks']} chunks")

    print(f"Ingestion complete: {len(valid)} files, {total_chunks} total chunks.")
    return [p.name for p in valid]


# ── query ─────────────────────────────────────────────────────────────────────

def run_query(query: str, top_k: int = 5) -> dict:
    r = httpx.post(QUERY_URL,
                   json={"query": query, "top_k": top_k},
                   timeout=60)
    r.raise_for_status()
    return r.json()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Mistral API key for scoring")
    parser.add_argument("--top_k",   type=int, default=5)
    args = parser.parse_args()

    print(f"PDFs dir    : {PDFS_DIR}")
    print(f"Eval CSV    : {EVAL_CSV}")
    print(f"Results file: {RESULTS_FILE}")
    print()

    assert PDFS_DIR.is_dir(),  f"PDFs directory not found: {PDFS_DIR}"
    assert EVAL_CSV.is_file(), f"Eval CSV not found: {EVAL_CSV}"
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(EVAL_CSV)))

    # ── Step 1: clear KB and ingest eval PDFs ────────────────────────────────
    clear_kb()
    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    assert pdf_files, f"No PDFs found in {PDFS_DIR}"
    ingested_names = set(ingest_pdfs(pdf_files))
    time.sleep(2)   # let indexing settle

    # Only run queries whose correlated PDF was actually ingested.
    all_rows = rows
    rows = [r for r in all_rows if r["pdf_filename"] in ingested_names]
    skipped_rows = len(all_rows) - len(rows)
    if skipped_rows:
        print(f"  ⚠ Skipping {skipped_rows} query row(s) whose PDF was not ingested.\n")

    # ── Step 2: run each query, collect results ───────────────────────────────
    results = []
    for i, row in enumerate(rows, 1):
        query        = row["query"]
        ground_truth = row.get("answer", "")   # adjust column name
        pdf_name     = row["pdf_filename"]

        print(f"[{i}/{len(rows)}] {query[:60]}…")

        response = run_query(query, top_k=args.top_k)

        agent_answer       = response.get("answer", "")
        sources            = ", ".join(
            f"{s['filename']} p.{s['page_number']}"
            for s in response.get("sources", [])
        )
        evidence_sufficient  = response.get("evidence_sufficient")
        hallucination_warning = response.get("hallucination_warning")
        query_shape          = response.get("query_shape", "")

        # ── Semantic similarity ───────────────────────────────────────────────
        sem_score = 0.0
        if ground_truth and agent_answer:
            try:
                gt_emb, ag_emb = get_embeddings(
                    [ground_truth, agent_answer], args.api_key
                )
                sem_score = round(cosine(gt_emb, ag_emb), 4)
            except Exception as e:
                print(f"  Embedding error: {e}")

        # ── LLM-as-judge ─────────────────────────────────────────────────────
        judge_score = -1.0
        if ground_truth and agent_answer:
            try:
                judge_score = round(
                    llm_judge(query, ground_truth, agent_answer, args.api_key), 1
                )
            except Exception as e:
                print(f"  Judge error: {e}")

        results.append({
            "query":                 query,
            "correlated_pdf":        pdf_name,
            "ground_truth":          ground_truth,
            "agent_answer":          agent_answer,
            "semantic_similarity":   sem_score,
            "llm_judge_score":       judge_score,
            "evidence_sufficient":   evidence_sufficient,
            "hallucination_warning": hallucination_warning,
            "query_shape":           query_shape,
            "sources_cited":         sources,
        })

        time.sleep(0.5)   # be gentle on rate limits

    # ── Step 3: write results CSV ─────────────────────────────────────────────
    fieldnames = list(results[0].keys())
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Step 4: print summary ─────────────────────────────────────────────────
    valid = [r for r in results if r["llm_judge_score"] >= 0]
    print(f"\n{'─'*55}")
    print(f"Results written to {RESULTS_FILE}")
    print(f"Queries run          : {len(results)}")
    if valid:
        avg_sem   = sum(r["semantic_similarity"] for r in valid) / len(valid)
        avg_judge = sum(r["llm_judge_score"]     for r in valid) / len(valid)
        print(f"Avg semantic sim     : {avg_sem:.4f}  (0–1)")
        print(f"Avg LLM judge score  : {avg_judge:.1f}  (0–10)")
    print(f"Hallucination flags  : {sum(1 for r in results if r['hallucination_warning'])}")
    print(f"Insufficient evidence: {sum(1 for r in results if not r['evidence_sufficient'])}")


if __name__ == "__main__":
    main()