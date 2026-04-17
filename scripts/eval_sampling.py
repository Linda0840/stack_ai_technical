import argparse
import pandas as pd
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description="Sample PDFs and queries from OpenRAGBench.")
parser.add_argument("--n_pdfs",    type=int, default=500, help="Number of PDFs to sample (default: 500)")
parser.add_argument("--n_queries", type=int, default=20,  help="Number of queries to sample (default: 20)")
args = parser.parse_args()

N_PDFS    = args.n_pdfs
N_QUERIES = args.n_queries

# ── 1. Paths ──────────────────────────────────────────────────────────────────
src_pdf_folder  = Path("/Users/xiaoyanzhang/Downloads/OpenRAGBench/pdfs")
dst_pdf_folder  = Path(f"/Users/xiaoyanzhang/Downloads/OpenRAGBench/pdfs_sample_{N_PDFS}")
eval_csv_path   = Path("/Users/xiaoyanzhang/Downloads/OpenRAGBench/Eval_data.csv")
output_csv_path = Path(f"{N_PDFS}_random.csv")

# ── 2. Create destination folder ──────────────────────────────────────────────
dst_pdf_folder.mkdir(parents=True, exist_ok=True)

# ── 3. Get all PDFs and randomly sample N_PDFS ────────────────────────────────
all_pdfs = [p for p in src_pdf_folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]

if len(all_pdfs) < N_PDFS:
    raise ValueError(f"Only found {len(all_pdfs)} PDFs; cannot sample {N_PDFS}.")

sampled_pdf_names = set(
    pd.Series([p.name for p in all_pdfs]).sample(N_PDFS, random_state=40)
)

# ── 4. Copy sampled PDFs into new folder ──────────────────────────────────────
copied_count = 0
for pdf_path in all_pdfs:
    if pdf_path.name in sampled_pdf_names:
        shutil.copy(pdf_path, dst_pdf_folder / pdf_path.name)
        copied_count += 1

# ── 5. Load eval CSV ──────────────────────────────────────────────────────────
df = pd.read_csv(eval_csv_path)

# ── 6. Filter rows whose pdf_filename is in the sampled set ──────────────────
filtered = df[df["pdf_filename"].isin(sampled_pdf_names)].copy()

if len(filtered) < N_QUERIES:
    print(f"Warning: only {len(filtered)} rows available, using all rows.")
    sampled_queries = filtered
else:
    sampled_queries = filtered.sample(n=N_QUERIES, random_state=42)

# ── 7. Inspect & save ─────────────────────────────────────────────────────────
print(f"Total rows      : {len(df)}")
print(f"Kept rows       : {len(sampled_queries)}")
print(f"PDFs sampled    : {len(sampled_pdf_names)}")
print(f"PDFs covered    : {sampled_queries['pdf_filename'].nunique()}")
print(f"PDFs copied     : {copied_count}")
print(f"Sample folder   : {dst_pdf_folder.resolve()}")
print(f"Filtered CSV    : {output_csv_path.resolve()}")

sampled_queries.to_csv(output_csv_path, index=False)
