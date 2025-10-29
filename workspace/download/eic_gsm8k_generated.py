#!/usr/bin/env python3
"""
build_eic_gsm8k_jsonl_dataset.py

Builds a Hugging Face dataset from all .jsonl (and .jsonl.gz) files under
data/generated_cases_GSM8K in the LittleCirc1e/EIC repository.

Output dataset fields:
  - prompt
  - correct_answer
  - incorrect_answer
  - error_types         (list[str])
  - source_file         (relative path inside the repo)
  - line_idx            (int, line index inside the file, 0-based)

Requirements:
  pip install datasets huggingface_hub
  git available if using --clone (otherwise point --repo to a local path)
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import gzip
import sys

error_type2valid_error_type = {
        "adding_irrelevant_information" : "Hallucination",
        "calculation_error" : "Calculation Error",
        "confusing_formula_error" : "Formula Confusion Error",
        "counting_error" : "Counting Error",
        "missing_step" : "Missing Step",
        "operator_error" : "Operator Error",
        "referencing_context_value_error" : "Context Value Error",
        "referencing_previous_step_value_error" : "Contradictory Step",
#        "referencing_value" : "Contradictory Step",
        "unit_conversion_error" : "Unit Conversion Error"
}
mapper = {k : set() for k in error_type2valid_error_type.keys()}

def clone_repo(git_url: str, dst: str):
    if os.path.exists(dst):
        print(f"[info] destination {dst} already exists; skipping clone")
        return
    print(f"[info] cloning {git_url} -> {dst} ...")
    subprocess.run(["git", "clone", "--depth", "1", git_url, dst], check=True)
    print("[info] clone complete")

def iter_jsonl_lines(path: Path):
    """Yield (line_str, line_index) for a .jsonl or .jsonl.gz file."""
    if path.suffix == ".gz":
        # support files like generated_cases_clean.jsonl.gz
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"
    with opener(path, mode, encoding="utf-8") as fh:
        for i, raw in enumerate(fh):
            line = raw.strip()
            if not line:
                continue
            yield line, i

def parse_json_line(line: str, source: str, line_idx: int) -> Optional[Dict[str, Any]]:
    """Parse a JSON line; return dict or None on failure."""
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            # some lines might be arrays or scalars; skip if not an object
            return None
        return obj
    except json.JSONDecodeError as e:
        print(f"[warn] JSONDecodeError in {source} @ line {line_idx}: {e} -- skipping line")
        return None

def json_to_record_list_object(obj: Any) -> List[Dict[str, Any]]:
    """
    Convert a JSON object that appears on a line into a list of dicts.
    For .jsonl we expect obj to usually already be a dict representing one record.
    Keep this small for safety.
    """
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        # If a list of dicts appears in a single line, return them
        if all(isinstance(x, dict) for x in obj):
            return obj
    return []

def derive_error_type_from_path(file_path: Path, repo_root: Path) -> str:
    """
    Attempt to pick a sensible fallback error-type string from the path.
    E.g., .../data/generated_cases_GSM8K/calculation_error/calculation_error_100/...
    -> "calculation_error"
    """
    try:
        parts = file_path.relative_to(repo_root).parts
    except Exception:
        parts = file_path.parts
    # find index of "generated_cases_GSM8K" if present
    if "generated_cases_GSM8K" in parts:
        idx = parts.index("generated_cases_GSM8K")
        # take the next segment if it exists
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # fallback: use parent folder name
    return file_path.parent.name

def normalize_record(rec: Dict[str, Any], filename_error_type: str) -> Dict[str, Any]:
    """
    Normalize a single record dict to the target schema.
    Fields we search for (common names in repo):
      - question, prompt, problem, original_question
      - original_answer, original_solution, answer
      - transformed_answer, transformed_solution, wrong_answer
      - wrong_type, wrong_types, error_type, type
      - explanation, wrong_step (kept as optional metadata if present)
    """
    
    # make naming scheme consistent with paper
    """
    Error Type Definition
        Calculation Error (CA) Error appears during the calculation process.
        Counting Error (CO) Error occurs during the counting process.
        Context Value Error (CV) Error arises when attributes of named entities do not align with the information provided.
        Hallucination (HA) Error involves adding fictitious unrelated statements contradictory to the question.
        Unit Conversion Error (UC) Error occurs during unit conversion process.
        Operator Error (OP) Error involves a single operator being erroneously applied within the expression.
        Formula Confusion Error (FC) Error appears when applying formula in inappropriate scenario.
        Missing Step (MS) Error entails an incomplete generation of reasoning process, lacking a necessary step.
        Contradictory Step (CS) Error manifests inconsistency between preceding and subsequent reasoning steps.

    present "wrong types" based on filename_error_type:
    adding_irrelevant_information :
        adding_irrelevant_information
    calculation_error :
        calculation_error
    confusing_formula_error :
        confusing_formula_error
    counting_error :
        counting_error
    missing_step :
        missing_step, calculation_error
    operator_error :
        operator_error
    referencing_context_value_error :
        referencing_value, referencing_context_value_error
    referencing_previous_step_value_error :
        referencing_previous_step_value_error, calculation_error_fixed_addition
    unit_conversion_error :
        unit_conversion_error

    => well just take the filename_error_type as wrong type.
    """
    

    prompt = rec.get("question")
    correct_solution = rec.get("original_solution")
    correct_answer = rec.get("original_answer")
    incorrect_solution = rec.get("transformed_solution")
    incorrect_answer = rec.get("transformed_answer")
    et = error_type2valid_error_type[filename_error_type]
    explanation = rec.get('explanation')
    sub_error_types = rec.get('wrong_type')
    mapper[filename_error_type].update([sub_error_types] if rec.get("is_single_error") else sub_error_types)

    out = {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "incorrect_answer": incorrect_answer,
        "correct_solution" : correct_solution,
        "incorrect_solution" : incorrect_solution,
        "explanation" : explanation,
        "error_type": et,
    }

    # keep some optional metadata if present
    if "wrong_step" in rec:
        out["wrong_step"] = rec.get("wrong_step")
    
    return out

def collect_records_from_folder(repo_root: Path, target_subpath: Path) -> List[Dict[str, Any]]:
    """
    Walk the target_subpath (full path) and collect normalized records
    from all .jsonl and .jsonl.gz files.
    """
    records: List[Dict[str, Any]] = []
    pattern = "**/*.jsonl*"
    files = sorted(target_subpath.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No jsonl files found under {target_subpath} (pattern {pattern})")
    print(f"[info] found {len(files)} jsonl/jsonl.gz files under {target_subpath}")

    for fp in files:
        
        # ignore everything which isn't error category
        if "wrong_step_calculation_error" in fp.parts:
            continue

        # ignore files that are not .jsonl or .jsonl.gz
        if not (fp.suffix == ".jsonl" or (fp.suffix == ".gz" and fp.name.endswith(".jsonl.gz"))):
            # allow ".jsonl" and ".jsonl.gz"
            if fp.suffixes and fp.suffixes[-1] == ".jsonl":
                # covers cases where suffixes is [".something", ".jsonl"]
                pass
            else:
                continue

        fallback_error_type = derive_error_type_from_path(fp, repo_root)
        rel_path = fp.relative_to(repo_root)
        count_in_file = 0
        for line, idx in iter_jsonl_lines(fp):
            obj = parse_json_line(line, str(rel_path), idx)
            if obj is None:
                continue
            # Some lines might contain a single record dict, or a list of dicts.
            candidates = json_to_record_list_object(obj)
            for c in candidates:
                norm = normalize_record(c, fallback_error_type)
                # only keep if there is at least a prompt or one of the answers
                if not (norm.get("prompt") or norm.get("correct_answer") or norm.get("incorrect_answer")):
                    continue
                # add provenance
                norm["source_file"] = str(rel_path)
                norm["line_idx"] = idx
                records.append(norm)
                count_in_file += 1
        print(f"[info] -> {rel_path}: collected {count_in_file} records")
    print(f"[info] total normalized records collected: {len(records)}")
    return records

def main():
    parser = argparse.ArgumentParser(description="Build HF dataset from EIC generated_cases_GSM8K (.jsonl files)")
    parser.add_argument("--repo-url", type=str, default="https://github.com/LittleCirc1e/EIC.git", help="Git repo URL")
    parser.add_argument("--repo", type=str, default=None, help="Local repo path to use instead of cloning")
    parser.add_argument("--out-dir", type=str, default="/u/rfechner/data/eic_gsm8k_generated", help="directory where dataset will be saved")
    parser.add_argument("--keep-clone", action="store_true", help="don't delete the cloned repository (useful for inspection)")
    parser.add_argument("--push-to-hub", action="store_true", help="push dataset to the HF Hub (requires hf auth token)")
    parser.add_argument("--hub-repo-id", type=str, default=None, help="Hub repo id (e.g. username/eic_gsm8k). Required if --push-to-hub")
    args = parser.parse_args()

    # determine repo root
    tempdir = None
    repo_root = None
    try:
        if args.repo:
            repo_root = Path(args.repo).resolve()
            if not repo_root.exists():
                print(f"[error] provided --repo path does not exist: {repo_root}")
                sys.exit(1)
        else:
            tempdir = tempfile.mkdtemp(prefix="eic_clone_")
            repo_dst = Path(tempdir) / "EIC"
            clone_repo(args.repo_url, str(repo_dst))
            repo_root = repo_dst.resolve()
        
        # locate data/generated_cases_GSM8K
        target = repo_root / "data" / "generated_cases_GSM8K"
        if not target.exists() or not target.is_dir():
            print(f"[error] expected folder not found: {target}")
            sys.exit(1)

        # collect normalized records
        records = collect_records_from_folder(repo_root, target)
        if not records:
            print("[error] no records found after parsing; exiting")
            sys.exit(1)

        # create HF dataset
        try:
            from datasets import Dataset
        except Exception:
            print("[error] please `pip install datasets` and re-run")
            sys.exit(1)

        hf_dataset = Dataset.from_list(records)
        print(f"[info] dataset created with {len(hf_dataset)} examples")

        # save to disk
        out_dir = Path(args.out_dir)
        hf_dataset.to_parquet(out_dir / 'test.parquet')
        print(f"[info] dataset saved to {out_dir.resolve()}")

        # optional push to hub
        if args.push_to_hub:
            if not args.hub_repo_id:
                print("[error] --hub-repo-id is required when --push-to-hub is set")
                sys.exit(1)
            print("[info] pushing dataset to HF Hub...")
            hf_dataset.push_to_hub(args.hub_repo_id, private=False)
            print(f"[info] pushed dataset to the hub as: {args.hub_repo_id}")

    finally:
        if tempdir and not args.keep_clone:
            import shutil
            shutil.rmtree(tempdir, ignore_errors=True)
            print(f"[info] removed temporary clone at {tempdir}")
        elif tempdir:
            print(f"[info] kept cloned repo at {repo_dst}")

if __name__ == "__main__":
    main()
