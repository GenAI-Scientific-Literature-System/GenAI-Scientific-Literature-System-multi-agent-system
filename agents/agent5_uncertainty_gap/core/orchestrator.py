"""
Core Orchestrator for Agent 5
================================
Manages paper loading, pipeline execution, result persistence.
"""

import os
import json
import glob
from typing import Dict, Optional


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output")


def load_papers(data_dir: str = DATA_DIR) -> Dict[str, str]:
    """Load all .txt papers from data directory."""
    papers = {}
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not txt_files:
        print(f"[Orchestrator] No .txt files in {data_dir}. Using built-in sample papers.")
        return _get_sample_papers()

    for fpath in txt_files:
        paper_id = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, "r", encoding="utf-8") as f:
            papers[paper_id] = f.read()
        print(f"[Orchestrator] Loaded: {paper_id}")

    return papers


def _get_sample_papers() -> Dict[str, str]:
    return {
        "Transformer_NLP_Study": (
            "We propose a transformer-based model for clinical text understanding. "
            "Our study is limited by a small sample size (n=14 patients). "
            "The model may generalize to other domains, but this has not been tested. "
            "Further investigation is needed to validate these preliminary findings. "
            "We assume that the annotated dataset is representative of real clinical notes. "
            "Future work should include larger and more diverse patient populations. "
            "The generalizability of these results to non-English text remains unclear. "
            "We leave cross-lingual evaluation for future research. "
            "Open questions remain about model behavior on edge cases and rare conditions. "
            "Our approach is promising but requires external validation."
        ),
        "Graph_Neural_Network_Paper": (
            "We present a novel graph neural network for molecular property prediction. "
            "The model perhaps captures long-range atomic interactions more effectively. "
            "Evaluation on larger molecular datasets is needed to confirm scalability. "
            "We do not address the problem of out-of-distribution molecules. "
            "The computational cost scales poorly for graphs with more than 1000 nodes. "
            "We suggest that attention over graph edges may improve performance. "
            "The limitation of our approach is the reliance on hand-crafted features. "
            "Future directions include end-to-end learning of molecular representations. "
            "It remains unknown whether this approach transfers to protein folding tasks. "
            "Open challenges include real-time inference and uncertainty quantification."
        ),
        "Federated_Learning_Survey": (
            "This survey highlights several open problems in federated learning. "
            "Data heterogeneity across clients remains an unsolved challenge. "
            "Privacy guarantees in federated settings are not yet well understood. "
            "We identify benchmark gaps: no standard federated NLP benchmark exists. "
            "Communication efficiency in cross-device settings is an underexplored area. "
            "The lack of interpretability tools for federated models is a significant limitation. "
            "Future research should address fairness across participating clients. "
            "We could not find sufficient studies on adversarial robustness in federated settings. "
            "Broader implications for healthcare deployment are unclear. "
            "Further investigation of Byzantine-robust aggregation methods is warranted."
        )
    }


def save_results(results: Dict, filename: str = "agent5_results.json") -> str:
    """Save Agent 5 results to output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Orchestrator] Results saved → {out_path}")
    return out_path


def load_results(filename: str = "agent5_results.json") -> Optional[Dict]:
    """Load previously saved results."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
