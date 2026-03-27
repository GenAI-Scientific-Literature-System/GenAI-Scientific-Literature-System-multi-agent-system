"""
Core Orchestrator for Agent 4
================================
Manages paper loading, agent execution, and result persistence.
"""

import os
import json
import glob
from typing import Dict, Optional


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output")


def load_papers(data_dir: str = DATA_DIR) -> Dict[str, str]:
    """
    Load all .txt papers from data directory.
    Returns dict: {paper_id -> text}
    """
    papers = {}
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not txt_files:
        print(f"[Orchestrator] No .txt files found in {data_dir}. Using built-in sample papers.")
        return _get_sample_papers()

    for fpath in txt_files:
        paper_id = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, "r", encoding="utf-8") as f:
            papers[paper_id] = f.read()
        print(f"[Orchestrator] Loaded: {paper_id} ({len(papers[paper_id])} chars)")

    return papers


def _get_sample_papers() -> Dict[str, str]:
    """Built-in sample papers for demo/testing."""
    return {
        "Attention_Is_All_You_Need": (
            "We propose the Transformer, a model architecture relying entirely on an attention mechanism. "
            "Our model achieves 28.4 BLEU on WMT translation, outperforming all previous baselines. "
            "We demonstrate that attention is the only component needed for state-of-the-art NLP performance. "
            "Results show that multi-head attention captures both local and global dependencies efficiently. "
            "The Transformer requires significantly less computation than recurrent architectures. "
            "We show that self-attention layers are more interpretable than recurrent layers. "
            "Pre-training on large corpora dramatically improves downstream task performance. "
            "Our experiments confirm that depth and attention heads are critical hyperparameters."
        ),
        "CNN_Beats_Transformer": (
            "We find that convolutional networks can match transformer performance on multiple NLP tasks. "
            "Our experiments demonstrate that attention mechanisms are not strictly necessary for good results. "
            "The proposed CNN achieves 87.3% accuracy on text classification, comparable to BERT. "
            "We show that locality in convolutions captures relevant patterns more efficiently. "
            "Unlike transformers, our model scales linearly with sequence length. "
            "Results indicate that architectural simplicity does not sacrifice accuracy. "
            "We demonstrate that pre-training is beneficial but not sufficient without proper inductive bias. "
            "Our findings suggest that the NLP community over-relies on attention-based models."
        ),
        "Hybrid_Architecture_Survey": (
            "This paper surveys hybrid architectures combining attention and convolution. "
            "We find that neither pure attention nor pure convolution is universally superior. "
            "Our analysis shows that task-specific inductive biases determine optimal architecture choice. "
            "Results from 47 benchmarks indicate that hybrid models outperform single-paradigm models by 3.2%. "
            "We demonstrate that computational cost is the primary bottleneck in real-world deployment. "
            "The survey identifies several open questions: optimal mixing ratios, training strategies, and robustness. "
            "We observe that pre-training objectives significantly interact with architectural choices. "
            "Future work should explore dynamic architecture selection based on input characteristics."
        )
    }


def save_results(results: Dict, filename: str = "agent4_results.json") -> str:
    """Save analysis results to output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)

    # Remove non-serializable items for JSON output
    serializable = {
        k: v for k, v in results.items()
        if k != "all_claims_structured"
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"[Orchestrator] Results saved → {out_path}")
    return out_path


def load_results(filename: str = "agent4_results.json") -> Optional[Dict]:
    """Load previously saved results."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_graph(graph_dict: Dict, filename: str = "claim_graph.json") -> str:
    """Save graph data for visualization."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=2)
    print(f"[Orchestrator] Graph saved → {out_path}")
    return out_path
