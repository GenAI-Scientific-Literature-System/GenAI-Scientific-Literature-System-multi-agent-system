"""
Claim Graph Builder
====================
Constructs a NetworkX graph from Agent 4 comparison results.
Nodes = papers or claims; Edges = semantic relationships.
"""

import json
from typing import Dict, List
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


EDGE_COLOR_MAP = {
    "agreement": "#2ecc71",
    "contradiction": "#e74c3c",
    "partial": "#f39c12",
    "novel": "#9b59b6"
}


def build_claim_graph(comparisons: List[Dict]) -> "nx.DiGraph":
    """
    Build a directed graph from claim comparison results.
    Nodes = paper IDs, Edges = relationship type + confidence.
    """
    if not HAS_NX:
        raise ImportError("networkx is required. Run: pip install networkx")

    G = nx.DiGraph()

    # Track edge weights for aggregation
    edge_data = {}

    for comp in comparisons:
        papers = comp.get("papers", [])
        if len(papers) < 2:
            continue
        p1, p2 = papers[0], papers[1]
        rel = comp["relationship"]
        conf = comp["confidence"]

        # Add nodes
        for p in [p1, p2]:
            if p not in G:
                G.add_node(p, type="paper", label=p)

        # Aggregate edges (take max confidence per paper pair + relationship)
        key = (p1, p2, rel)
        if key not in edge_data or edge_data[key]["confidence"] < conf:
            edge_data[key] = {
                "relationship": rel,
                "confidence": conf,
                "color": EDGE_COLOR_MAP.get(rel, "#888888"),
                "explanation": comp.get("explanation", ""),
                "evidence": comp.get("evidence", [])
            }

    for (p1, p2, rel), data in edge_data.items():
        G.add_edge(p1, p2,
                   relationship=data["relationship"],
                   confidence=data["confidence"],
                   color=data["color"],
                   explanation=data["explanation"],
                   weight=data["confidence"])

    return G


def graph_to_dict(G: "nx.DiGraph") -> Dict:
    """Serialize graph to JSON-compatible dict."""
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        nodes.append({
            "id": node_id,
            "label": attrs.get("label", node_id),
            "type": attrs.get("type", "paper"),
            "degree": G.degree(node_id)
        })

    edges = []
    for u, v, attrs in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "relationship": attrs.get("relationship", "unknown"),
            "confidence": attrs.get("confidence", 0.5),
            "color": attrs.get("color", "#888888"),
            "explanation": attrs.get("explanation", "")
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges)
    }


def compute_graph_metrics(G: "nx.DiGraph") -> Dict:
    """Compute basic graph statistics."""
    if not HAS_NX or len(G.nodes()) == 0:
        return {}

    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "most_connected": max(dict(G.degree()).items(), key=lambda x: x[1], default=("N/A", 0))[0],
    }

    try:
        metrics["avg_clustering"] = round(nx.average_clustering(G.to_undirected()), 4)
    except Exception:
        metrics["avg_clustering"] = 0.0

    # Relationship breakdown
    rel_counts = {}
    for _, _, d in G.edges(data=True):
        rel = d.get("relationship", "unknown")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    metrics["relationship_breakdown"] = rel_counts

    return metrics


if __name__ == "__main__":
    # Smoke test
    sample_comparisons = [
        {
            "papers": ["PaperA", "PaperB"],
            "relationship": "agreement",
            "confidence": 0.85,
            "explanation": "Both papers agree on transformer superiority.",
            "evidence": ["Transformers achieve SOTA", "Our model surpasses BERT"]
        },
        {
            "papers": ["PaperA", "PaperC"],
            "relationship": "contradiction",
            "confidence": 0.72,
            "explanation": "Papers disagree on attention necessity.",
            "evidence": ["Attention is crucial", "Attention is not required"]
        },
        {
            "papers": ["PaperB", "PaperC"],
            "relationship": "partial",
            "confidence": 0.61,
            "explanation": "Partial overlap in efficiency claims.",
            "evidence": ["Efficient inference", "Reduced compute cost"]
        }
    ]

    if HAS_NX:
        G = build_claim_graph(sample_comparisons)
        gdict = graph_to_dict(G)
        metrics = compute_graph_metrics(G)
        print("Graph dict:", json.dumps(gdict, indent=2))
        print("Metrics:", json.dumps(metrics, indent=2))
    else:
        print("networkx not installed — skipping graph build.")
