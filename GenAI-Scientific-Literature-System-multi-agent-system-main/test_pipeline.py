#!/usr/bin/env python3
# test_pipeline.py

import os
from dotenv import load_dotenv

from utils.api_keys import load_groq_api_keys
from pipeline.preprocessing import QueryPreprocessor
from pipeline.retrieval import Retriever
from pipeline.embedding import EmbeddingEngine
from pipeline.aggregator import Aggregator

load_dotenv()

api_keys = load_groq_api_keys()
query = "sleep deprivation and cognitive decline in older adults"

# step 1 — preprocess + classify domain
print("\n=== STEP 1: Preprocessing ===")
preprocessor = QueryPreprocessor(api_key=api_keys)
processed = preprocessor.process(query)
print(f"Cleaned query: {processed['cleaned_query']}")
print(f"Domains: {processed['domains']}")

# step 2 — retrieve papers
print("\n=== STEP 2: Retrieval ===")
retriever = Retriever(top_k_per_source=8, debug=True)
raw_papers = retriever.retrieve(
    query=processed["cleaned_query"],
    domains=processed["domains"]
)
print(f"Raw papers retrieved: {len(raw_papers)}")

# step 3 — embed + aggregate + dedup + rank
print("\n=== STEP 3: Embedding + Aggregation ===")
engine = EmbeddingEngine(debug=True)
aggregator = Aggregator(embedding_engine=engine, final_top_k=15, debug=True)
final_papers = aggregator.aggregate(raw_papers, query=processed["cleaned_query"])

# print final results
print(f"\n=== FINAL OUTPUT: {len(final_papers)} papers ===")
for i, paper in enumerate(final_papers, 1):
    print(f"\n[{i}] {paper.get('title', 'No title')}")
    print(f"    Source : {paper.get('source')}")
    print(f"    Year   : {paper.get('year')}")
    print(f"    Score  : {paper.get('score', 0.0):.4f}")
    print(f"    DOI    : {paper.get('doi')}")
    print(f"    Authors: {', '.join(paper.get('authors', [])[:3])}")
