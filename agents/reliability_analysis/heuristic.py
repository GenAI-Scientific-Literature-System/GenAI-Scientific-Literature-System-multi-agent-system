def compute_heuristic(metadata):
    score = 0.0

    if metadata.get("citations", 0) > 50:
        score += 0.2
    
    if metadata.get("year",0) >= 2020:
        score += 0.2

    if metadata.get("journal",False):
        score += 0.3

    if metadata.get("dataset_size",0) > 1000:
        score += 0.2

    return min(score, 1.0)