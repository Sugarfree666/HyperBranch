You are choosing the most valuable hyperedges for one reasoning branch in an iterative hypergraph RAG system.

Return JSON only:
{
  "selected_hyperedge_ids": ["..."],
  "candidate_answer": "...",
  "supporting_facts": ["..."],
  "missing_requirements": ["..."],
  "confidence": 0.0,
  "notes": "..."
}

Requirements:
- Select at most `top_k` hyperedges from `candidate_hyperedges`.
- Respect the branch role:
  - `constraint`: prefer hyperedges that satisfy the most hard constraints.
  - `relation`: prefer hyperedges whose relation structure best matches the question.
  - `anchor`: prefer hyperedges that cover more topic entities or tighten the anchor cluster.
- `candidate_answer` should be the branch's current best hypothesis, grounded only in the selected hyperedges.
- `supporting_facts` should be short factual bullets distilled from the selected hyperedges.
- If evidence is still incomplete, list the unresolved parts in `missing_requirements`.
- Keep `confidence` between 0 and 1.
