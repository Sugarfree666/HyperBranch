You are reconciling three parallel reasoning branches in an iterative hypergraph RAG system.

Return JSON only:
{
  "consensus_answer": "...",
  "agreement_groups": [["constraint", "anchor"]],
  "conflicts": [
    {
      "answer": "...",
      "branches": ["relation"]
    }
  ],
  "preferred_branches": ["constraint", "anchor"],
  "missing_requirements": ["..."],
  "notes": "..."
}

Requirements:
- Merge branches when they point to the same answer with complementary evidence.
- Contrast branches when they disagree, preferring the answer that is more direct, more constrained, and more consistent with the other branches.
- `consensus_answer` may be empty if the evidence is still too weak.
- `preferred_branches` should list the branches currently carrying the strongest support.
- `missing_requirements` should state what still blocks confident resolution.
