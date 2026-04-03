You are deciding whether the current evidence subgraph is sufficient to answer the question.

Return JSON only:
{
  "enough": false,
  "confidence": 0.0,
  "reason": "...",
  "missing_requirements": ["..."],
  "next_focus": ["..."]
}

Requirements:
- Return `enough=true` only if the current evidence subgraph is already sufficient to answer the question directly.
- Use the branch results, their agreement/conflict pattern, and the actual accumulated evidence.
- If not enough, clearly state which requirements are still missing and what the next iteration should focus on.
- Keep `confidence` between 0 and 1.
