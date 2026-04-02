You are generating the final answer for a Graph-of-Thoughts over Knowledge Hypergraphs pipeline.

Return JSON only:
{
  "answer": "...",
  "reasoning_summary": "...",
  "confidence": 0.0,
  "remaining_gaps": ["..."]
}

Requirements:
- Answer the user's question directly.
- Use the verified reasoning thoughts and their grounding evidence.
- If evidence is incomplete, say so in `remaining_gaps` instead of fabricating certainty.
- Keep `confidence` between 0 and 1.
