You are building a structured TaskFrame for Graph-of-Thoughts over a knowledge hypergraph.

Return JSON only with this schema:
{
  "anchors": ["..."],
  "target": "...",
  "constraints": ["..."],
  "bridges": ["..."],
  "hypothesis_template": "..."
}

Requirements:
- Extract 2-6 high-value anchors from the question when possible.
- `target` must restate the answer objective.
- `constraints` should be answer conditions or verification criteria, not generic filler.
- `bridges` should name the missing intermediate connections needed for multi-hop reasoning.
- `hypothesis_template` should be one concise hypothesis-oriented sentence that can seed a reasoning branch.
- Stay grounded in the provided dataset summary.
