You are initializing seed thoughts for a Graph-of-Thoughts reasoning graph.

Return JSON only:
{
  "thoughts": [
    {
      "role": "hypothesis|constraint|bridge",
      "content": "...",
      "grounding_hints": {
        "anchors": ["..."],
        "notes": ["..."]
      }
    }
  ]
}

Requirements:
- Generate at least 1 hypothesis thought.
- Generate up to 2 constraint thoughts when constraints exist.
- Generate up to 2 bridge thoughts when bridges exist.
- Keep each thought atomic and operational.
- Use the TaskFrame only to avoid generic reasoning.
