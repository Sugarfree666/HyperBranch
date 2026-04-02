You are deciding the next thought operation after new evidence has been retrieved.

You must choose exactly one operation from:
- expand
- merge
- revise
- verify

Return JSON only:
{
  "operation": "expand|merge|revise|verify",
  "reason": "...",
  "new_status": "active|expanded|revised|verified|rejected|merged",
  "revised_content": "...",
  "merge_with_thought_ids": ["th-0003"],
  "new_thoughts": [
    {
      "role": "hypothesis|constraint|bridge",
      "content": "...",
      "grounding_hints": {
        "anchors": ["..."],
        "notes": ["..."]
      }
    }
  ],
  "verification": {
    "verdict": "supported|refuted|insufficient",
    "confidence": 0.0,
    "evidence_ids": ["ev-th-0002-1"],
    "notes": "..."
  }
}

Requirements:
- `expand`: create new child thoughts that move the reasoning frontier.
- `merge`: only use when another thought materially complements this one.
- `revise`: repair the current thought using the new evidence.
- `verify`: use when the evidence is enough to support or refute the thought.
- Keep outputs minimal and executable by a controller.
