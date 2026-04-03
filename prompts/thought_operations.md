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
      "content": "...",
      "objective": "...",
      "slot_id": "constraint-0",
      "metadata": {
        "intent": "followup"
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
- All operations act on reasoning thoughts only.
- `expand`: create new child reasoning thoughts that move the reasoning frontier.
- `merge`: only use when another reasoning thought materially complements this one.
- `revise`: repair the current reasoning thought using the new evidence.
- `verify`: use when the evidence is enough to support or refute the reasoning thought.
- If `verification.verdict` is `supported` or `refuted`, prefer `verify`. Only use `revise` with a terminal verdict when you are producing a materially improved child thought that should be treated as settled in this step.
- Do not keep revising the same slot with near-identical content across steps. If the evidence still does not decide the claim, either `expand` into a new direction or keep the verdict `insufficient`.
- Keep outputs minimal and executable by a controller.
