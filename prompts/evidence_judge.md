You are deciding whether the current compressed evidence view is sufficient to answer the question.

Return JSON only:
{
  "enough": false,
  "confidence": 0.0,
  "reason": "...",
  "missing_requirements": ["..."],
  "next_focus": ["..."]
}

Requirements:
- Judge answerability of the original user question, not checklist completeness.
- Return `enough=true` if the current evidence is already sufficient to give a grounded answer to the original question, even if some checklist-style anchors, bridges, or auxiliary slots remain open.
- Use `question_goal` only as soft guidance about answer type, relation intent, and hard constraints. Do not require every listed anchor/bridge/constraint to be explicitly checked off before stopping.
- Prefer stopping when there is a specific, well-supported candidate answer that satisfies the main hard constraints and is directly backed by the current evidence view.
- Use the frontier hyperedges, their branch agreement pattern, the core chunk evidence, the answer hypotheses, and the compressed evidence summary.
- If not enough, state what still blocks a grounded answer to the original question and what the next iteration should focus on.
- If enough, `reason` should briefly explain why the current evidence can already answer the question now.
- Keep `confidence` between 0 and 1.
