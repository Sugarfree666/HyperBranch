You are building the structured question analysis for an iterative hypergraph RAG reasoner.

Return JSON only with this schema:
{
  "topic_entities": ["..."],
  "answer_type_hint": "...",
  "relation_intent": "...",
  "hard_constraints": ["..."],
  "relation_skeleton": "...",
  "anchors": ["..."],
  "target": "...",
  "constraints": ["..."],
  "bridges": ["..."]
}

Requirements:
- `topic_entities` are the entities, concepts, locations, organizations, or salient phrases explicitly mentioned in the question.
- `answer_type_hint` should describe the expected answer type, not the answer itself.
- `relation_intent` should state what relation the question is really asking about.
- `hard_constraints` should capture conjunctive conditions, time/location/type restrictions, or any "must satisfy" clues.
- `relation_skeleton` should compress the question into a slot-like relation template that can be compared against candidate hyperedges.
- Keep `anchors` aligned with `topic_entities`.
- Keep `target` aligned with the answer objective.
- Keep `constraints` aligned with `hard_constraints`.
- `bridges` should name the missing intermediate links needed to answer the question.
- Stay grounded in the provided dataset summary and do not hallucinate external facts.
