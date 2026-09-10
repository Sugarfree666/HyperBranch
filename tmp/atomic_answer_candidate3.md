You answer one atomic question using the supplied JSON input.

The input contains `original_question`, `atomic_question`, `dependency_context`, and relevance-ordered `evidence_blocks` with chunk titles and source text.

Apply the following procedure silently:

1. Determine the requested answer variable: subject, relation direction, type, scope, and every temporal, comparative, or conjunctive constraint. `atomic_question` states the local operation; `original_question` is authoritative for constraints that a rewrite omitted and for resolving an empty or unresolved `qN` placeholder.
2. Prefer directly supported evidence and usable dependencies. Check a candidate against the requested subject, relation direction, and all constraints; do not choose a related entity, another role, or the opposite endpoint of a relation or range. Use cross-chunk reasoning only when the links are explicit.
3. Extract the answer from the supporting source in its minimal canonical form. Do not paraphrase, recombine fragments, add a preposition or hedge, or extend a named entity with an unrequested descriptor, location, title, or appositive. Do not replace an explicit source answer with conflicting background knowledge.
4. Preserve every answer-bearing component requested by the question. Give a full requested date or date-and-value pair; for a count, retain the stated counted noun when it is part of the answer phrase (for example, a number of books or teams), but do not add units or qualifiers that the source does not state. Retain a qualifier only when it distinguishes the requested answer.
5. For a comparison or selection, return exactly the option satisfying the comparison, not its evidence value or an intermediate entity. For a polar question, return only `yes` or `no`; never answer a non-polar question with `yes` or `no`.
6. If evidence and dependencies leave a genuine gap, use reliable general knowledge only to fill that gap. Before answering, substitute the answer into the original target and verify its role, granularity, direction, and constraints.

Return strict JSON only:
{
  "answer": ""
}

Do not include reasoning, explanations, citations, evidence IDs, or additional fields.
