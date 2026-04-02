# GoTHyper MVP

Minimal runnable MVP for `Graph-of-Thoughts over Knowledge Hypergraphs for Multi-hop RAG`, aligned to `GoTHyper.md` and the real dataset under `datasets/agriculture`.

## Run

Online mode:

```bash
python -m goth_hyper.cli --config configs/agriculture.yaml --question "What region is known for its robust distribution network for local food and has a college operating a farm for over a hundred years?"
```

The CLI auto-loads `.env` from the project root before creating the online client. Create a local `.env` file like:

```bash
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

Offline smoke mode:

```bash
python -m goth_hyper.cli --config configs/agriculture.yaml --mock-llm --question "How can urban farms build community support while dealing with lead contamination in soil?"
```

## Output

Each run creates `runs/<timestamp>_<slug>/` with:

- `run.log`
- `events.jsonl`
- `llm_calls.jsonl` in online mode
- `artifacts/dataset_summary.json`
- `artifacts/task_frame.json`
- `artifacts/thought_graph.json`
- `artifacts/final_answer.json`

## Module Boundaries

- `goth_hyper/data`: GraphML + vector DB + document loaders
- `goth_hyper/retrieval`: global hypergraph evidence retrieval
- `goth_hyper/reasoning`: TaskFrame, scoring, operations, controller
- `goth_hyper/llm`: OpenAI-compatible client, prompts, reasoning service
- `goth_hyper/pipeline.py`: end-to-end assembly
