from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

#项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#加入paython的搜索路径
sys.path[:0] = [str(PROJECT_ROOT / "depo"), str(PROJECT_ROOT)]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from hyper_branch.pipeline import HyperBranchPipeline
from hyper_branch.client import OpenAIClient
from eval.eval import cal_em, cal_f1


def _gold_answers(question: dict[str, Any]) -> list[str]:
    for key in ("golden_answers", "answers", "answer"):
        value = question.get(key)
        if isinstance(value, list):
            return [str(answer).strip() for answer in value if str(answer).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []


def _result_answer(result_file: Path) -> str:
    if not result_file.exists():
        return ""
    nodes = json.loads(result_file.read_text(encoding="utf-8")).get("nodes", [])
    if not nodes or not isinstance(nodes[-1], dict):
        return ""
    return str(nodes[-1].get("answer", "") or "").strip()


def save_scores(
    dataset: str,
    run_id: str,
    output_dir: Path,
    indexed_questions: list[tuple[int, dict[str, Any]]],
    question_file: Path | None = None,
    question_structure_override: list[list[str]] | None = None,
) -> Path:
    records = []
    for index, question in indexed_questions:
        gold_answers = _gold_answers(question)
        answer = _result_answer(output_dir / f"{index:05d}" / "result.json")
        records.append(
            {
                "index": index,
                "question": question["question"],
                "golden_answers": gold_answers,
                "answer": answer,
                "em": float(cal_em([gold_answers], [answer])) if gold_answers else 0.0,
                "f1": float(cal_f1([gold_answers], [answer])) if gold_answers else 0.0,
            }
        )

    score_dir = PROJECT_ROOT / "eval" / "results" / "depo_hyperbranch" / dataset / run_id
    score_dir.mkdir(parents=True, exist_ok=True)
    completed = sum(
        (output_dir / f"{index:05d}" / "result.json").exists()
        for index, _ in indexed_questions
    )
    resolved_question_file = question_file or (
        PROJECT_ROOT / "questions" / dataset / "questions.json"
    )
    score = {
        "meta": {
            "question_file": str(resolved_question_file.resolve()),
            "runs_dir": str(output_dir.resolve()),
            "indices": [index for index, _ in indexed_questions],
            "metrics": ["em", "f1"],
            "question_structure_override": question_structure_override,
        },
        "counts": {"total": len(records), "completed": completed, "missing": len(records) - completed},
        "overall": {
            "em": sum(record["em"] for record in records) / len(records) if records else 0.0,
            "f1": sum(record["f1"] for record in records) / len(records) if records else 0.0,
        },
    }
    (score_dir / "test_result.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (score_dir / "test_score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return score_dir / "test_score.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DEPO and HyperBranch for one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--question-file",
        help="Question JSON file. Defaults to questions/<dataset>/questions.json.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument(
        "--empty-question-structure",
        action="store_true",
        help=(
            "Pass an empty question_structure to atomic-question DAG generation "
            "while preserving all other DEPO stages."
        ),
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ["OPENAI_API_KEY"]
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    #读取指定数据集的问题文件，然后根据参数截取运行的问题数量
    question_file = (
        Path(args.question_file)
        if args.question_file
        else PROJECT_ROOT / "questions" / args.dataset / "questions.json"
    )
    if not question_file.is_file():
        parser.error(f"question file not found: {question_file}")
    question_file = question_file.resolve()
    questions = json.loads(question_file.read_text(encoding="utf-8"))[args.start - 1 : args.end]
    #需要运行的题目数量
    if args.limit is not None:
        questions = questions[: args.limit]
    indexed_questions = [
        (args.start + offset, question) for offset, question in enumerate(questions)
    ]
    question_structure_override = [] if args.empty_question_structure else None

    from hanlp_sdp_parser import HanLPSDPParser
    from pipeline import run_depo
    #创建llm客户端
    #读取yaml配置文件，转成字典格式
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8")
    )
    llm = OpenAIClient(
        api_key=api_key,
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=base_url,
    )

    #创建一个检索器对象，并完成初始化
    hyperbranch = HyperBranchPipeline(
        #超图数据库路径
        PROJECT_ROOT / config["dataset_root"],
        model=args.llm_model,
        #嵌入模型
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        api_key=api_key,
        base_url=base_url,
        client=llm,
    )
    #创建解析器对象
    sdp_parser = HanLPSDPParser()
    #输出目录路径
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / run_id
    #读取问题列表
    for index, item in indexed_questions:
        #计算真实问题编号
        #获取问题文本，
        question = item["question"].strip()
        #输出文件位置
        result_file = output_dir / f"{index:05d}" / "result.json"
        #判断是否运行过，断点续跑
        if args.resume and result_file.exists():
            continue
        try:
            decomposition = run_depo(
                question,
                sdp_parser,
                llm,
                question_structure_override=question_structure_override,
            )
            result = hyperbranch.run(
                question,
                decomposition["atomic_question_dag"],
                decomposition["entities"],
            )
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(
                    {
                        "topic_entities": decomposition["entities"],
                        "atomic_question_dag": decomposition["atomic_question_dag"],
                        "topic_entity_ids": result["topic_entity_ids"],
                        "nodes": [
                            {
                                "id": node["node_id"],
                                "rewritten_question": node["question"],
                                "entities": node["entities"],
                                "entity_ids": node["entity_ids"],
                                "evidence_blocks": node["evidence_blocks"],
                                "answer": node["answer"],
                            }
                            for node in result["atomic_answers"]
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{args.dataset} #{index}: {result['final_answer']['answer']}")
            completed_questions = [
                (completed_index, completed_item)
                for completed_index, completed_item in indexed_questions
                if (output_dir / f"{completed_index:05d}" / "result.json").exists()
            ]
            if len(completed_questions) % 10 == 0:
                score_file = save_scores(
                    args.dataset,
                    run_id,
                    output_dir,
                    completed_questions,
                    question_file,
                    question_structure_override,
                )
                score = json.loads(score_file.read_text(encoding="utf-8"))
                print(
                    f"checkpoint={len(completed_questions)}/{len(indexed_questions)} "
                    f"EM={score['overall']['em']:.4f} "
                    f"F1={score['overall']['f1']:.4f}"
                )
        except Exception as exc:
            print(f"{args.dataset} #{index} failed: {exc}", file=sys.stderr)

    score_file = save_scores(
        args.dataset,
        run_id,
        output_dir,
        indexed_questions,
        question_file,
        question_structure_override,
    )
    score = json.loads(score_file.read_text(encoding="utf-8"))
    print(f"saved_scores={score_file}")
    print(f"EM={score['overall']['em']:.4f}")
    print(f"F1={score['overall']['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
