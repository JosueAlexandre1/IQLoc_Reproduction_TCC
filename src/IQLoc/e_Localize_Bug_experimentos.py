from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ijson
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from IR_Reretrieval import Searcher_RE
from Utils.IR_Evaluation_Metrics.IR_Performance_Evaluator import Performance_Evaluator

DEFAULT_TOP_K = 10
DEFAULT_AT_KS = [1, 5, 10]


def report_key(report: dict) -> str:
    return json.dumps([str(report.get(field, "")) for field in (
        "project", "sub_project", "version", "bug_id"
    )], ensure_ascii=False, separators=(",", ":"))


def load_keyword_index(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Mapa B inválido: {path}")
    return data


def _extract_method_keywords_with_scores(bug_report: dict) -> list[tuple[float, str]]:
    keyword_entries = []
    for result in bug_report.get("Results", []):
        if not isinstance(result, dict):
            continue
        result_sim = result.get("src_kw_sim_15", 0.0)
        methods = result.get("methods")
        if isinstance(methods, list):
            values = methods
        elif isinstance(methods, dict):
            values = methods.values()
        else:
            values = [result]
        for method in values:
            if not isinstance(method, dict):
                continue
            keyword_text = method.get("src_kw_15")
            if not isinstance(keyword_text, str) or not keyword_text.strip():
                continue
            score = method.get("src_kw_sim_15", method.get("ce_score", result_sim))
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                score = 0.0
            keyword_entries.append((float(score), keyword_text))
    return keyword_entries


def mix_and_match_embeddings(keywords: str, bug_report: dict, shortlist_size: int = 5) -> str:
    entries = _extract_method_keywords_with_scores(bug_report)
    if not entries:
        return keywords
    entries.sort(key=lambda entry: entry[0], reverse=True)
    token_lists = [entry[1].split() for entry in entries[:shortlist_size] if entry[1].strip()]
    if not token_lists:
        return keywords
    common_keywords = set(token_lists[0])
    for tokens in token_lists[1:]:
        common_keywords.intersection_update(tokens)
    if not common_keywords:
        return keywords
    return (keywords + " " + " ".join(sorted(common_keywords))).strip()


def run(input_path: Path, keyword_map_path: Path, localization_out: Path,
        metrics_out: Path, shortlist_size: int) -> None:
    keyword_map = load_keyword_index(keyword_map_path)
    searcher = Searcher_RE()
    evaluator = Performance_Evaluator()
    ground_truths = []
    search_results = []
    missing = 0
    processed = 0
    expanded = 0
    localization_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    partial = localization_out.with_suffix(localization_out.suffix + ".partial")
    try:
        with input_path.open("rb") as infile, partial.open("w", encoding="utf-8") as outfile:
            outfile.write("[\n")
            first = True
            for report in tqdm(ijson.items(infile, "item", use_float=True), desc="Localizing bugs"):
                if not isinstance(report, dict):
                    raise ValueError("Entrada D não contém relatórios JSON válidos")
                project = report.get("project")
                sub_project = report.get("sub_project")
                version = report.get("version")
                bug_id = report.get("bug_id")
                if not all([project, sub_project, version, bug_id]):
                    continue
                key = report_key(report)
                if key not in keyword_map:
                    missing += 1
                    # Não reutilizar silenciosamente keywords de outro extrator.
                    raise ValueError(f"Bug ausente no mapa de palavras-chave da etapa B: {key}")
                keywords = keyword_map[key].strip()
                if not keywords:
                    continue
                fixed_files = report.get("fixed_files", [])
                if not isinstance(fixed_files, list):
                    fixed_files = [fixed_files] if fixed_files else []
                extended = mix_and_match_embeddings(keywords, report, shortlist_size)
                expanded += (extended != keywords)
                results = searcher.search(
                    bug_id=bug_id, project=project, sub_project=sub_project,
                    version=version, query=extended, top_K_results=DEFAULT_TOP_K,
                )
                top_results = results[:DEFAULT_TOP_K]
                ground_truths.append(fixed_files)
                search_results.append(top_results)
                record = {
                    "project": project, "sub_project": sub_project,
                    "version": version, "bug_id": bug_id,
                    "keywords": keywords, "extended_keywords": extended,
                    "fixed_files": fixed_files, "localized_results": top_results,
                }
                if not first:
                    outfile.write(",\n")
                json.dump(record, outfile, ensure_ascii=False)
                first = False
                processed += 1
            outfile.write("\n]\n")
        partial.replace(localization_out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if processed == 0:
        raise ValueError("Nenhum relatório foi avaliado; verifique a etapa B e os identificadores")
    performance = evaluator.evaluate_several(ground_truths, search_results, at_Ks=DEFAULT_AT_KS)
    tmp = metrics_out.with_suffix(metrics_out.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(performance, file, ensure_ascii=False, indent=2)
    tmp.replace(metrics_out)
    print(f"Localizados: {processed}; consultas expandidas: {expanded}; faltantes B: {missing}")
    print(json.dumps(performance, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Etapa E com cache B independente de C")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--keyword-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--shortlist-size", type=int, default=5)
    args = parser.parse_args()
    if args.shortlist_size <= 0:
        parser.error("--shortlist-size precisa ser positivo")
    for path in (args.input, args.keyword_map):
        if not path.is_file():
            parser.error(f"Arquivo não encontrado: {path}")
    run(args.input, args.keyword_map, args.output, args.metrics, args.shortlist_size)


if __name__ == "__main__":
    main()
