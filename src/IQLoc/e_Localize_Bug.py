from __future__ import annotations

from pathlib import Path
import sys

from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.CrossEncoderConfig import CROSS_ENCODER_PREFIX
from IR_Reretrieval import Searcher_RE
from Utils.IO import JSON_File_IO
from Utils.IR_Evaluation_Metrics.IR_Performance_Evaluator import Performance_Evaluator


DEFAULT_INPUT_FILE = PROJECT_ROOT / "Output" / "Generation" / f"{CROSS_ENCODER_PREFIX}_Cached_Source_Keywords.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "Localization"
DEFAULT_LOCALIZATION_OUTPUT_FILE = f"{CROSS_ENCODER_PREFIX}_Localized_Bugs.json"
DEFAULT_PERFORMANCE_OUTPUT_FILE = f"{CROSS_ENCODER_PREFIX}_Localization_Performance.json"
DEFAULT_TOP_K = 10
DEFAULT_AT_KS = [1, 5, 10]


def load_json(file_path: str) -> list[dict]:
    bug_reports = JSON_File_IO.load_JSON_to_Dict(file_path)
    if not isinstance(bug_reports, list):
        raise ValueError("Input JSON must be a list of bug report objects.")
    return bug_reports


def _extract_method_keywords_with_scores(bug_report: dict) -> list[tuple[float, str]]:
    keyword_entries: list[tuple[float, str]] = []
    for result in bug_report.get("Results", []):
        result_sim = result.get("src_kw_sim_15", 0.0)
        methods = result.get("methods")

        if isinstance(methods, list):
            for method in methods:
                if not isinstance(method, dict):
                    continue
                keyword_text = method.get("src_kw_15")
                if not isinstance(keyword_text, str) or not keyword_text.strip():
                    continue
                score = method.get("src_kw_sim_15", method.get("ce_score", result_sim))
                if not isinstance(score, (int, float)):
                    score = 0.0
                keyword_entries.append((float(score), keyword_text))

        elif isinstance(methods, dict):
            for method_value in methods.values():
                if not isinstance(method_value, dict):
                    continue
                keyword_text = method_value.get("src_kw_15")
                if not isinstance(keyword_text, str) or not keyword_text.strip():
                    continue
                score = method_value.get("src_kw_sim_15", method_value.get("ce_score", result_sim))
                if not isinstance(score, (int, float)):
                    score = 0.0
                keyword_entries.append((float(score), keyword_text))

        else:
            keyword_text = result.get("src_kw_15")
            if isinstance(keyword_text, str) and keyword_text.strip():
                score = result.get("src_kw_sim_15", result.get("ce_score", 0.0))
                if not isinstance(score, (int, float)):
                    score = 0.0
                keyword_entries.append((float(score), keyword_text))

    return keyword_entries


def mix_and_match_embeddings(keywords: str, bug_report: dict, shortlist_size: int = 5) -> str:
    keyword_entries = _extract_method_keywords_with_scores(bug_report)
    if not keyword_entries:
        return keywords

    keyword_entries = sorted(keyword_entries, key=lambda x: x[0], reverse=True)
    shortlisted_keywords = [entry[1] for entry in keyword_entries[:shortlist_size]]

    token_lists = [keyword_text.split() for keyword_text in shortlisted_keywords if keyword_text.strip()]
    if not token_lists:
        return keywords

    common_keywords = set(token_lists[0])
    for token_list in token_lists[1:]:
        common_keywords.intersection_update(token_list)

    if not common_keywords:
        return keywords

    high_match_kw = " ".join(sorted(common_keywords))
    return (keywords + " " + high_match_kw).strip()


if __name__ == "__main__":
    input_file_path = str(DEFAULT_INPUT_FILE)
    output_dir = str(DEFAULT_OUTPUT_DIR)

    bug_reports = load_json(input_file_path)
    searcher = Searcher_RE()

    ground_truths: list[list[str]] = []
    search_results: list[list[str]] = []
    localization_records: list[dict] = []

    for bug_report in tqdm(bug_reports, desc="Localizing bugs"):
        project = bug_report.get("project")
        sub_project = bug_report.get("sub_project")
        version = bug_report.get("version")
        bug_id = bug_report.get("bug_id")
        keywords = str(bug_report.get("keywords", "")).strip()

        fixed_files = bug_report.get("fixed_files", [])
        if not isinstance(fixed_files, list):
            fixed_files = [fixed_files] if fixed_files else []

        if not all([project, sub_project, version, bug_id]) or not keywords:
            continue

        extended_keywords = mix_and_match_embeddings(keywords, bug_report)

        results = searcher.search(
            bug_id=bug_id,
            project=project,
            sub_project=sub_project,
            version=version,
            query=extended_keywords,
            top_K_results=DEFAULT_TOP_K,
        )

        top_results = results[:DEFAULT_TOP_K]

        ground_truths.append(fixed_files)
        search_results.append(top_results)

        localization_records.append({
            "project": project,
            "sub_project": sub_project,
            "version": version,
            "bug_id": bug_id,
            "keywords": keywords,
            "extended_keywords": extended_keywords,
            "fixed_files": fixed_files,
            "localized_results": top_results,
        })

    performance_evaluator = Performance_Evaluator()
    performance = performance_evaluator.evaluate_several(
        ground_truths,
        search_results,
        at_Ks=DEFAULT_AT_KS,
    )

    JSON_File_IO.save_Dict_to_JSON(
        localization_records,
        file_path=output_dir,
        file_name=DEFAULT_LOCALIZATION_OUTPUT_FILE,
    )
    JSON_File_IO.save_Dict_to_JSON(
        performance,
        file_path=output_dir,
        file_name=DEFAULT_PERFORMANCE_OUTPUT_FILE,
    )

    print(performance)

