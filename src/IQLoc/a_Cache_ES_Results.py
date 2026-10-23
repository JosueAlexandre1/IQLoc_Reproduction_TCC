from __future__ import annotations

from pathlib import Path
import sys

from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from IR.Searcher.Searcher import Searcher
from Utils.IO import JSON_File_IO
from Utils.Parser.JavaSourceParser import JavaSourceParser

DEFAULT_INPUT_FILE = PROJECT_ROOT / "Dataset" / "Bench4BLExtended.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "Caching"
DEFAULT_OUTPUT_FILE_NAME = "Cached_ES_Results.json"
DEFAULT_TOP_K = 100
DEFAULT_INDEX_NAME = None


def _build_query(bug_report: dict) -> str:
	keywords = bug_report.get("keywords")
	if keywords is not None and str(keywords).strip() != "":
		return str(keywords)

	bug_title = str(bug_report.get("bug_title", "")).strip()
	bug_description = str(bug_report.get("bug_description", "")).strip()
	return (bug_title + " . " + bug_description).strip(" .")


def _extract_methods(source_code: str) -> list[dict]:
	if not isinstance(source_code, str) or source_code.strip() == "":
		return []

	try:
		parser = JavaSourceParser(source_code, clear_formatting=True)
		methods_dict = parser.parse_methods()
	except Exception:
		return []

	methods = []
	for method_name, method_body in methods_dict.items():
		if not isinstance(method_body, str) or method_body.strip() == "":
			continue
		methods.append({
			"method_name": method_name,
			"method_body": method_body,
		})

	return methods


def _normalize_results(search_results: list[dict]) -> list[dict]:
	normalized_results = []

	for result in search_results:
		file_url = result.get("file_url")
		source_code = result.get("source_code")
		bm25_score = result.get("bm25_score")

		normalized_results.append({
			"file_url": file_url,
			"source_code": source_code,
			"bm25_score": bm25_score,
			"methods": _extract_methods(source_code),
		})

	return normalized_results


def cache_es_results(
		input_file_path: str,
		output_dir: str,
		output_file_name: str,
		top_k_results: int = DEFAULT_TOP_K,
		index_name: str | None = None,
) -> list[dict]:
	bug_reports = JSON_File_IO.load_JSON_to_Dict(input_file_path)

	if not isinstance(bug_reports, list):
		raise ValueError("Input JSON must be a list of bug report objects.")

	searcher = Searcher(index_name=index_name)
	cached_reports = []

	for bug_report in tqdm(bug_reports, total=len(bug_reports), desc="Caching ES results"):
		project = bug_report.get("project")
		sub_project = bug_report.get("sub_project")
		version = bug_report.get("version")

		if project is None or sub_project is None or version is None:
			cached_report = dict(bug_report)
			cached_report["Results"] = []
			cached_report["retrieval_query"] = _build_query(bug_report)
			cached_report["retrieval_top_k"] = top_k_results
			cached_reports.append(cached_report)
			continue

		query = _build_query(bug_report)
		search_results = searcher.search_Extended(
			project=project,
			sub_project=sub_project,
			version=version,
			query=query,
			top_K_results=top_k_results,
			field_to_return=["file_url", "source_code"],
		)

		cached_report = dict(bug_report)
		cached_report["Results"] = _normalize_results(search_results)
		cached_report["retrieval_query"] = query
		cached_report["retrieval_top_k"] = top_k_results
		cached_reports.append(cached_report)

	JSON_File_IO.save_Dict_to_JSON(cached_reports, file_path=output_dir, file_name=output_file_name)
	return cached_reports


if __name__ == "__main__":
	input_file_path = str(DEFAULT_INPUT_FILE)
	output_dir = str(DEFAULT_OUTPUT_DIR)
	output_file_name = DEFAULT_OUTPUT_FILE_NAME
	top_k_results = DEFAULT_TOP_K
	index_name = DEFAULT_INDEX_NAME

	cache_es_results(
		input_file_path=input_file_path,
		output_dir=output_dir,
		output_file_name=output_file_name,
		top_k_results=top_k_results,
		index_name=index_name,
	)
	print(
		f"Cached ES results to {Path(output_dir) / output_file_name} "
		f"(top_k={top_k_results}, index={index_name or 'default'})"
	)
