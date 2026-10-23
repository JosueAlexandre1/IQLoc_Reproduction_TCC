from __future__ import annotations

from pathlib import Path
import sys

from flair.embeddings import TransformerDocumentEmbeddings
from keybert import KeyBERT
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from Utils.IO import JSON_File_IO
from scripts.CrossEncoderConfig import CROSS_ENCODER_PREFIX

DEFAULT_INPUT_FILE = PROJECT_ROOT / "Output" / "Caching" / f"{CROSS_ENCODER_PREFIX}_Cached_CE_Scores.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "Generation"
DEFAULT_OUTPUT_FILE_NAME = f"{CROSS_ENCODER_PREFIX}_Cached_Source_Keywords.json"

DEFAULT_MODEL_NAME = "Salesforce/codet5-small"
DEFAULT_SOURCE_KEYWORD_LENGTH = 15
DEFAULT_CE_SCORE_THRESHOLD = 0.5


def find_keywords(query: str, keyword_length: int, keyword_model: KeyBERT) -> list[tuple[str, float]]:
    keywords = keyword_model.extract_keywords(
        query,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        top_n=keyword_length,
        use_mmr=True,
        diversity=0.5,
    )
    return keywords


def gather_src_keywords(bug_reports: list[dict], keyword_model: KeyBERT) -> list[dict]:
    for bug_report in tqdm(bug_reports, desc="Generating source keywords"):
        es_results = bug_report.get("Results", [])
        if not isinstance(es_results, list):
            continue

        for result in es_results:
            methods = result.get("methods")

            if isinstance(methods, list):
                for method in methods:
                    if not isinstance(method, dict):
                        continue

                    method_body = method.get("method_body")
                    ce_score = method.get("ce_score")

                    if not isinstance(method_body, str) or not isinstance(ce_score, (float, int)):
                        continue
                    if ce_score < DEFAULT_CE_SCORE_THRESHOLD:
                        continue

                    source_keywords = find_keywords(method_body, DEFAULT_SOURCE_KEYWORD_LENGTH, keyword_model)
                    method[f"src_kw_{DEFAULT_SOURCE_KEYWORD_LENGTH}"] = " ".join(
                        key for key, _ in source_keywords
                    )
                    method[f"src_kw_sim_{DEFAULT_SOURCE_KEYWORD_LENGTH}"] = float(ce_score)

            elif isinstance(methods, dict):
                for _, method_value in methods.items():
                    if not isinstance(method_value, dict):
                        continue

                    method_body = method_value.get("method_body")
                    ce_score = method_value.get("ce_score")

                    if not isinstance(method_body, str) or not isinstance(ce_score, (float, int)):
                        continue
                    if ce_score < DEFAULT_CE_SCORE_THRESHOLD:
                        continue

                    source_keywords = find_keywords(method_body, DEFAULT_SOURCE_KEYWORD_LENGTH, keyword_model)
                    method_value[f"src_kw_{DEFAULT_SOURCE_KEYWORD_LENGTH}"] = " ".join(
                        key for key, _ in source_keywords
                    )
                    method_value[f"src_kw_sim_{DEFAULT_SOURCE_KEYWORD_LENGTH}"] = float(ce_score)

    return bug_reports


if __name__ == "__main__":
    input_file_path = str(DEFAULT_INPUT_FILE)
    output_dir = str(DEFAULT_OUTPUT_DIR)
    output_file_name = DEFAULT_OUTPUT_FILE_NAME

    model = TransformerDocumentEmbeddings(DEFAULT_MODEL_NAME)
    keyword_model = KeyBERT(model=model)

    bug_reports = JSON_File_IO.load_JSON_to_Dict(input_file_path)
    if not isinstance(bug_reports, list):
        raise ValueError("Input JSON must be a list of bug report objects.")

    generated_reports = gather_src_keywords(bug_reports, keyword_model)

    JSON_File_IO.save_Dict_to_JSON(
        generated_reports,
        file_path=output_dir,
        file_name=output_file_name,
    )
    print(f"Saved source keywords to {Path(output_dir) / output_file_name}")