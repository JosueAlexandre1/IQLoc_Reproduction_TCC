from __future__ import annotations

from pathlib import Path
import sys

from rake_nltk import Rake
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
PROJECT_ROOT = SRC_DIR.parent

for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from Utils.IO import JSON_File_IO
from scripts.CrossEncoderConfig import CROSS_ENCODER_PREFIX

from IQLoc.b_Generate_BR_Keywords import (
    _build_bug_report_text,
)
# ============================================================
# Configuração
# ============================================================

INPUT_FILE = PROJECT_ROOT / "Output" / "Caching" / f"{CROSS_ENCODER_PREFIX}_Cached_ES_Results.json"
OUTPUT_DIR = PROJECT_ROOT / "Output" / "Generation"/ "Modified"/ "Bug Reports"
OUTPUT_FILE_NAME = f"{CROSS_ENCODER_PREFIX}_Rake_Cached_BR_Keywords.json"

KEYWORD_MODEL_NAME = "Salesforce/codet5-small"


DEFAULT_N_GRAM_LEFT = 1
DEFAULT_N_GRAM_RIGHT = 1
DEFAULT_STOP_WORDS = "english"
DEFAULT_DIVERSITY = 0.5
DEFAULT_TOP_N = 15

# ============================================================
# Alternativa experimental: extrator
# ============================================================

def generate_keywords_for_reports(bug_reports: list[dict], kw_model: Rake) -> list[dict]:
    generated_reports = []

    for bug_report in tqdm(
    bug_reports,
    total=len(bug_reports),
    desc="Generating BR keywords with RAKE",
    ):
        bug_text = _build_bug_report_text(bug_report)

        kw_model.extract_keywords_from_text(bug_text)
        keywords = kw_model.get_ranked_phrases()[:DEFAULT_TOP_N]
        keywords_joined = " ".join(keywords)

        output_bug_report = dict(bug_report)
        output_bug_report["keywords"] = keywords_joined
        output_bug_report["n_gram_left"] = DEFAULT_N_GRAM_LEFT
        output_bug_report["n_gram_right"] = DEFAULT_N_GRAM_RIGHT
        output_bug_report["stop_words"] = DEFAULT_STOP_WORDS
        output_bug_report["nr_candidates"] = DEFAULT_DIVERSITY

        generated_reports.append(output_bug_report)

    return generated_reports


def main():
    input_file_path = str(INPUT_FILE)
    output_dir = str(OUTPUT_DIR)
    output_file_name = OUTPUT_FILE_NAME

    keyword_model = Rake(
        language="english",
        max_length=1,
    )
    # 1. Carregar os relatórios em cache.
    cached_bug_reports = JSON_File_IO.load_JSON_to_Dict(str(input_file_path))

    if not isinstance(cached_bug_reports, list):
        raise ValueError("Input JSON must be a list of bug report objects.")

    generated_bug_reports = generate_keywords_for_reports(cached_bug_reports, keyword_model)

    # 5. Salvar os resultados.
    JSON_File_IO.save_Dict_to_JSON(
        generated_bug_reports,
        file_path=output_dir,
        file_name=output_file_name,
    )

    print(f"Saved BR keywords to {OUTPUT_DIR / OUTPUT_FILE_NAME}")


if __name__ == "__main__":
    main()