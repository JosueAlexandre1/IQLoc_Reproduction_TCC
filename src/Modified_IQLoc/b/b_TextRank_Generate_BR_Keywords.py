from __future__ import annotations

from pathlib import Path
import sys

from summa.keywords import keywords as textrank_keywords
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
OUTPUT_FILE_NAME = f"{CROSS_ENCODER_PREFIX}_TextRank_Cached_BR_Keywords.json"

DEFAULT_N_GRAM_LEFT = 1
DEFAULT_N_GRAM_RIGHT = 1
DEFAULT_STOP_WORDS = "english"
DEFAULT_DIVERSITY = 0.5
DEFAULT_TOP_N = 15

# ============================================================
# Alternativa experimental: TextRank
# ============================================================

def generate_keywords_for_reports(bug_reports: list[dict]) -> list[dict]:
    generated_reports = []

    for bug_report in tqdm(
        bug_reports,
        total=len(bug_reports),
        desc="Generating BR keywords with TextRank",
    ):
        bug_text = _build_bug_report_text(bug_report)

        extracted_keywords = textrank_keywords(
            bug_text,
            ratio=1.0,
            words=None,
            split=True,
            scores=False,
            language="english",
            additional_stopwords=None,
        )

        # Manter apenas palavras-chave de uma palavra e limitar a 15.
        keywords = [
            keyword.strip()
            for keyword in extracted_keywords
            if keyword.strip()
            and len(keyword.strip().split()) == 1
        ][:DEFAULT_TOP_N]

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
    # 1. Carregar os relatórios em cache.
    cached_bug_reports = JSON_File_IO.load_JSON_to_Dict(str(INPUT_FILE))

    if not isinstance(cached_bug_reports, list):
        raise ValueError("Input JSON must be a list of bug report objects.")

    # 2. Gerar palavras-chave com TextRank.
    generated_bug_reports = generate_keywords_for_reports(cached_bug_reports)

    # 3. Salvar os resultados.
    JSON_File_IO.save_Dict_to_JSON(
        generated_bug_reports,
        file_path=str(OUTPUT_DIR),
        file_name=OUTPUT_FILE_NAME,
    )

    print(f"Saved BR keywords to {OUTPUT_DIR / OUTPUT_FILE_NAME}")


if __name__ == "__main__":
    main()