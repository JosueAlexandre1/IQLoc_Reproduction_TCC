from __future__ import annotations

import json
from pathlib import Path
import sys

import ijson
from flair.embeddings import TransformerDocumentEmbeddings
from keybert import KeyBERT
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.CrossEncoderConfig import CROSS_ENCODER_PREFIX


DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "Output"
    / "Caching"
    / f"{CROSS_ENCODER_PREFIX}_Cached_CE_Scores.json"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "Generation"

DEFAULT_OUTPUT_FILE_NAME = (
    f"{CROSS_ENCODER_PREFIX}_Cached_Source_Keywords.json"
)

DEFAULT_MODEL_NAME = "Salesforce/codet5-small"
DEFAULT_SOURCE_KEYWORD_LENGTH = 15
DEFAULT_CE_SCORE_THRESHOLD = 0.5


def find_keywords(
    query: str,
    keyword_length: int,
    keyword_model: KeyBERT,
) -> list[tuple[str, float]]:
    return keyword_model.extract_keywords(
        query,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        top_n=keyword_length,
        use_mmr=True,
        diversity=0.5,
    )


def process_method(
    method: dict,
    keyword_model: KeyBERT,
) -> None:
    method_body = method.get("method_body")
    ce_score = method.get("ce_score")

    if (
        not isinstance(method_body, str)
        or not isinstance(ce_score, (float, int))
    ):
        return

    if ce_score < DEFAULT_CE_SCORE_THRESHOLD:
        return

    source_keywords = find_keywords(
        method_body,
        DEFAULT_SOURCE_KEYWORD_LENGTH,
        keyword_model,
    )

    method[f"src_kw_{DEFAULT_SOURCE_KEYWORD_LENGTH}"] = " ".join(
        key for key, _ in source_keywords
    )

    method[f"src_kw_sim_{DEFAULT_SOURCE_KEYWORD_LENGTH}"] = float(
        ce_score
    )


def process_bug_report(
    bug_report: dict,
    keyword_model: KeyBERT,
) -> dict:
    es_results = bug_report.get("Results", [])

    if not isinstance(es_results, list):
        return bug_report

    for result in es_results:
        if not isinstance(result, dict):
            continue

        methods = result.get("methods")

        if isinstance(methods, list):
            for method in methods:
                if isinstance(method, dict):
                    process_method(method, keyword_model)

        elif isinstance(methods, dict):
            for method_value in methods.values():
                if isinstance(method_value, dict):
                    process_method(method_value, keyword_model)

    return bug_report


def gather_src_keywords(
    input_file: str | Path,
    output_file: str | Path,
    keyword_model: KeyBERT,
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "rb") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        outfile.write("[\n")
        first = True

        for bug_report in tqdm(
            ijson.items(infile, "item", use_float=True),
            desc="Generating source keywords",
        ):
            if not isinstance(bug_report, dict):
                raise ValueError(
                    "Input JSON must contain a list of bug report objects."
                )

            process_bug_report(bug_report, keyword_model)

            if not first:
                outfile.write(",\n")

            json.dump(
                bug_report,
                outfile,
                ensure_ascii=False,
            )

            outfile.flush()
            first = False

        outfile.write("\n]\n")
        outfile.flush()


if __name__ == "__main__":
    input_file_path = DEFAULT_INPUT_FILE
    output_file_path = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILE_NAME

    if not input_file_path.is_file():
        raise FileNotFoundError(
            f"Input JSON not found: {input_file_path}"
        )

    model = TransformerDocumentEmbeddings(DEFAULT_MODEL_NAME)
    keyword_model = KeyBERT(model=model)

    gather_src_keywords(
        input_file=input_file_path,
        output_file=output_file_path,
        keyword_model=keyword_model,
    )

    print(f"Saved source keywords to {output_file_path}")