from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

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

DEFAULT_MODEL_NAME = "Salesforce/codet5-small"
DEFAULT_SOURCE_KEYWORD_LENGTH = 15


def find_keywords(query: str, keyword_length: int, keyword_model: KeyBERT):
    return keyword_model.extract_keywords(
        query,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        top_n=keyword_length,
        use_mmr=True,
        diversity=0.5,
    )


def iter_methods(bug_report: dict):
    """Itera pelos métodos de todos os arquivos candidatos de UM bug."""
    es_results = bug_report.get("Results", [])
    if not isinstance(es_results, list):
        return

    for result in es_results:
        if not isinstance(result, dict):
            continue
        methods = result.get("methods")
        if isinstance(methods, list):
            values = methods
        elif isinstance(methods, dict):
            values = methods.values()
        else:
            continue
        for method in values:
            if isinstance(method, dict):
                yield method


def eligible_methods(bug_report: dict):
    """Limpa campos anteriores e retorna métodos com corpo e score válidos."""
    for method in iter_methods(bug_report):
        method.pop(f"src_kw_{DEFAULT_SOURCE_KEYWORD_LENGTH}", None)
        method.pop(f"src_kw_sim_{DEFAULT_SOURCE_KEYWORD_LENGTH}", None)

        body = method.get("method_body")
        score = method.get("ce_score")
        if not isinstance(body, str) or not body.strip():
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        if not math.isfinite(score):
            continue
        yield method


def process_method(method: dict, keyword_model: KeyBERT) -> None:
    """Extrai termos somente de um método já selecionado."""
    source_keywords = find_keywords(
        method["method_body"], DEFAULT_SOURCE_KEYWORD_LENGTH, keyword_model
    )
    method[f"src_kw_{DEFAULT_SOURCE_KEYWORD_LENGTH}"] = " ".join(
        word for word, _ in source_keywords
    )
    method[f"src_kw_sim_{DEFAULT_SOURCE_KEYWORD_LENGTH}"] = float(
        method["ce_score"]
    )


def process_bug_report(
    bug_report: dict,
    keyword_model: KeyBERT,
    mode: str = "top_k",
    top_k: int = 5,
    threshold: float = 0.5,
) -> dict:
    if mode == "threshold":
        for method in eligible_methods(bug_report):
            if float(method["ce_score"]) >= threshold:
                process_method(method, keyword_model)
    elif mode == "top_k":
        # heapq.nlargest usa memória O(k), em vez de ordenar todos os métodos.
        # A ordem original dos candidatos desempata scores idênticos.
        selected = heapq.nlargest(
            top_k,
            enumerate(eligible_methods(bug_report)),
            key=lambda item: (float(item[1]["ce_score"]), -item[0]),
        )
        for _, method in selected:
            process_method(method, keyword_model)
    else:
        raise ValueError(f"Modo desconhecido: {mode}")

    return bug_report


def gather_src_keywords(
    input_file: str | Path,
    output_file: str | Path,
    keyword_model: KeyBERT,
    mode: str,
    top_k: int,
    threshold: float,
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise FileExistsError(
            f"Saída já existe: {output_path}. Escolha outro --output para não sobrescrever."
        )

    # Só publica o JSON final quando terminou. O .partial NÃO é retomável.
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    with open(input_file, "rb") as infile, open(
        partial_path, "w", encoding="utf-8"
    ) as outfile:
        outfile.write("[\n")
        first = True
        for bug_report in tqdm(
            ijson.items(infile, "item", use_float=True),
            desc=f"Generating source keywords ({mode})",
        ):
            if not isinstance(bug_report, dict):
                raise ValueError("Input JSON must contain bug report objects.")
            process_bug_report(bug_report, keyword_model, mode, top_k, threshold)
            if not first:
                outfile.write(",\n")
            json.dump(bug_report, outfile, ensure_ascii=False)
            outfile.flush()
            first = False
        outfile.write("\n]\n")
        outfile.flush()

    partial_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Etapa D do IQLoc com seleção por threshold ou top-k por bug"
    )
    parser.add_argument("--prefix", help="Prefixo do cache do Cross-Encoder")
    parser.add_argument("--input", type=Path, help="JSON gerado pela etapa C")
    parser.add_argument("--output", type=Path, help="JSON de saída da etapa D")
    parser.add_argument("--mode", choices=["threshold", "top_k"], default="top_k")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k deve ser maior ou igual a 1")
    if not math.isfinite(args.threshold):
        parser.error("--threshold deve ser finito")
    if not args.prefix and (args.input is None or args.output is None):
        parser.error("Informe --prefix ou ambos --input e --output")

    input_path = args.input or (
        PROJECT_ROOT / "Output" / "Caching" /
        f"{args.prefix}_Cached_CE_Scores.json"
    )
    if args.mode == "top_k":
        variant = f"top_{args.top_k}"
    else:
        variant = "threshold_" + str(args.threshold).replace("-", "neg_").replace(".", "p")

    output_path = args.output or (
        PROJECT_ROOT / "Output" / "Generation" / "Experiments" /
        args.prefix / variant / "Cached_Source_Keywords.json"
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Cache de C não encontrado: {input_path}")

    print(f"Entrada C: {input_path}")
    print(f"Saída D: {output_path}")
    print(f"Seleção: {args.mode} (top_k={args.top_k}, threshold={args.threshold})")

    model = TransformerDocumentEmbeddings(DEFAULT_MODEL_NAME)
    keyword_model = KeyBERT(model=model)
    gather_src_keywords(
        input_file=input_path,
        output_file=output_path,
        keyword_model=keyword_model,
        mode=args.mode,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    print(f"Saved source keywords to {output_path}")


if __name__ == "__main__":
    main()
