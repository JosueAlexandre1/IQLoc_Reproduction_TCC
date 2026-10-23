#!/usr/bin/env python3

import json
import sys
from pathlib import Path

from decimal import Decimal
import ijson


# merge_cache.py está dentro de:
# iqloc_reproduction/scripts/
#
# Então:
# parent       = scripts/
# parent.parent = iqloc_reproduction/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CACHE = (
    PROJECT_ROOT
    / "Output"
    / "Caching"
    / "Cached_ES_Results.json"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "Output"
    / "Caching"
    / "Cached_ES_Results_repair_checkpoint.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Output"
    / "Caching"
    / "Cached_ES_Results_repaired.json"
)

EXPECTED_TOTAL = 9459
EXPECTED_REPAIRED = 4799
EXPECTED_UNCHANGED = 4660


def get_key(record):
    return (
        record.get("sub_project"),
        record.get("project"),
        str(record.get("bug_id")),
    )

def json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )

def fail(message):
    print(f"\nERRO: {message}")
    sys.exit(1)


def main():

    print("=== MERGE DO CACHE ===")
    print(f"Projeto:     {PROJECT_ROOT}")
    print(f"Original:    {INPUT_CACHE}")
    print(f"Checkpoint:  {CHECKPOINT_FILE}")
    print(f"Saída:       {OUTPUT_FILE}")
    print()

    if not INPUT_CACHE.exists():
        fail(f"Cache original não encontrado:\n{INPUT_CACHE}")

    if not CHECKPOINT_FILE.exists():
        fail(f"Checkpoint não encontrado:\n{CHECKPOINT_FILE}")

    # Não sobrescreve um resultado existente por acidente.
    if OUTPUT_FILE.exists():
        fail(
            f"O arquivo de saída já existe:\n{OUTPUT_FILE}\n\n"
            "Remova-o manualmente se quiser gerar novamente."
        )

    original_count = 0
    repaired_count = 0
    unchanged_count = 0
    checkpoint_count = 0

    with open(
        INPUT_CACHE,
        "rb",
        buffering=1024 * 1024
    ) as original, open(
        CHECKPOINT_FILE,
        "r",
        encoding="utf-8",
        buffering=1024 * 1024
    ) as checkpoint, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024
    ) as output:

        # O checkpoint é consumido UMA LINHA POR VEZ.
        checkpoint_iter = iter(checkpoint)

        output.write("[\n")

        first = True

        # ijson lê o JSON original incrementalmente.
        # NÃO carrega os 13+ GB na memória.
        for bug_report in ijson.items(original, "item"):

            original_count += 1

            version = bug_report.get("version", "")

            # -----------------------------------------
            # Registro que já estava correto
            # -----------------------------------------
            if version != "":

                final_record = bug_report
                unchanged_count += 1

            # -----------------------------------------
            # Registro que foi reparado
            # -----------------------------------------
            else:

                checkpoint_line = next(
                    checkpoint_iter,
                    None
                )

                if checkpoint_line is None:
                    fail(
                        "O cache original possui mais registros "
                        "sem versão do que o checkpoint."
                    )

                try:
                    repaired_record = json.loads(
                        checkpoint_line
                    )
                except json.JSONDecodeError as e:
                    fail(
                        f"JSON inválido no checkpoint "
                        f"na linha {checkpoint_count + 1}: {e}"
                    )

                checkpoint_count += 1

                original_key = get_key(bug_report)
                repaired_key = get_key(repaired_record)

                # Segurança: garante que o checkpoint corresponde
                # exatamente ao bug que estamos substituindo.
                if original_key != repaired_key:
                    fail(
                        "MISMATCH ENTRE CACHE E CHECKPOINT\n\n"
                        f"Original:   {original_key}\n"
                        f"Checkpoint: {repaired_key}\n"
                        f"Registro original: {original_count}\n"
                        f"Linha checkpoint:   {checkpoint_count}"
                    )

                repaired_version = repaired_record.get(
                    "version",
                    ""
                )

                if repaired_version == "":
                    fail(
                        f"O registro reparado continua sem versão:\n"
                        f"{repaired_key}"
                    )

                final_record = repaired_record
                repaired_count += 1

            # -----------------------------------------
            # Escreve imediatamente no arquivo final
            # -----------------------------------------

            if not first:
                output.write(",\n")

            json.dump(
                final_record,
                output,
                ensure_ascii=False,
                separators=(",", ":"),
                default=json_default
            )

            first = False

            if original_count % 100 == 0:

                output.flush()

                print(
                    f"Processados: {original_count}/{EXPECTED_TOTAL} | "
                    f"reparados: {repaired_count} | "
                    f"mantidos: {unchanged_count}",
                    flush=True
                )

        output.write("\n]\n")
        output.flush()

        # Se ainda existe alguma linha no checkpoint,
        # significa que sobraram reparações sem correspondente.
        leftover = next(checkpoint_iter, None)

        if leftover is not None:
            fail(
                "O checkpoint possui registros que não foram consumidos."
            )

    # -----------------------------------------
    # Validações finais
    # -----------------------------------------

    print()
    print("=== VALIDAÇÃO ===")

    print(f"Total original:      {original_count}")
    print(f"Reparados:           {repaired_count}")
    print(f"Sem alteração:       {unchanged_count}")
    print(f"Checkpoint usado:    {checkpoint_count}")

    if original_count != EXPECTED_TOTAL:
        fail(
            f"Esperados {EXPECTED_TOTAL} registros, "
            f"mas foram encontrados {original_count}."
        )

    if repaired_count != EXPECTED_REPAIRED:
        fail(
            f"Esperados {EXPECTED_REPAIRED} reparados, "
            f"mas foram usados {repaired_count}."
        )

    if unchanged_count != EXPECTED_UNCHANGED:
        fail(
            f"Esperados {EXPECTED_UNCHANGED} registros mantidos, "
            f"mas foram encontrados {unchanged_count}."
        )

    if checkpoint_count != EXPECTED_REPAIRED:
        fail(
            f"Esperadas {EXPECTED_REPAIRED} linhas no checkpoint, "
            f"mas foram consumidas {checkpoint_count}."
        )

    print()
    print("=== MERGE CONCLUÍDO COM SUCESSO ===")
    print()
    print(f"Arquivo final:")
    print(OUTPUT_FILE)

    try:
        size_gb = OUTPUT_FILE.stat().st_size / (1024 ** 3)
        print(f"Tamanho: {size_gb:.2f} GB")
    except OSError:
        pass


if __name__ == "__main__":
    main()