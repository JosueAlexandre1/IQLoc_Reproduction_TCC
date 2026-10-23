from __future__ import annotations

from pathlib import Path
import sys
import os
import json
import xml.etree.ElementTree as ET

import ijson
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

DATA_ROOT = Path(
    "/home/josue/Área de trabalho/Bench4BL/data"
)

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

TOP_K_RESULTS = 100

# Salva o checkpoint a cada N bugs corrigidos
CHECKPOINT_INTERVAL = 100


# ============================================================
# IMPORTS DO PROJETO
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from IR.Searcher.Searcher import Searcher
from Utils.Parser.JavaSourceParser import JavaSourceParser


# ============================================================
# MESMA LÓGICA DA ETAPA A
# ============================================================

def _build_query(bug_report: dict) -> str:

    keywords = bug_report.get("keywords")

    if keywords is not None and str(keywords).strip() != "":
        return str(keywords)

    bug_title = str(
        bug_report.get("bug_title", "")
    ).strip()

    bug_description = str(
        bug_report.get("bug_description", "")
    ).strip()

    return (
        bug_title + " . " + bug_description
    ).strip(" .")


def _extract_methods(source_code: str) -> list[dict]:

    if (
        not isinstance(source_code, str)
        or source_code.strip() == ""
    ):
        return []

    try:
        parser = JavaSourceParser(
            source_code,
            clear_formatting=True
        )

        methods_dict = parser.parse_methods()

    except Exception:
        return []

    methods = []

    for method_name, method_body in methods_dict.items():

        if (
            not isinstance(method_body, str)
            or method_body.strip() == ""
        ):
            continue

        methods.append({
            "method_name": method_name,
            "method_body": method_body,
        })

    return methods


def _normalize_results(
    search_results: list[dict]
) -> list[dict]:

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


# ============================================================
# LÊ SOMENTE OS BUGS COM VERSION VAZIA
# ============================================================

def find_empty_version_bugs():

    targets = {}

    print("Lendo cache para encontrar bugs com version vazia...")

    with open(
        INPUT_CACHE,
        "r",
        encoding="utf-8"
    ) as f:

        for bug_report in ijson.items(f, "item"):

            version = bug_report.get("version")

            if version != "":
                continue

            project = bug_report.get("project")
            sub_project = bug_report.get("sub_project")
            bug_id = str(
                bug_report.get("bug_id", "")
            )

            if not project or not sub_project or not bug_id:
                print(
                    "AVISO: registro sem dados suficientes:",
                    project,
                    sub_project,
                    bug_id
                )
                continue

            key = (
                sub_project,
                project,
                bug_id
            )

            targets[key] = True

    print(
        "Bugs com version vazia:",
        len(targets)
    )

    return targets


# ============================================================
# ENCONTRA A VERSÃO PELO NOME DO XML
# ============================================================

def build_version_map(targets):

    version_map = {}

    # Agrupa os bugs por sub_project/project.
    #
    # Isso evita procurar todos os XMLs do Bench4BL
    # para cada bug individualmente.

    grouped_targets = {}

    for (
        sub_project,
        project,
        bug_id
    ) in targets:

        key = (
            sub_project,
            project
        )

        if key not in grouped_targets:
            grouped_targets[key] = set()

        grouped_targets[key].add(bug_id)

    total_projects = len(grouped_targets)

    print(
        "Projetos que precisam ser pesquisados:",
        total_projects
    )

    for index, (
        sub_project,
        project
    ) in enumerate(
        grouped_targets,
        start=1
    ):

        target_bug_ids = grouped_targets[
            (sub_project, project)
        ]

        repository_path = (
            DATA_ROOT
            / sub_project
            / project
            / "bugrepo"
            / "repository"
        )

        print(
            "[{}/{}] Procurando XMLs: {}/{}".format(
                index,
                total_projects,
                sub_project,
                project
            )
        )

        if not repository_path.is_dir():

            print(
                "  AVISO: repository não encontrado:",
                repository_path
            )

            continue

        found_here = set()

        xml_files = sorted(
            repository_path.glob("*.xml")
        )

        for xml_path in xml_files:

            # Se todos os bugs desse projeto já foram
            # encontrados, não precisamos continuar lendo XMLs.
            if found_here == target_bug_ids:
                break

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

            except ET.ParseError as e:

                print(
                    "  ERRO XML:",
                    xml_path
                )
                print(
                    "  ",
                    e
                )

                continue

            # O nome do XML É a versão usada pelo índice.
            #
            # CAMEL_1_6_0.xml
            #       ↓
            # CAMEL_1_6_0

            version = xml_path.stem

            for bug in root.findall("bug"):

                bug_id = str(
                    bug.get("id", "")
                )

                if bug_id not in target_bug_ids:
                    continue

                key = (
                    sub_project,
                    project,
                    bug_id
                )

                version_map[key] = version
                found_here.add(bug_id)

        missing = target_bug_ids - found_here

        print(
            "  Encontrados:",
            len(found_here),
            "/",
            len(target_bug_ids)
        )

        if missing:
            print(
                "  AVISO: bugs sem XML:",
                len(missing)
            )

            for bug_id in sorted(missing):
                print(
                    "    ",
                    sub_project,
                    project,
                    bug_id
                )

    print(
        "\nTotal de versões encontradas:",
        len(version_map)
    )

    missing_total = len(targets) - len(version_map)

    print(
        "Total sem versão encontrada:",
        missing_total
    )

    return version_map


# ============================================================
# LÊ CHECKPOINT
# ============================================================

def load_checkpoint():

    processed = {}

    if not CHECKPOINT_FILE.exists():
        return processed

    print(
        "Checkpoint encontrado:",
        CHECKPOINT_FILE
    )

    with open(
        CHECKPOINT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            key = (
                record.get("sub_project"),
                record.get("project"),
                str(record.get("bug_id", ""))
            )

            processed[key] = record

    print(
        "Registros já processados no checkpoint:",
        len(processed)
    )

    return processed


# ============================================================
# REPROCESSA SOMENTE OS BUGS NECESSÁRIOS
# ============================================================

def process_empty_versions(
    version_map,
    checkpoint
):

    searcher = Searcher()

    processed_count = 0
    skipped_checkpoint = 0
    missing_version = 0
    search_errors = 0

    checkpoint_file = open(
        CHECKPOINT_FILE,
        "a",
        encoding="utf-8"
    )

    try:

        with open(
            INPUT_CACHE,
            "r",
            encoding="utf-8"
        ) as input_file:

            for bug_report in tqdm(
                ijson.items(input_file, "item"),
                desc="Reprocessando versões vazias"
            ):

                # ------------------------------------------------
                # REGISTROS QUE JÁ TÊM VERSÃO:
                # NÃO PROCESSAR
                # ------------------------------------------------

                if bug_report.get("version") != "":
                    continue

                project = bug_report.get("project")
                sub_project = bug_report.get("sub_project")
                bug_id = str(
                    bug_report.get("bug_id", "")
                )

                key = (
                    sub_project,
                    project,
                    bug_id
                )

                # ------------------------------------------------
                # SE JÁ FOI PROCESSADO NO CHECKPOINT,
                # NÃO FAZER NOVAMENTE
                # ------------------------------------------------

                if key in checkpoint:

                    skipped_checkpoint += 1
                    continue

                # ------------------------------------------------
                # ENCONTRA A VERSÃO PELO XML
                # ------------------------------------------------

                version = version_map.get(key)

                if not version:

                    print(
                        "\nAVISO: versão não encontrada:",
                        key
                    )

                    missing_version += 1

                    continue

                # ------------------------------------------------
                # MESMA QUERY DA ETAPA A
                # ------------------------------------------------

                query = _build_query(
                    bug_report
                )

                # ------------------------------------------------
                # BUSCA NO ELASTICSEARCH
                # ------------------------------------------------

                try:

                    search_results = (
                        searcher.search_Extended(
                            project=project,
                            sub_project=sub_project,
                            version=version,
                            query=query,
                            top_K_results=TOP_K_RESULTS,
                            field_to_return=[
                                "file_url",
                                "source_code"
                            ],
                        )
                    )

                except Exception as e:

                    search_errors += 1

                    print(
                        "\nERRO no Elasticsearch:",
                        sub_project,
                        project,
                        bug_id
                    )

                    print(
                        "versão:",
                        version
                    )

                    print(
                        "erro:",
                        repr(e)
                    )

                    continue

                # ------------------------------------------------
                # ATUALIZA SOMENTE ESTE BUG
                # ------------------------------------------------

                repaired_report = dict(
                    bug_report
                )

                repaired_report["version"] = version

                repaired_report["Results"] = (
                    _normalize_results(
                        search_results
                    )
                )

                repaired_report[
                    "retrieval_query"
                ] = query

                repaired_report[
                    "retrieval_top_k"
                ] = TOP_K_RESULTS

                # ------------------------------------------------
                # SALVA NO CHECKPOINT
                # ------------------------------------------------

                checkpoint_file.write(
                    json.dumps(
                        repaired_report,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                processed_count += 1

                if (
                    processed_count
                    % CHECKPOINT_INTERVAL
                    == 0
                ):

                    checkpoint_file.flush()
                    os.fsync(
                        checkpoint_file.fileno()
                    )

                    print(
                        "\nCheckpoint salvo:",
                        processed_count,
                        "bugs"
                    )

    finally:

        checkpoint_file.flush()
        checkpoint_file.close()

    print("\n===================================")
    print("REPROCESSAMENTO TERMINADO")
    print("===================================")
    print(
        "Processados agora:",
        processed_count
    )
    print(
        "Já estavam no checkpoint:",
        skipped_checkpoint
    )
    print(
        "Sem versão encontrada:",
        missing_version
    )
    print(
        "Erros no Elasticsearch:",
        search_errors
    )


# ============================================================
# MONTA O CACHE FINAL
# ============================================================

def build_final_cache(
    checkpoint
):

    print(
        "\nMontando cache final..."
    )

    with open(
        INPUT_CACHE,
        "r",
        encoding="utf-8"
    ) as input_file, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        output_file.write("[\n")

        first = True

        for bug_report in tqdm(
            ijson.items(input_file, "item"),
            desc="Gerando cache final"
        ):

            # Chave do bug
            key = (
                bug_report.get("sub_project"),
                bug_report.get("project"),
                str(
                    bug_report.get(
                        "bug_id",
                        ""
                    )
                )
            )

            # Se foi um dos bugs corrigidos,
            # usa a versão/resultados do checkpoint.
            if key in checkpoint:

                final_report = checkpoint[key]

            else:

                # Todo o restante é mantido exatamente
                # em termos de conteúdo.
                final_report = bug_report

            if not first:
                output_file.write(",\n")

            json.dump(
                final_report,
                output_file,
                ensure_ascii=False
            )

            first = False

        output_file.write("\n]\n")
        output_file.flush()

    print(
        "\nCache final criado:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("===================================")
    print("REPARAÇÃO DE VERSÕES VAZIAS")
    print("===================================")

    print(
        "Cache:",
        INPUT_CACHE
    )

    print(
        "Bench4BL:",
        DATA_ROOT
    )

    print()

    if not INPUT_CACHE.exists():

        raise FileNotFoundError(
            "Cache não encontrado: {}".format(
                INPUT_CACHE
            )
        )

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            "DATA_ROOT não encontrado: {}".format(
                DATA_ROOT
            )
        )

    # --------------------------------------------------------
    # 1. Descobre SOMENTE os bugs com version vazia
    # --------------------------------------------------------

    targets = find_empty_version_bugs()

    if not targets:

        print(
            "Nenhum bug com version vazia."
        )

        return

    # --------------------------------------------------------
    # 2. Procura versões SOMENTE desses bugs
    # --------------------------------------------------------

    version_map = build_version_map(
        targets
    )

    # --------------------------------------------------------
    # 3. Carrega checkpoint existente
    # --------------------------------------------------------

    checkpoint = load_checkpoint()

    # --------------------------------------------------------
    # 4. Executa search_Extended SOMENTE nos bugs
    #    com version vazia que ainda não foram processados
    # --------------------------------------------------------

    process_empty_versions(
        version_map,
        checkpoint
    )

    # --------------------------------------------------------
    # 5. Recarrega checkpoint porque ele foi atualizado
    # --------------------------------------------------------

    checkpoint = load_checkpoint()

    # --------------------------------------------------------
    # 6. Gera novo cache final
    # --------------------------------------------------------

    build_final_cache(
        checkpoint
    )

    print(
        "\n==================================="
    )
    print(
        "CONCLUÍDO"
    )
    print(
        "==================================="
    )


if __name__ == "__main__":
    main()