import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_IR_CONFIG = PROJECT_ROOT / "src" / "IR" / "config" / "IR_config.yaml"
RERETRIEVAL_CONFIG = PROJECT_ROOT / "src" / "IR_Reretrieval" / "config" / "IR_config_2.yaml"

NORMAL_INDEX_CREATOR = PROJECT_ROOT / "src" / "IR" / "Indexer" / "Index_Creator.py"
RERETRIEVAL_INDEX_CREATOR = PROJECT_ROOT / "src" / "IR_Reretrieval" / "Indexer" / "Index_Creator.py"
REINDEXER_FULL = PROJECT_ROOT / "src" / "IR_Reretrieval" / "ReIndexCreator" / "Reindexer_FULL.py"

LOAD_SCRIPT = PROJECT_ROOT / "scripts" / "load.py"

A_SCRIPT = PROJECT_ROOT / "src" / "IQLoc" / "a_Cache_ES_Results.py"
B_SCRIPT = PROJECT_ROOT / "src" / "IQLoc" / "b_Generate_BR_Keywords.py"
C_SCRIPT = PROJECT_ROOT / "src" / "IQLoc" / "c_Generate_CE_Score.py"
D_SCRIPT = PROJECT_ROOT / "src" / "IQLoc" / "d_Generate_Source_Keywords.py"
E_SCRIPT = PROJECT_ROOT / "src" / "IQLoc" / "e_Localize_Bug.py"

NORMALIZE_SCRIPT = PROJECT_ROOT / "scripts" / "normalize_es_file_urls.py"

PERFORMANCE_FILE = PROJECT_ROOT / "Output" / "Localization" / "Localization_Performance.json"
LOCALIZED_BUGS_FILE = PROJECT_ROOT / "Output" / "Localization" / "Localized_Bugs.json"


def step(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def run_python(script_path: Path, env: dict, extra_args=None):
    if extra_args is None:
        extra_args = []

    if not script_path.exists():
        raise FileNotFoundError(f"Script não encontrado: {script_path}")

    cmd = [sys.executable, str(script_path), *extra_args]
    print(f"\nRodando: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def http_json(method: str, url: str, body=None):
    data = None
    headers = {}

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Erro HTTP {error.code} em {url}\n{body}") from error


def check_elasticsearch(elastic_url: str):
    try:
        result = http_json("GET", elastic_url)
    except Exception as exc:
        raise RuntimeError(
            f"Elasticsearch não respondeu em {elastic_url}. "
            f"Inicie o Elasticsearch antes de rodar o pipeline."
        ) from exc

    print(f"Elasticsearch OK.")
    print(f"Cluster: {result.get('cluster_name')}")
    print(f"Version: {result.get('version', {}).get('number')}")


def set_index_name(config_path: Path, index_name: str):
    if not config_path.exists():
        raise FileNotFoundError(f"Config não encontrado: {config_path}")

    text = config_path.read_text(encoding="utf-8")

    pattern = r"(?m)^(\s*index_name\s*:\s*).*$"

    if not re.search(pattern, text):
        raise RuntimeError(f"Não encontrei index_name em {config_path}")

    new_text = re.sub(pattern, rf"\g<1>{index_name}", text)

    config_path.write_text(new_text, encoding="utf-8")

    print(f"Config atualizado: {config_path}")
    print(f"index_name: {index_name}")


def index_count(elastic_url: str, index_name: str):
    result = http_json("GET", f"{elastic_url}/{index_name}/_count")
    return result.get("count", 0)


def delete_index_if_exists(elastic_url: str, index_name: str):
    try:
        http_json("DELETE", f"{elastic_url}/{index_name}")
        print(f"Índice removido: {index_name}")
    except Exception:
        print(f"Índice não existia ou não foi removido: {index_name}")


def reindex(elastic_url: str, source_index: str, dest_index: str):
    body = {
        "source": {"index": source_index},
        "dest": {"index": dest_index},
    }

    http_json("POST", f"{elastic_url}/_reindex", body)
    print(f"Reindex concluído: {source_index} -> {dest_index}")


def run_c_with_temp_model(env: dict, ce_model: str | None):
    if not C_SCRIPT.exists():
        raise FileNotFoundError(f"Script da etapa C não encontrado: {C_SCRIPT}")

    if not ce_model:
        run_python(C_SCRIPT, env)
        return

    temp_script = C_SCRIPT.parent / "_tmp_c_Generate_CE_Score.py"

    original_text = C_SCRIPT.read_text(encoding="utf-8")

    pattern = r'DEFAULT_CE_MODEL\s*=\s*["\'][^"\']+["\']'
    replacement = f'DEFAULT_CE_MODEL = "{ce_model}"'

    if not re.search(pattern, original_text):
        raise RuntimeError(
            "Não encontrei DEFAULT_CE_MODEL no c_Generate_CE_Score.py. "
            "Confira o nome da variável."
        )

    temp_text = re.sub(pattern, replacement, original_text, count=1)
    temp_script.write_text(temp_text, encoding="utf-8")

    try:
        print(f"Usando modelo temporário na etapa C: {ce_model}")
        run_python(temp_script, env)
    finally:
        if temp_script.exists():
            temp_script.unlink()
            print(f"Arquivo temporário removido: {temp_script}")


def show_final_summary():
    if LOCALIZED_BUGS_FILE.exists():
        data = json.loads(LOCALIZED_BUGS_FILE.read_text(encoding="utf-8"))
        total = len(data)
        empty = sum(1 for item in data if not item.get("localized_results"))
        non_empty = total - empty

        print()
        print("Resumo dos resultados localizados:")
        print(f"Total: {total}")
        print(f"Vazios: {empty}")
        print(f"Com resultado: {non_empty}")

    if PERFORMANCE_FILE.exists():
        performance = json.loads(PERFORMANCE_FILE.read_text(encoding="utf-8"))

        print()
        print("Métricas finais:")
        print(json.dumps(performance, indent=4, ensure_ascii=False))
    else:
        print(f"Arquivo de performance não encontrado: {PERFORMANCE_FILE}")


def ensure_required_files():
    required = [
        NORMAL_IR_CONFIG,
        RERETRIEVAL_CONFIG,
        NORMAL_INDEX_CREATOR,
        RERETRIEVAL_INDEX_CREATOR,
        REINDEXER_FULL,
        LOAD_SCRIPT,
        A_SCRIPT,
        B_SCRIPT,
        C_SCRIPT,
        D_SCRIPT,
        E_SCRIPT,
        NORMALIZE_SCRIPT,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        print("Arquivos obrigatórios não encontrados:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline local do IQLoc em sequência."
    )

    parser.add_argument("--elastic-url", default="http://localhost:9200")
    parser.add_argument("--normal-index", default="bench4bl")
    parser.add_argument("--reretrieval-index", default="bench4bl_reretrieval")
    parser.add_argument("--eval-index", default="bench4bl_reretrieval_eval")

    parser.add_argument(
        "--ce-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help=(
            "Modelo temporário para a etapa C. "
            "Use --ce-model original para rodar sem alterar o modelo do arquivo."
        ),
    )

    parser.add_argument(
        "--skip-normal-index",
        action="store_true",
        help="Pula criação e indexação do índice normal bench4bl.",
    )

    parser.add_argument(
        "--skip-abcd",
        action="store_true",
        help="Pula etapas A, B, C e D.",
    )

    parser.add_argument(
        "--skip-reretrieval-build",
        action="store_true",
        help="Pula criação e população do índice de reretrieval.",
    )

    parser.add_argument(
        "--skip-clone-normalize",
        action="store_true",
        help="Pula clonagem e normalização do índice eval.",
    )

    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path

    ce_model = None if args.ce_model.lower() == "original" else args.ce_model

    print(f"Raiz do projeto: {PROJECT_ROOT}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")

    ensure_required_files()

    step("0. Verificando Elasticsearch")
    check_elasticsearch(args.elastic_url)

    if not args.skip_normal_index:
        step("1. Configurando índice normal do IR")
        set_index_name(NORMAL_IR_CONFIG, args.normal_index)

        step("2. Criando índice normal do IR")
        run_python(NORMAL_INDEX_CREATOR, env)

        step("3. Indexando arquivos-fonte")
        run_python(LOAD_SCRIPT, env)

        count = index_count(args.elastic_url, args.normal_index)
        print(f"Documentos em {args.normal_index}: {count}")

    if not args.skip_abcd:
        step("4. Etapa A: cache dos resultados do Elasticsearch")
        run_python(A_SCRIPT, env)

        step("5. Etapa B: geração de keywords dos bug reports")
        run_python(B_SCRIPT, env)

        step("6. Etapa C: geração de scores do Cross-Encoder")
        run_c_with_temp_model(env, ce_model)

        step("7. Etapa D: geração de keywords dos códigos-fonte")
        run_python(D_SCRIPT, env)

    if not args.skip_reretrieval_build:
        step("8. Configurando índice de reretrieval")
        set_index_name(RERETRIEVAL_CONFIG, args.reretrieval_index)

        step("9. Criando índice de reretrieval")
        run_python(RERETRIEVAL_INDEX_CREATOR, env)

        step("10. Populando índice de reretrieval")
        run_python(REINDEXER_FULL, env)

        count = index_count(args.elastic_url, args.reretrieval_index)
        bug_count = http_json(
            "GET",
            f"{args.elastic_url}/{args.reretrieval_index}/_count?q=_exists_:bug_id",
        ).get("count", 0)

        print(f"Documentos em {args.reretrieval_index}: {count}")
        print(f"Documentos com bug_id em {args.reretrieval_index}: {bug_count}")

    if not args.skip_clone_normalize:
        step("11. Clonando índice de reretrieval para índice de avaliação")
        delete_index_if_exists(args.elastic_url, args.eval_index)
        reindex(args.elastic_url, args.reretrieval_index, args.eval_index)

        time.sleep(1)

        count = index_count(args.elastic_url, args.eval_index)
        print(f"Documentos em {args.eval_index}: {count}")

        step("12. Normalizando file_url no índice de avaliação")
        run_python(
            NORMALIZE_SCRIPT,
            env,
            extra_args=["--index", args.eval_index],
        )

    step("13. Apontando etapa E para o índice de avaliação")
    set_index_name(RERETRIEVAL_CONFIG, args.eval_index)

    step("14. Etapa E: localização e avaliação final")
    run_python(E_SCRIPT, env)

    step("15. Resultado final")
    show_final_summary()

    print()
    print("Pipeline concluído com sucesso.")


if __name__ == "__main__":
    main()