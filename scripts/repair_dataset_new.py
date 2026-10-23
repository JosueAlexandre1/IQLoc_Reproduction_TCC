from pathlib import Path
import json
import shutil
import ijson


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

EXTENDED_FILE = (
    PROJECT_ROOT
    / "Dataset"
    / "Bench4BLExtended.json"
)

OLD_FILE = (
    PROJECT_ROOT
    / "Dataset"
    / "Bench4BLExtended_old.json"
)

KEYWORDS_FILE = (
    PROJECT_ROOT
    / "Output"
    / "Generation"
    / "Cached_Source_Keywords.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_bug_key(item):
    """
    Identifica um bug usando:
        project + sub_project + bug_id

    A version NÃO faz parte da chave.
    """

    if not isinstance(item, dict):
        return None

    project = item.get("project")
    sub_project = item.get("sub_project")
    bug_id = item.get("bug_id")

    if bug_id is None:
        bug_id = item.get("id")

    if bug_id is None:
        return None

    return (
        str(project or ""),
        str(sub_project or ""),
        str(bug_id),
    )


def extract_records(data):
    """
    Aceita tanto uma lista de registros quanto estruturas
    onde os registros estejam dentro de um dicionário.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value

    raise ValueError(
        "Não foi possível encontrar uma lista de registros no JSON."
    )


def stream_keyword_records(path):
    """
    Lê Cached_Source_Keywords.json em streaming.

    Não carrega o arquivo inteiro na memória.
    """

    with open(path, "rb") as f:

        # Caso o JSON seja uma lista:
        try:
            yield from ijson.items(f, "item")
            return
        except Exception:
            pass

    # Caso os registros estejam dentro de alguma chave do objeto raiz,
    # descobrimos a primeira lista usando o parser incremental.
    with open(path, "rb") as f:

        parser = ijson.parse(f)

        current_prefix = None

        for prefix, event, value in parser:

            if event == "start_array":
                current_prefix = prefix
                break

        if current_prefix is None:
            raise ValueError(
                "Não foi encontrada uma lista de registros "
                "no Cached_Source_Keywords.json."
            )

    with open(path, "rb") as f:
        yield from ijson.items(f, f"{current_prefix}.item")


def main():

    print("=" * 70)
    print("FILTRANDO Bench4BLExtended PELO Cached_Source_Keywords")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Verificar arquivos
    # ---------------------------------------------------------

    if not EXTENDED_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{EXTENDED_FILE}"
        )

    if not KEYWORDS_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{KEYWORDS_FILE}"
        )

    # ---------------------------------------------------------
    # 2. Ler Bench4BLExtended
    # ---------------------------------------------------------

    extended_data = extract_records(
        load_json(EXTENDED_FILE)
    )

    print()
    print(f"Bench4BLExtended atual: {len(extended_data)} bugs")
    print(
        "Lendo Cached_Source_Keywords em streaming..."
    )

    # ---------------------------------------------------------
    # 3. Ler Cached_Source_Keywords em streaming
    # ---------------------------------------------------------

    keyword_keys = set()
    keywords_total = 0
    keywords_without_key = 0

    for item in stream_keyword_records(KEYWORDS_FILE):

        keywords_total += 1

        key = get_bug_key(item)

        if key is None:
            keywords_without_key += 1
            continue

        keyword_keys.add(key)

    print(
        f"Cached_Source_Keywords: {keywords_total} registros"
    )

    print(
        f"Bugs únicos no Cached_Source_Keywords: "
        f"{len(keyword_keys)}"
    )

    if keywords_without_key:
        print(
            f"Registros sem identificação: "
            f"{keywords_without_key}"
        )

    # ---------------------------------------------------------
    # 4. Filtrar Bench4BLExtended
    # ---------------------------------------------------------

    filtered_data = []
    extended_keys_found = set()

    for item in extended_data:

        key = get_bug_key(item)

        if key is None:
            continue

        if key in keyword_keys:

            filtered_data.append(item)
            extended_keys_found.add(key)

    # ---------------------------------------------------------
    # 5. Estatísticas
    # ---------------------------------------------------------

    missing_from_extended = (
        keyword_keys - extended_keys_found
    )

    print()
    print("=" * 70)
    print("RESULTADO DA COMPARAÇÃO")
    print("=" * 70)

    print(
        f"Bugs no Cached_Source_Keywords: "
        f"{len(keyword_keys)}"
    )

    print(
        f"Bugs encontrados no Bench4BLExtended: "
        f"{len(extended_keys_found)}"
    )

    print(
        f"Bugs selecionados para o novo dataset: "
        f"{len(filtered_data)}"
    )

    print(
        f"Bugs do Cached_Source_Keywords ausentes no Extended: "
        f"{len(missing_from_extended)}"
    )

    # ---------------------------------------------------------
    # 6. Segurança
    # ---------------------------------------------------------

    if len(filtered_data) == 0:
        raise RuntimeError(
            "O filtro resultou em 0 bugs. "
            "Nenhum arquivo foi alterado."
        )

    if missing_from_extended:

        print()
        print(
            "ATENÇÃO: existem bugs no Cached_Source_Keywords "
            "que não foram encontrados no Bench4BLExtended."
        )

        print()
        print("Primeiros exemplos:")

        for key in list(missing_from_extended)[:20]:
            print(" ", key)

        print()

        response = input(
            "Deseja continuar mesmo assim? [s/N]: "
        ).strip().lower()

        if response != "s":
            print(
                "Operação cancelada. "
                "Nenhum arquivo foi alterado."
            )
            return

    # ---------------------------------------------------------
    # 7. Criar backup
    # ---------------------------------------------------------

    if OLD_FILE.exists():
        raise FileExistsError(
            f"O arquivo de backup já existe:\n{OLD_FILE}\n\n"
            "Remova/renomeie esse backup manualmente antes de "
            "executar novamente, para evitar sobrescrevê-lo."
        )

    print()
    print("Backup:")
    print(f"  {EXTENDED_FILE}")
    print(f"→ {OLD_FILE}")

    shutil.move(
        EXTENDED_FILE,
        OLD_FILE
    )

    # ---------------------------------------------------------
    # 8. Salvar novo dataset
    # ---------------------------------------------------------

    with open(
        EXTENDED_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            filtered_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---------------------------------------------------------
    # 9. Verificação final sem carregar o arquivo inteiro
    # ---------------------------------------------------------

    final_count = 0

    with open(
        EXTENDED_FILE,
        "rb"
    ) as f:

        for _ in ijson.items(f, "item"):
            final_count += 1

    print()
    print("=" * 70)
    print("CONCLUÍDO")
    print("=" * 70)

    print(f"Dataset antigo: {OLD_FILE}")
    print(f"Novo dataset:   {EXTENDED_FILE}")
    print(f"Bugs no novo dataset: {final_count}")

    print()
    print("Nenhum campo dos bugs foi alterado.")
    print("Apenas o subconjunto de bugs foi selecionado.")


if __name__ == "__main__":
    main()