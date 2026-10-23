from pathlib import Path
import json
import xml.etree.ElementTree as ET


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_ROOT = Path(
    "/home/josue/Área de trabalho/Bench4BL/data"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "Dataset"
    / "Bench4BLExtended.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Dataset"
    / "Bench4BLExtended_repaired.json"
)


def build_version_map():

    version_map = {}

    print("Construindo mapa de versões...")

    # sub_project/
    for sub_project_dir in DATA_ROOT.iterdir():

        if not sub_project_dir.is_dir():
            continue

        sub_project = sub_project_dir.name

        # sub_project/project/
        for project_dir in sub_project_dir.iterdir():

            if not project_dir.is_dir():
                continue

            project = project_dir.name

            repository_dir = (
                project_dir
                / "bugrepo"
                / "repository"
            )

            if not repository_dir.exists():
                continue

            xml_files = list(repository_dir.glob("*.xml"))

            for xml_file in xml_files:

                version = xml_file.stem

                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                except Exception as e:
                    print(
                        f"Erro lendo {xml_file}: {e}"
                    )
                    continue

                for bug in root.iter("bug"):

                    bug_id = bug.attrib.get("id")

                    if not bug_id:
                        continue

                    key = (
                        sub_project,
                        project,
                        str(bug_id)
                    )

                    version_map[key] = version

    print(
        f"Versões encontradas: {len(version_map)}"
    )

    return version_map


def repair_dataset():

    print(f"Lendo: {INPUT_FILE}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Total de bugs: {len(dataset)}")

    version_map = build_version_map()

    repaired = 0
    still_empty = 0

    for bug in dataset:

        version = bug.get("version")

        # Só mexe nas versões vazias
        if version != "":
            continue

        project = bug.get("project")
        sub_project = bug.get("sub_project")
        bug_id = bug.get("bug_id")

        key = (
            sub_project,
            project,
            str(bug_id)
        )

        recovered_version = version_map.get(key)

        if recovered_version:

            bug["version"] = recovered_version
            repaired += 1

        else:

            still_empty += 1

            print(
                "Não encontrada:",
                f"project={project}",
                f"sub_project={sub_project}",
                f"bug_id={bug_id}"
            )

    print()
    print("Resultado:")
    print(f"  Corrigidos: {repaired}")
    print(f"  Ainda vazios: {still_empty}")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(f"Arquivo salvo em:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    repair_dataset()