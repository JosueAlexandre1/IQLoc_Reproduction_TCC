import os
import json
import xml.etree.ElementTree as ET


def build_iqloc_dataset(data_root, output_file):
    dataset = []

    # data/<sub_project>/
    for sub_project in sorted(os.listdir(data_root)):
        sub_project_path = os.path.join(data_root, sub_project)

        if not os.path.isdir(sub_project_path):
            continue

        print("\n========== Sub-project: {} ==========".format(sub_project))

        # data/<sub_project>/<project>/
        for project in sorted(os.listdir(sub_project_path)):
            project_path = os.path.join(sub_project_path, project)

            if not os.path.isdir(project_path):
                continue

            bugrepo_path = os.path.join(project_path, "bugrepo")
            repository_path = os.path.join(bugrepo_path, "repository")
            sources_path = os.path.join(project_path, "sources")

            if not os.path.isdir(repository_path):
                continue

            print("\nProcessando Project: {}".format(project))

            # ---------------------------------------------------------
            # Mapeia as versões reais existentes em sources/
            #
            # Exemplo:
            # sources/
            #   CAMEL_1_1_0/
            #   CAMEL_1_2_0/
            #   CAMEL_1_3_0/
            #
            # Essas são exatamente as strings usadas pelo indexador.
            # ---------------------------------------------------------

            source_versions = {}

            if os.path.isdir(sources_path):
                for version_dir in os.listdir(sources_path):
                    version_path = os.path.join(
                        sources_path,
                        version_dir
                    )

                    if os.path.isdir(version_path):
                        source_versions[version_dir] = version_dir

            # bugrepo/repository/*.xml
            for xml_file in sorted(os.listdir(repository_path)):

                if not xml_file.endswith(".xml"):
                    continue

                xml_path = os.path.join(repository_path, xml_file)

                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                except ET.ParseError as e:
                    print("  ERRO XML: {}".format(xml_path))
                    print("  {}".format(e))
                    continue

                # Cada <bug> do XML
                for bug in root.findall("bug"):

                    bug_id = bug.get("id", "")

                    bug_info = bug.find("buginformation")

                    if bug_info is None:
                        continue

                    summary_node = bug_info.find("summary")
                    description_node = bug_info.find("description")
                    version_node = bug_info.find("version")

                    summary = (
                        summary_node.text.strip()
                        if summary_node is not None and summary_node.text
                        else ""
                    )

                    description = (
                        description_node.text.strip()
                        if description_node is not None and description_node.text
                        else ""
                    )

                    xml_version = (
                        version_node.text.strip()
                        if version_node is not None and version_node.text
                        else ""
                    )

                    # -------------------------------------------------
                    # Descobre a versão REAL usada pelo índice.
                    #
                    # O indexador usa:
                    #
                    #     version_name
                    #
                    # que corresponde diretamente ao nome da pasta
                    # dentro de sources/.
                    #
                    # Portanto o JSON precisa usar exatamente o mesmo
                    # nome da pasta.
                    # -------------------------------------------------

                    version = ""

                    if xml_version:
                        # 1. Caso o XML já tenha exatamente o nome
                        #    da pasta.
                        if xml_version in source_versions:
                            version = source_versions[xml_version]

                        else:
                            # 2. Caso o XML tenha apenas a versão,
                            #    por exemplo:
                            #
                            #    1.1.0
                            #
                            #    e a pasta seja:
                            #
                            #    CAMEL_1_1_0
                            #
                            normalized_xml_version = (
                                xml_version
                                .replace(".", "_")
                                .replace("-", "_")
                            )

                            expected_version = "{}_{}".format(
                                project,
                                normalized_xml_version
                            )

                            if expected_version in source_versions:
                                version = source_versions[expected_version]

                            else:
                                # 3. Procura uma pasta cujo nome
                                #    termine com a versão normalizada.
                                #
                                #    Isso evita depender de uma
                                #    convenção específica além do
                                #    nome real da pasta.
                                for source_version in source_versions:
                                    if source_version.endswith(
                                        "_" + normalized_xml_version
                                    ):
                                        version = source_version
                                        break

                    # -------------------------------------------------
                    # Se ainda não encontrou a versão, não inventa uma
                    # versão que pode não existir no índice.
                    #
                    # Isso é importante para evitar novamente uma
                    # inconsistência entre Dataset e Elasticsearch.
                    # -------------------------------------------------

                    if not version:
                        print(
                            "  AVISO: versão não encontrada em sources: "
                            "{} | projeto: {} | bug: {}".format(
                                xml_version,
                                project,
                                bug_id
                            )
                        )

                    # -------------------------------------------------
                    # Arquivos modificados/corrigidos
                    # -------------------------------------------------

                    fixed_files = []

                    fixed_files_node = bug.find("fixedFiles")

                    if fixed_files_node is not None:
                        for file_node in fixed_files_node.findall("file"):
                            if file_node.text:
                                fixed_files.append(
                                    file_node.text.strip()
                                )

                    # -------------------------------------------------
                    # Adiciona TODOS os bugs encontrados no XML
                    # -------------------------------------------------

                    dataset.append({
                        "bug_id": bug_id,
                        "project": project,
                        "sub_project": sub_project,
                        "version": version,
                        "bug_title": summary,
                        "bug_description": description,
                        "fixed_files": fixed_files
                    })

    # Cria o diretório de saída
    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Salva o JSON
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(
            dataset,
            out,
            indent=4,
            ensure_ascii=False
        )

    print("\n===================================")
    print("Dataset gerado com sucesso!")
    print("Total de bugs: {}".format(len(dataset)))
    print("Arquivo: {}".format(output_file))
    print("===================================")


build_iqloc_dataset(
    "/home/josue/Área de trabalho/Bench4BL/data",
    "/home/josue/Área de trabalho/iqloc/iqloc_reproduction/Dataset/Bench4BLExtended.json"
)