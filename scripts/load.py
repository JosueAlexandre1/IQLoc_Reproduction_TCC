import sys

import os



current_dir = os.getcwd()

src_path = os.path.abspath(os.path.join(current_dir, "src"))



if src_path not in sys.path:

    sys.path.append(src_path)



try:

    from IR.Indexer.Indexer import Indexer

    print("Indexer and dependencies loaded successfully!")

except ModuleNotFoundError as e:

    print(f"Failed to find module: {e}")

    print(f"DEBUG: Searched in {src_path}")

    sys.exit(1)





# ============================================================

# CONFIGURAÇÃO

# ============================================================



DATASET_ROOT_PATH = r"C:\\Users\\PET\\Bench4BL\\data" #Raiz da pasta data do Bench4BL na máquina





# ============================================================

# INDEXAÇÃO

# ============================================================



def start_indexing_process():



    print("Connecting to Elasticsearch...")



    source_indexer = Indexer()



    total_indexed = 0



    # data/

    for sub_project in sorted(os.listdir(DATASET_ROOT_PATH)):



        sub_project_path = os.path.join(

            DATASET_ROOT_PATH,

            sub_project

        )



        if not os.path.isdir(sub_project_path):

            continue



        print(f"\n========== Sub-project: {sub_project} ==========")



        # data/<sub_project>/

        for project_name in sorted(os.listdir(sub_project_path)):



            project_path = os.path.join(

                sub_project_path,

                project_name

            )



            if not os.path.isdir(project_path):

                continue



            sources_dir = os.path.join(

                project_path,

                "sources"

            )



            if not os.path.isdir(sources_dir):

                print(

                    f"Sources folder not found: {sources_dir}"

                )

                continue



            print(f"\nProcessing Project: {project_name}")



            # sources/<version>/

            for version_name in sorted(os.listdir(sources_dir)):



                version_path = os.path.join(

                    sources_dir,

                    version_name

                )



                if not os.path.isdir(version_path):

                    continue



                print(f"  Indexing version: {version_name}")



                indexed_files = 0



                # Percorre recursivamente os arquivos da versão

                for root, dirs, files in os.walk(version_path):



                    for filename in files:



                        # A abordagem trabalha com arquivos Java

                        if not filename.endswith(".java"):

                            continue



                        file_full_path = os.path.join(

                            root,

                            filename

                        )



                        try:



                            with open(

                                file_full_path,

                                "r",

                                encoding="utf-8",

                                errors="ignore"

                            ) as f:



                                source_content = f.read()



                            source_indexer.bulk_index(

                                project=project_name,

                                sub_project=sub_project,

                                version=version_name,

                                source_code=source_content,

                                file_url=file_full_path

                            )



                            indexed_files += 1

                            total_indexed += 1



                        except Exception as e:



                            print(

                                f"    Error reading "

                                f"{file_full_path}: {e}"

                            )



                print(

                    f"    Indexed files in version: "

                    f"{indexed_files}"

                )



    # Envia o último lote pendente e faz refresh

    source_indexer.refresh()



    print("\n===================================")

    print("Indexing completed successfully!")

    print(f"Total files indexed: {total_indexed}")

    print("===================================")





if __name__ == "__main__":

    start_indexing_process()
