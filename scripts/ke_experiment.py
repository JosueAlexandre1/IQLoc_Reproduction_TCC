from pathlib import Path
import os
import subprocess
import sys

# ============================================================
# Caminhos do projeto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

EXPERIMENTS = {
    "1": {
        "name": "KeyBERT (original)",
        "bug_report": SRC_DIR / "IQLoc" / "b" / "b_Generate_BR_Keywords.py",
        "source_code": SRC_DIR / "IQLoc" / "d" / "d_Generate_Source_Keywords.py",
    },
    "2": {
        "name": "YAKE",
        "bug_report": SRC_DIR / "Modified_IQLoc" / "b" / "b_YAKE_Generate_BR_Keywords.py",
        "source_code": SRC_DIR / "Modified_IQLoc" / "d" / "d_YAKE_Generate_Source_Keywords.py",
    },
    "3": {
        "name": "RAKE",
        "bug_report": SRC_DIR / "Modified_IQLoc" / "b" / "b_RAKE_Generate_BR_Keywords.py",
        "source_code": SRC_DIR / "Modified_IQLoc" / "d" / "d_RAKE_Generate_Source_Keywords.py",
    },
    "4": {
        "name": "TextRank",
        "bug_report": SRC_DIR / "Modified_IQLoc" / "b" / "b_TextRank_Generate_BR_Keywords.py",
        "source_code": SRC_DIR / "Modified_IQLoc" / "d" / "d_TextRank_Generate_Source_Keywords.py",
    },
}


def main():
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(SRC_DIR)
        if not existing_pythonpath
        else str(SRC_DIR) + os.pathsep + existing_pythonpath
    )

    while True:
        print("\n" + "=" * 50)
        print(" IQLoc - Experimental Keyword Extractors")
        print("=" * 50)
        print("1. Extrair palavras-chave dos Bug Reports")
        print("2. Extrair palavras-chave do Source Code")
        print("0. Sair")

        match input("Escolha a etapa: ").strip():
            case "1":
                etapa = "bug_report"
                nome_etapa = "Bug Report"
            case "2":
                etapa = "source_code"
                nome_etapa = "Source Code"
            case "0":
                print("Encerrando.")
                return
            case _:
                print("Opção inválida.")
                continue

        print(f"\n--- Extratores: {nome_etapa} ---")
        for option, experiment in EXPERIMENTS.items():
            print(f"{option}. {experiment['name']}")
        print("0. Voltar")

        match input("Escolha o extrator: ").strip():
            case "0":
                continue
            case choice if choice in EXPERIMENTS:
                selected = EXPERIMENTS[choice]
                script_path = selected[etapa]

                if not script_path.is_file():
                    print(f"Erro: script não encontrado:\n{script_path}")
                    continue

                print(f"\nExtrator: {selected['name']}")
                print(f"Etapa: {nome_etapa}")
                print(f"Script: {script_path}")
                print("-" * 50)

                try:
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        cwd=str(PROJECT_ROOT),
                        env=env,
                        check=False,
                    )
                except KeyboardInterrupt:
                    print("\nExecução interrompida pelo usuário.")
                    return

                if result.returncode == 0:
                    print("\nExecução concluída com sucesso.")
                else:
                    print(
                        f"\nO script terminou com código "
                        f"{result.returncode}."
                    )
            case _:
                print("Opção inválida.")


if __name__ == "__main__":
    main()


