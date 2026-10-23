from pathlib import Path
import argparse
import re


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TARGET_FILE = Path("src/IQLoc/c_Generate_CE_Score.py")


def main():
    parser = argparse.ArgumentParser(
        description="Altera o modelo DEFAULT_CE_MODEL usado em c_Generate_CE_Score.py"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo a ser usado. Padrão: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--file",
        default=str(TARGET_FILE),
        help=f"Arquivo alvo. Padrão: {TARGET_FILE}",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Cria uma cópia .bak antes de alterar o arquivo.",
    )
    args = parser.parse_args()

    target_file = Path(args.file)

    if not target_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {target_file}")

    text = target_file.read_text(encoding="utf-8")

    pattern = r'DEFAULT_CE_MODEL\s*=\s*["\'][^"\']+["\']'
    replacement = f'DEFAULT_CE_MODEL = "{args.model}"'

    if not re.search(pattern, text):
        raise RuntimeError(
            "Não encontrei a variável DEFAULT_CE_MODEL no arquivo. "
            "Confira se o nome da variável mudou."
        )

    new_text = re.sub(pattern, replacement, text, count=1)

    if args.backup:
        backup_file = target_file.with_suffix(target_file.suffix + ".bak")
        backup_file.write_text(text, encoding="utf-8")
        print(f"Backup criado em: {backup_file}")

    target_file.write_text(new_text, encoding="utf-8")

    print("Modelo alterado com sucesso.")
    print(f"Arquivo: {target_file}")
    print(f"Novo modelo: {args.model}")


if __name__ == "__main__":
    main()