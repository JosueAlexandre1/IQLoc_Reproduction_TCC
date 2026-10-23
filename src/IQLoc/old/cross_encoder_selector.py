
import importlib.util
import subprocess
import sys

MODELS = {
    "1": {
        "name": "Ettin Reranker 400M",
        "id": "cross-encoder/ettin-reranker-400m-v1",
    },
    "2": {
        "name": "MS MARCO MiniLM L12",
        "id": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    },
    "3": {
        "name": "MS MARCO MiniLM L6",
        "id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
    "4": {
        "name": "MS MARCO ELECTRA Base",
        "id": "cross-encoder/ms-marco-electra-base",
    },
}


def ensure_dependencies():
    required_packages = {
        "sentence_transformers": "sentence-transformers",
    }

    missing = [
        package
        for module, package in required_packages.items()
        if importlib.util.find_spec(module) is None
    ]

    if missing:
        print(f"\nInstalando dependências ausentes: {', '.join(missing)}")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *missing,
            ],
            check=True,
        )

        print("Dependências instaladas com sucesso.")
    else:
        print("\nDependências já instaladas.")


def select_model():
    print("\n=== Seleção de Cross-Encoder ===")

    for option, model in MODELS.items():
        print(f"{option}. {model['name']} ({model['id']})")

    while True:
        choice = input("\nEscolha o modelo (1-4): ").strip()

        if choice in MODELS:
            return MODELS[choice]["id"]

        print("Opção inválida. Escolha de 1 a 4.")


ensure_dependencies()

# Identificador do modelo escolhido, disponível para importação.
MODEL_NAME = select_model()
