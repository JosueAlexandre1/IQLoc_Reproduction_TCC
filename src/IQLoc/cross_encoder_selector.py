

import importlib.util
import subprocess
import sys

MODELS = {
    "1": {
        "name": "Qwen3 Reranker 0.6B",
        "id": "Qwen/Qwen3-Reranker-0.6B",
        "compatible": True,
    },
    "2": {
        "name": "Qwen3 Reranker 4B",
        "id": "Qwen/Qwen3-Reranker-4B",
        "compatible": True,
    },
    "3": {
        "name": "Jina Reranker v3.5 (API própria)",
        "id": "jinaai/jina-reranker-v3.5",
        "compatible": False,
    },
    "4": {
        "name": "Ettin Reranker 400M",
        "id": "cross-encoder/ettin-reranker-400m-v1",
        "compatible": True,
    },
    "5": {
        "name": "BGE Reranker v2 M3",
        "id": "BAAI/bge-reranker-v2-m3",
        "compatible": True,
    },
    "6": {
        "name": "MS MARCO MiniLM L6 (baseline)",
        "id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "compatible": True,
    },
    "7": {
        "name": "MS MARCO ELECTRA (original)",
        "id": "cross-encoder/ms-marco-electra-base",
        "compatible": True,
    },
}

MODEL_CONFIGS = {
    "Qwen/Qwen3-Reranker-0.6B": {
        "batch_size": 1,
        "max_length": 256,
        "fp16": True,
        "empty_cuda_cache_after_result": True,
    },
    "Qwen/Qwen3-Reranker-4B": {
        "batch_size": 1,
        "max_length": 512,
        "fp16": True,
        "empty_cuda_cache_after_result": True,
    },
}

DEFAULT_MODEL_CONFIG = {
    "batch_size": 16,
    "max_length": None,
    "fp16": False,
    "empty_cuda_cache_after_result": False,
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
        print(
            f"\nInstalando dependências ausentes: "
            f"{', '.join(missing)}"
        )

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
        status = (
            "CrossEncoder"
            if model["compatible"]
            else "requer implementação própria"
        )

        print(
            f"{option}. {model['name']} "
            f"({model['id']}) [{status}]"
        )

    while True:
        choice = input("\nEscolha o modelo (1-7): ").strip()

        if choice not in MODELS:
            print("Opção inválida. Escolha de 1 a 7.")
            continue

        model = MODELS[choice]

        if not model["compatible"]:
            print(
                "\nEsse modelo não pode ser usado diretamente "
                "pelo scorer atual do IQLoc."
            )
            print(
                "Selecione outro modelo ou implemente a integração "
                "específica do Jina."
            )
            continue

        return model["id"]


ensure_dependencies()

MODEL_NAME = select_model()

# Prefixo para os nomes dos arquivos.
# Exemplo: cross-encoder_ms-marco-electra-base_Cached_CE_Scores.json
CROSS_ENCODER_PREFIX = MODEL_NAME.replace("/", "_")

# Configuração correspondente ao modelo selecionado.
MODEL_CONFIG = {
    **DEFAULT_MODEL_CONFIG,
    **MODEL_CONFIGS.get(MODEL_NAME, {}),
}


