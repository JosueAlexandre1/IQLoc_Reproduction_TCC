import json
import random
from collections import defaultdict
from pathlib import Path

# Estrutura esperada:
# Projeto/
# ├── Dataset/
# │   ├── Bench4BLExtended.json
# │   ├── amostra.json       (criado pelo script)
# │   └── restante.json      (criado pelo script)
# └── scripts/
#     └── amostrar_dataset_estratificado.py

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_DATASET = RAIZ_PROJETO / "Dataset"

ARQUIVO_ORIGINAL = PASTA_DATASET / "Bench4BLExtended.json"
ARQUIVO_AMOSTRA = PASTA_DATASET / "amostra.json"
ARQUIVO_RESTANTE = PASTA_DATASET / "restante.json"

QUANTIDADE = 500      # Seleciona 100 bugs POR EXECUÇÃO
SEMENTE = 42          # Mantém o sorteio reproduzível


def carregar_lista(caminho):
    with caminho.open("r", encoding="utf-8") as f:
        dados = json.load(f)
    if isinstance(dados, list):
        return dados
    if isinstance(dados, dict):
        return list(dados.values())
    raise ValueError(f"{caminho}: o JSON deve conter uma lista ou objeto.")


def estrato(registro):
    """Estratifica por combinação de projeto e subprojeto."""
    if not isinstance(registro, dict):
        raise ValueError("Cada bug deve ser um objeto JSON.")
    projeto = registro.get("project")
    subprojeto = registro.get("sub_project")
    if not isinstance(projeto, str) or not projeto.strip():
        raise ValueError(f"Bug sem campo 'project' válido: {registro.get('bug_id')}")
    if not isinstance(subprojeto, str) or not subprojeto.strip():
        raise ValueError(f"Bug sem campo 'sub_project' válido: {registro.get('bug_id')}")
    return (projeto.strip(), subprojeto.strip())


def cotas_proporcionais(grupos, quantidade, total):
    """Método dos maiores restos: cotas inteiras somando exatamente 'quantidade'."""
    cotas = {}
    restos = []
    for chave, indices in grupos.items():
        cota_real = quantidade * len(indices) / total
        cota_inteira = int(cota_real)
        cotas[chave] = cota_inteira
        restos.append((cota_real - cota_inteira, chave))

    vagas = quantidade - sum(cotas.values())
    for _, chave in sorted(restos, key=lambda item: (-item[0], item[1]))[:vagas]:
        cotas[chave] += 1
    return cotas


def main():
    # Evita executar sobre saídas incompletas ou inconsistentes.
    existe_amostra = ARQUIVO_AMOSTRA.is_file()
    existe_restante = ARQUIVO_RESTANTE.is_file()
    if existe_amostra != existe_restante:
        raise RuntimeError(
            "Encontrado apenas um de 'amostra.json' e 'restante.json'. "
            "Confira os arquivos antes de executar para não duplicar bugs."
        )

    if existe_restante:
        registros = carregar_lista(ARQUIVO_RESTANTE)
        amostra_anterior = carregar_lista(ARQUIVO_AMOSTRA)
        print("Continuando a partir de restante.json.")
    else:
        if not ARQUIVO_ORIGINAL.is_file():
            raise FileNotFoundError(f"Dataset original não encontrado: {ARQUIVO_ORIGINAL}")
        registros = carregar_lista(ARQUIVO_ORIGINAL)
        amostra_anterior = []
        print("Iniciando a partir de Bench4BLExtended.json.")

    total = len(registros)
    print(f"Bugs disponíveis: {total}")
    if total < QUANTIDADE:
        raise ValueError(
            f"Há apenas {total} bugs disponíveis; não é possível selecionar {QUANTIDADE}."
        )

    # Cada índice pertence a exatamente um estrato.
    grupos = defaultdict(list)
    for indice, registro in enumerate(registros):
        grupos[estrato(registro)].append(indice)

    cotas = cotas_proporcionais(grupos, QUANTIDADE, total)
    rng = random.Random(SEMENTE + len(amostra_anterior))
    indices_sorteados = set()

    print("\nDistribuição da seleção desta execução:")
    for chave in sorted(grupos):
        quantidade_estrato = cotas[chave]
        if quantidade_estrato:
            indices_sorteados.update(rng.sample(grupos[chave], quantidade_estrato))
        print(f"  {chave[0]} / {chave[1]}: {quantidade_estrato} de {len(grupos[chave])}")

    # Separa por ÍNDICE, não por igualdade do JSON: isso evita remover bugs
    # adicionais quando dois registros possuem conteúdo idêntico.
    nova_amostra = [r for i, r in enumerate(registros) if i in indices_sorteados]
    restante = [r for i, r in enumerate(registros) if i not in indices_sorteados]
    amostra_acumulada = amostra_anterior + nova_amostra

    for caminho, dados in (
        (ARQUIVO_AMOSTRA, amostra_acumulada),
        (ARQUIVO_RESTANTE, restante),
    ):
        with caminho.open("w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=4)

    print("\n========== CONCLUÍDO ==========")
    print("Selecionados nesta execução:", len(nova_amostra))
    print("Total acumulado na amostra:", len(amostra_acumulada))
    print("Bugs restantes:", len(restante))
    print("Amostra:", ARQUIVO_AMOSTRA)
    print("Restante:", ARQUIVO_RESTANTE)


if __name__ == "__main__":
    main()
