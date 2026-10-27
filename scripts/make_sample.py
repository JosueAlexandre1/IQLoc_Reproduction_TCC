
import json
import random
from pathlib import Path

# Estrutura:
# Projeto/
# ├── Dataset/
# │   ├── Bench4BLExtended.json
# │   ├── amostra.json
# │   └── restante.json
# └── scripts/
#     └── amostrar_dataset.py

raiz_projeto = Path(__file__).resolve().parent.parent
pasta_dataset = raiz_projeto / "Dataset"

arquivo_original = pasta_dataset / "Bench4BLExtended.json"
arquivo_amostra = pasta_dataset / "amostra.json"
arquivo_restante = pasta_dataset / "restante.json"


def main():
    # Usa o restante da execução anterior, se existir.
    # Caso contrário, começa pelo dataset original.
    if arquivo_restante.is_file():
        arquivo_entrada = arquivo_restante
        print("Continuando com os bugs restantes da execução anterior.")
    else:
        arquivo_entrada = arquivo_original
        print("Iniciando pelo dataset original.")

    if not arquivo_entrada.is_file():
        print("ERRO: arquivo de entrada não encontrado:")
        print(arquivo_entrada)
        return

    with arquivo_entrada.open("r", encoding="utf-8") as f:
        dados = json.load(f)

    if isinstance(dados, list):
        registros = dados
    elif isinstance(dados, dict):
        registros = list(dados.values())
    else:
        raise ValueError("O JSON precisa conter uma lista ou um objeto.")

    total = len(registros)
    print("Bugs disponíveis para seleção:", total)

    while True:
        try:
            quantidade = int(input("Quantos bugs deseja selecionar? "))

            if quantidade < 0:
                print("Digite um número maior ou igual a zero.")
            elif quantidade > total:
                print("Você só pode selecionar até {} bugs.".format(total))
            else:
                break

        except ValueError:
            print("Entrada inválida. Digite um número inteiro.")

    # Sorteia bugs sem repetição dentro desta seleção
    amostra = random.sample(registros, quantidade)

    # Acumula os selecionados no arquivo amostra.json existente
    amostra_anterior = []

    if arquivo_amostra.is_file():
        with arquivo_amostra.open("r", encoding="utf-8") as f:
            amostra_anterior = json.load(f)

    amostra_acumulada = amostra_anterior + amostra

    # Salva a amostra acumulada
    with arquivo_amostra.open("w", encoding="utf-8") as f:
        json.dump(amostra_acumulada, f, ensure_ascii=False, indent=4)

    # Remove os bugs selecionados da lista de disponíveis.
    # Como os registros são objetos JSON, comparamos seu conteúdo.
    selecionados_json = {
        json.dumps(registro, sort_keys=True, ensure_ascii=False)
        for registro in amostra
    }

    restante = [
        registro for registro in registros
        if json.dumps(registro, sort_keys=True, ensure_ascii=False)
        not in selecionados_json
    ]

    # Atualiza os bugs restantes
    with arquivo_restante.open("w", encoding="utf-8") as f:
        json.dump(restante, f, ensure_ascii=False, indent=4)

    print("\n========== CONCLUÍDO ==========")
    print("Selecionados nesta execução:", len(amostra))
    print("Total acumulado na amostra:", len(amostra_acumulada))
    print("Bugs restantes:", len(restante))
    print("Amostra:", arquivo_amostra)
    print("Restante:", arquivo_restante)


if __name__ == "__main__":
    main()
