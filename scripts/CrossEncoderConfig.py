OPCOES = {
    "1": "cross-encoder_ms-marco-electra-base",
    "2": "BAAI_bge-reranker-v2-m3",
    "3": "cross-encoder_ms-marco-MiniLM-L-6-v2",
    "4": "YAKE",
    "5": "RAKE",
    "6": "TextRank",
}

print("\nEscolha o experimento:")
print("1 - Original (cross-encoder/ms-marco-electra-base)")
print("2 - BAAI/bge-reranker-v2-m3")
print("3 - cross-encoder/ms-marco-MiniLM-L-6-v2")
print("4 - YAKE")
print("5 - RAKE")
print("6 - TextRank")

escolha = input("Opção (1-6): ").strip()

if escolha not in OPCOES:
    raise ValueError("Opção inválida. Escolha de 1 a 6.")

CROSS_ENCODER_PREFIX = OPCOES[escolha]