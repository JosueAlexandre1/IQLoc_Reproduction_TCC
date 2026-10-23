
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUTPUT = ROOT / "Output"

# Scripts existentes
A = SRC / "IQLoc" / "a_Cache_ES_Results.py"
B = SRC / "IQLoc" / "b_Generate_BR_Keywords.py"
C = SRC / "IQLoc" / "c_Generate_CE_Score.py"
D = SRC / "IQLoc" / "d_Generate_Source_Keywords.py"
E = SRC / "IQLoc" / "e_Localize_Bug.py"

B_MOD = SRC / "Modified_IQLoc" / "b"
D_MOD = SRC / "Modified_IQLoc" / "d"

EXTRACTORS = {
    "Original": {
        "b": B,
        "d": D,
        "br": "Cached_BR_Keywords.json",
        "source": "Cached_Source_Keywords.json",
    },
    "RAKE": {
        "b": B_MOD / "b_RAKE_Generate_BR_Keywords.py",
        "d": D_MOD / "d_Rake_Generate_Source_Keywords.py",
        "br": "Rake_Cached_BR_Keywords.json",
        "source": "Rake_Cached_Source_Keywords.json",
    },
    "TextRank": {
        "b": B_MOD / "b_TextRank_Generate_BR_Keywords.py",
        "d": D_MOD / "d_TextRank_Generate_Source_Keywords.py",
        "br": "TextRank_Cached_BR_Keywords.json",
        "source": "TextRank_Cached_Source_Keywords.json",
    },
    "YAKE": {
        "b": B_MOD / "b_YAKE_Generate_BR_Keywords.py",
        "d": D_MOD / "d_Yake_Generate_Source_Keywords.py",
        "br": "Yake_Cached_BR_Keywords.json",
        "source": "Yake_Cached_Source_Keywords.json",
    },
}

CROSS_ENCODERS = {
    "Original": "Cached_CE_Scores.json",
    "BAAI_bge-reranker-v2-m3":
        "BAAI_bge-reranker-v2-m3_Cached_CE_Scores.json",
    "MiniLM-L-6-v2":
        "cross-encoder_ms-marco-MiniLM-L-6-v2_Cached_CE_Scores.json",
}


def run_step(label, script, input_file, output_file, extra_args=None):
    """Executa uma etapa com caminhos explícitos."""
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not script.exists():
        raise FileNotFoundError(f"Script não encontrado: {script}")

    if not input_file.exists():
        raise FileNotFoundError(f"Entrada não encontrada: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Os scripts precisam aceitar --input e --output.
    command = [
        sys.executable, str(script),
        "--input", str(input_file),
        "--output", str(output_file),
    ]

    if extra_args:
        command.extend(extra_args)

    print(f"\n{'=' * 72}")
    print(f"ETAPA: {label}")
    print(f"Entrada: {input_file}")
    print(f"Saída:   {output_file}")
    print(f"{'=' * 72}")

    subprocess.run(command, cwd=ROOT, check=True)

    if not output_file.exists():
        raise FileNotFoundError(
            f"A etapa terminou, mas não gerou a saída esperada: {output_file}"
        )


def main():
    print("1 - Pipeline completa: A -> B -> C -> D -> E")
    print("2 - Reutilizar A; executar B -> C -> D -> E")
    print("3 - Reutilizar B; executar C -> D -> E")
    print("4 - Reutilizar C; executar D -> E")
    start = input("Escolha a partir de qual etapa executar: ").strip()

    print("\nExtrator de palavras dos relatórios (B):")
    print("1 - Original | 2 - RAKE | 3 - TextRank | 4 - YAKE")
    br_choice = input("Escolha: ").strip()
    br_name = {"1": "Original", "2": "RAKE",
               "3": "TextRank", "4": "YAKE"}.get(br_choice)
    if not br_name:
        raise ValueError("Extrator B inválido.")

    print("\nCrossEncoder (C):")
    print("1 - Original | 2 - BAAI bge-reranker-v2-m3")
    print("3 - cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce_choice = input("Escolha: ").strip()
    ce_name = {
        "1": "Original",
        "2": "BAAI_bge-reranker-v2-m3",
        "3": "MiniLM-L-6-v2",
    }.get(ce_choice)
    if not ce_name:
        raise ValueError("CrossEncoder inválido.")

    print("\nExtrator de palavras do código-fonte (D):")
    print("1 - Original | 2 - RAKE | 3 - TextRank | 4 - YAKE")
    source_choice = input("Escolha: ").strip()
    source_name = {"1": "Original", "2": "RAKE",
                   "3": "TextRank", "4": "YAKE"}.get(source_choice)
    if not source_name:
        raise ValueError("Extrator D inválido.")

    br = EXTRACTORS[br_name]
    source = EXTRACTORS[source_name]
    ce_file = CROSS_ENCODERS[ce_name]

    # A
    a_output = OUTPUT / "Caching" / "Cached_ES_Results.json"

    # B
    b_output = OUTPUT / "Generation" / "Experiments" / (
        f"{br_name}_Cached_BR_Keywords.json"
    )

    # C
    c_output = OUTPUT / "Caching" / "Experiments" / ce_file

    # D
    d_output = OUTPUT / "Generation" / "Experiments" / (
        f"{br_name}_{ce_name}_{source_name}_Cached_Source_Keywords.json"
    )

    # E
    e_dir = OUTPUT / "Localization" / "Experiments"
    experiment = f"{br_name}_{ce_name}_{source_name}"
    e_localized = e_dir / f"{experiment}_Localized_Bugs.json"
    e_performance = e_dir / f"{experiment}_Localization_Performance.json"

    if start == "1":
        run_step(
            "A - Recuperação inicial",
            A,
            ROOT / "Dataset" / "Bench4BLExtended.json",
            a_output,
        )
        start = "2"

    if start == "2":
        run_step("B - Palavras do relatório", br["b"], a_output, b_output)
        start = "3"

    if start == "3":
        run_step("C - Pontuação CrossEncoder", C, b_output, c_output)
        start = "4"

    if start == "4":
        run_step("D - Palavras do código-fonte", source["d"],
                 c_output, d_output)

        # E será parametrizado para receber os dois arquivos de saída.
        run_step(
            "E - Localização e avaliação",
            E,
            d_output,
            e_localized,
            extra_args=["--performance-output", str(e_performance)],
        )

    print("\nPipeline concluída.")


if __name__ == "__main__":
    main()
