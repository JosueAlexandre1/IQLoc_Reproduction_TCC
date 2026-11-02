#!/usr/bin/env python3
"""Orquestrador IQLoc: ELECTRA -> MiniLM -> comparação de extratores com ELECTRA.

Coloque na raiz do repositório. Também são necessários:
  src/IQLoc/d_Generate_Source_Keywords_topk.py
  src/IQLoc/e_Localize_Bug_experimentos.py

Etapa B (extrator) é independente de C (Cross-Encoder); E recebe o mapa B
explicitamente para não aproveitar por engano as keywords contidas no cache C.
Para comparar C, D KeyBERT usa top-5. Para comparar extratores, D usa
threshold 0.5 com ELECTRA para manter as variantes originais comparáveis.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None

MODELS = {
    "7": ("ELECTRA", "cross-encoder/ms-marco-electra-base", "1"),
    "6": ("MiniLM", "cross-encoder/ms-marco-MiniLM-L6-v2", "3"),
}
MINILM_LEGACY = "cross-encoder_ms-marco-MiniLM-L-6-v2"
EXTRACTOR_CHOICES = {"keybert": "1", "rake": "5", "textrank": "6", "yake": "4"}
# Prefixos utilizados pelo menu legado scripts/CrossEncoderConfig.py.
# Mantê-los diferentes dos IDs de modelo: são somente aliases de entrada.
EXTRACTOR_CACHE_PREFIXES = {
    "rake": ("RAKE", "Rake", "rake"),
    "textrank": ("TextRank", "Textrank", "TEXTRANK", "textrank"),
    "yake": ("YAKE", "Yake", "yake"),
}



def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_json(path, kind="list"):
    """Validação rápida de completude, NÃO substitui análise integral de JSON."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 2:
        return False
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 8192))
        return f.read().rstrip().endswith(b"]" if kind == "list" else b"}")


def link_or_copy(source, target):
    """Hardlink no mesmo volume para evitar copiar JSONs enormes.

    Os arquivos associados devem ser considerados IMUTÁVEIS; o orquestrador
    nunca reexecuta a etapa B escrevendo em links já criados.
    """
    source, target = Path(source), Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        print(f"[CACHE] Hardlink, sem cópia de dados: {target.name}")
    except OSError:
        print(f"[CACHE] Hardlink indisponível. Copiando: {target.name}")
        shutil.copy2(source, target)


def report_key(report):
    return json.dumps([str(report.get(field, "")) for field in
        ("project", "sub_project", "version", "bug_id")],
        ensure_ascii=False, separators=(",", ":"))


class Monitor:
    def __init__(self, interval):
        self.interval = max(0.25, interval)
        self.end = threading.Event()
        self.samples = []
        self.thread = None
        self.gpu = None
        if pynvml:
            try:
                pynvml.nvmlInit()
                if pynvml.nvmlDeviceGetCount():
                    self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

    def loop(self):
        while not self.end.is_set():
            row = {}
            if psutil:
                try:
                    vm = psutil.virtual_memory()
                    row.update(cpu_percent=psutil.cpu_percent(),
                               ram_used_mb=vm.used / 1048576,
                               ram_available_mb=vm.available / 1048576)
                except Exception:
                    pass
            if self.gpu is not None:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
                    row.update(vram_used_mb=mem.used / 1048576,
                               gpu_percent=util.gpu)
                except Exception:
                    pass
            self.samples.append(row)
            self.end.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.end.set()
        if self.thread:
            self.thread.join(timeout=self.interval + 3)
        ret = {"sample_count": len(self.samples)}
        keys = {k for sample in self.samples for k in sample}
        for key in keys:
            nums = [sample[key] for sample in self.samples if key in sample]
            if nums:
                ret[key + "_max"] = round(max(nums), 2)
                ret[key + "_avg"] = round(sum(nums) / len(nums), 2)
        return ret


class Runner:
    def __init__(self, root, interval, dry_run, strict_cache, force_c, reindex, force_a=False):
        self.root = root.resolve()
        self.interval = interval
        self.dry_run = dry_run
        self.strict_cache = strict_cache
        self.force_c = force_c
        self.reindex = reindex
        self.force_a = force_a
        self._a_ready = False
        self.out = root / "Output"
        self.logs = self.out / "OrchestratorLogs" / "Intelligent"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.records_file = self.logs / "stage_records.json"
        self.records = read_json(self.records_file) if self.records_file.exists() else []
        self.paths = {
            "A": root / "src/IQLoc/a_Cache_ES_Results.py",
            "B": root / "src/IQLoc/b_Generate_BR_Keywords.py",
            "C": root / "src/IQLoc/c_Generate_CE_Score.py",
            "D": root / "src/IQLoc/d_Generate_Source_Keywords_topk.py",
            "E": root / "src/IQLoc/e_Localize_Bug_experimentos.py",
            "INDEX": root / "src/IR_Reretrieval/Indexer/Index_Creator.py",
            "REINDEX": root / "src/IR_Reretrieval/ReIndexCreator/Reindexer_FULL.py",
        }
        self.variants = {s: self.discover(s) for s in "bd"}
        self.b_index_dir = self.out / "Experiments" / "B_Keyword_Maps"
        self.b_index_dir.mkdir(parents=True, exist_ok=True)
        self._reindex_done = False

    def discover(self, stage):
        base = self.root / "src/Modified_IQLoc" / stage
        found = {}
        if not base.is_dir():
            return found
        for p in sorted(base.rglob("*.py")):
            name = p.stem.lower()
            if any(x in name for x in ("selector", "config", "helper", "util", "test")):
                continue
            for key in ("rake", "textrank", "yake"):
                if key in name or (key == "textrank" and "text_rank" in name):
                    found.setdefault(key, p)
        return found

    def log_record(self, row):
        self.records.append(row)
        write_json(self.records_file, self.records)
        cols = sorted({k for item in self.records for k in item if k != "machine"})
        cols += sorted({"machine_" + k for item in self.records for k in item.get("machine", {})})
        with (self.logs / "stage_records.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for item in self.records:
                flat = {k: v for k, v in item.items() if k != "machine"}
                flat.update({"machine_" + k: v for k, v in item.get("machine", {}).items()})
                writer.writerow(flat)

    def run(self, exp, stage, script, *, answers=(), args=(), cwd=None):
        script = Path(script).resolve()
        if not script.is_file():
            raise FileNotFoundError(f"Script ausente para {stage}: {script}")
        working_directory = Path(cwd).resolve() if cwd is not None else self.root
        if not working_directory.is_dir():
            raise FileNotFoundError(
                f"Diretório de execução inexistente para {stage}: {working_directory}"
            )
        command = [sys.executable, str(script), *map(str, args)]
        row = {"experiment": exp, "stage": stage, "script": str(script),
               "cwd": str(working_directory), "started": now(),
               "status": "running", "command": command}
        print(f"\n[{exp}] {stage}: {script.name}")
        if self.dry_run:
            row.update(status="dry_run", duration_seconds=0, finished=now())
            self.log_record(row)
            return
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = (str(self.root / "src") + os.pathsep + str(self.root) +
                             os.pathsep + env.get("PYTHONPATH", ""))
        log_file = self.logs / f"{slug(exp)}_{slug(stage)}_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
        row["log_file"] = str(log_file)
        self.log_record(row)  # Registro ANTES de começar, inclusive contra reboot.
        record = self.records[-1]
        begin = time.time()
        monitor = Monitor(self.interval)
        monitor.start()
        tail = []
        try:
            with log_file.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    command, cwd=working_directory, env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                )
                try:
                    if answers:
                        proc.stdin.write("".join(f"{answer}\n" for answer in answers))
                        proc.stdin.flush()
                    proc.stdin.close()
                except (OSError, BrokenPipeError):
                    pass
                for line in proc.stdout:
                    print(f"[{stage}] {line}", end="")
                    log.write(line)
                    log.flush()
                    tail.append(line)
                    if len(tail) > 60:
                        tail.pop(0)
                code = proc.wait()
            record.update(status="success" if code == 0 else "failed", return_code=code)
            if code:
                record["error_tail"] = "".join(tail)[-6000:]
        except BaseException as exc:
            record.update(status="failed", error=repr(exc))
            raise
        finally:
            record.update(finished=now(), duration_seconds=round(time.time() - begin, 3),
                          machine=monitor.stop())
            write_json(self.records_file, self.records)
        if record["status"] != "success":
            raise RuntimeError(f"{stage} falhou; consulte {log_file}")

    def reused(self, exp, stage, path, status="reused"):
        print(f"[{exp}] {stage}: REUTILIZADO -> {path}")
        self.log_record({"experiment": exp, "stage": stage, "status": status,
                         "started": now(), "finished": now(),
                         "duration_seconds": 0, "output": str(path)})

    def cached_run(self, exp, stage, output, dependencies, script, *, args=(), answers=()):
        output = Path(output)
        signature = {"deps": [file_signature(p) for p in dependencies],
                     "script_sha256": file_sha(script),
                     "args": list(map(str, args)), "answers": list(map(str, answers))}
        sidecar = Path(str(output) + ".meta.json")
        if output.is_file():
            companion_ok = True
            if stage == "E":
                command = list(map(str, args))
                companion_ok = ("--output" in command and completed_json(
                    Path(command[command.index("--output") + 1])))
            if (sidecar.exists() and read_json(sidecar) == signature
                    and completed_json(output, kind="dict" if stage == "E" else "list")
                    and companion_ok):
                self.reused(exp, stage, output)
                return
            if not sidecar.exists():
                raise RuntimeError(f"Saída existente sem manifesto de proveniência: {output}. "
                                   "Mova-a ou gere novamente após verificar sua origem.")
            print(f"[{exp}] Cache {stage} DESATUALIZADO; recalculando.")
            output.unlink()  # Exclusivamente resultados derivados do novo orquestrador.
            sidecar.unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.run(exp, stage, script, args=args, answers=answers)
        if not completed_json(output, kind="dict" if stage == "E" else "list"):
            raise RuntimeError(f"Etapa {stage} terminou sem JSON de saída completo: {output}")
        write_json(sidecar, signature)

    def prefix(self, model):
        main = MODELS[model][1].replace("/", "_")
        if model == "6":
            candidate = self.out / "Caching" / (MINILM_LEGACY + "_Cached_CE_Scores.json")
            canonical = self.out / "Caching" / (main + "_Cached_CE_Scores.json")
            if candidate.exists() and not canonical.exists():
                return MINILM_LEGACY
        return main

    def ensure_a(self):
        """Executa A uma única vez quando falta cache; reusa cache válido.

        Não apaga automaticamente arquivos possivelmente incompletos.
        --force-a só é permitido se não houver saídas dependentes a invalidar.
        """
        cache_dir = self.out / "Caching"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "Cached_ES_Results.json"

        if self._a_ready and completed_json(path):
            return path

        if self.force_a:
            # Não misturar uma nova versão do cache A com B/C já calculados.
            dependent_patterns = [
                self.out / "Generation" / "*_Cached_BR_Keywords.json",
                cache_dir / "*_Cached_CE_Scores.json",
                self.out / "Experiments" / "B_Keyword_Maps" / "*.json",
                self.out / "Experiments" / "*" / "Cached_Source_Keywords.json",
            ]
            existing_dependents = [
                p for pattern in dependent_patterns
                for p in pattern.parent.glob(pattern.name) if p.is_file()
            ]
            if existing_dependents:
                examples = "\n  ".join(map(str, existing_dependents[:5]))
                raise RuntimeError(
                    "--force-a não pode ser usado com caches dependentes de A. "
                    "É necessário invalidar B/C/D/E de maneira controlada antes. "
                    f"Arquivos encontrados:\n  {examples}"
                )

            # Renomear é rápido mesmo para caches gigantes e mantém backup.
            previous = [path, *sorted(cache_dir.glob("*_Cached_ES_Results.json"))]
            for old in previous:
                if old.is_file():
                    backup = Path(str(old) + ".before_force_a")
                    counter = 2
                    while backup.exists():
                        backup = Path(str(old) + f".before_force_a_{counter}")
                        counter += 1
                    old.rename(backup)
                    print(f"[A] Cache anterior preservado em {backup}")
            self.force_a = False

        if completed_json(path):
            self.reused("shared", "A", path)
            self._a_ready = True
            return path

        if path.exists():
            # Não sobrescrever um cache parcial que talvez seja recuperável.
            raise RuntimeError(
                f"Cache A encontrado, mas parece incompleto: {path}. "
                "Verifique ou renomeie o arquivo antes de executar novamente."
            )

        # É possível que execuções antigas tenham salvo somente o cache prefixado.
        # Reutilize apenas arquivos cuja terminação de JSON pareça completa.
        candidates = [
            p for p in sorted(cache_dir.glob("*_Cached_ES_Results.json"))
            if completed_json(p)
        ]
        if candidates:
            # Cache A deveria ser independente do Cross-Encoder. Quando há
            # múltiplos candidatos com tamanhos diferentes, não escolher às cegas.
            sizes = {p.stat().st_size for p in candidates}
            if len(sizes) > 1:
                raise RuntimeError(
                    "Há caches A prefixados de tamanhos diferentes e nenhum "
                    "cache canônico. Verifique qual representa o dataset correto: "
                    + ", ".join(str(p) for p in candidates)
                )
            link_or_copy(candidates[0], path)
            if not completed_json(path):
                raise RuntimeError(f"Falha ao recuperar cache A de {candidates[0]}")
            self.reused("shared", "A", path, "reused_prefixed_cache")
            self._a_ready = True
            return path

        print("[A] Nenhum cache encontrado. Executando a etapa A uma única vez...")
        self.run("shared", "A", self.paths["A"])
        if not completed_json(path):
            raise RuntimeError(
                "A terminou, mas não produziu um JSON aparentemente completo "
                f"em {path}. Verifique o log de execução da etapa A."
            )
        self._a_ready = True
        print(f"[A] Cache pronto; será compartilhado por todos os experimentos: {path}")
        return path

    def ensure_a_alias(self, prefix):
        a = self.ensure_a()
        target = self.out / "Caching" / f"{prefix}_Cached_ES_Results.json"
        link_or_copy(a, target)

    def ensure_legacy_alias(self, source, target, label):
        """Cria alias seguro para scripts legados de B/D sem copiar JSONs grandes.

        Se um alias já existe apontando para outro arquivo, interrompe em vez
        de associar palavras/scores do experimento errado.
        """
        source, target = Path(source), Path(target)
        if not completed_json(source):
            raise RuntimeError(f"{label}: cache de origem ausente/incompleto: {source}")
        if target.exists():
            if not completed_json(target):
                raise RuntimeError(f"{label}: alias existente incompleto: {target}")
            if not os.path.samefile(source, target):
                raise RuntimeError(
                    f"{label}: já existe um cache independente no caminho do alias: {target}. "
                    f"Não vou substituí-lo nem presumir que equivale a {source}. "
                    "Verifique a origem e renomeie o alias antigo se for seguro."
                )
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
            print(f"[CACHE] Alias por hardlink, sem cópia: {target.name}")
        except OSError as exc:
            raise RuntimeError(
                f"{label}: não consegui criar hardlink {target} -> {source}. "
                "Não farei cópia automática de um JSON gigante. "
                "Confirme que origem e destino estão no mesmo volume NTFS."
            ) from exc
        return target

    def ensure_extractor_a_aliases(self, extractor):
        """Variantes B antigas leem A por prefixo do seletor de extratores."""
        source = self.ensure_a()
        for prefix in EXTRACTOR_CACHE_PREFIXES[extractor]:
            target = self.out / "Caching" / f"{prefix}_Cached_ES_Results.json"
            self.ensure_legacy_alias(source, target, f"B/{extractor}")
        print(f"[B/{extractor}] Cache A disponível para os prefixos do extrator")

    def ensure_extractor_c_aliases(self, extractor, c_source):
        """Variantes D antigas leem C por prefixo do extrator, NÃO do modelo."""
        for prefix in EXTRACTOR_CACHE_PREFIXES[extractor]:
            target = self.out / "Caching" / f"{prefix}_Cached_CE_Scores.json"
            self.ensure_legacy_alias(c_source, target, f"D/{extractor}")
        print(f"[D/{extractor}] C do ELECTRA exposto com o prefixo do extrator")

    def ensure_rake_nltk(self):
        """Instala apenas corpora NLTK ausentes, nunca modelos nem pacotes pip."""
        try:
            import nltk
        except ImportError as exc:
            raise RuntimeError(
                "RAKE requer nltk instalado no venv. Instale com: python -m pip install nltk"
            ) from exc
        resources = (
            ("corpora/stopwords", "stopwords"),
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
        )
        for resource_path, package in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                print(f"[RAKE] Recurso NLTK ausente: {package}; baixando...")
                success = nltk.download(package, quiet=True)
                if not success:
                    raise RuntimeError(
                        f"Falha ao baixar {package} via NLTK. "
                        "Execute: python -m nltk.downloader stopwords punkt punkt_tab"
                    )
                try:
                    nltk.data.find(resource_path)
                except LookupError as exc:
                    raise RuntimeError(
                        f"NLTK ainda não encontra {package}. "
                        "Execute: python -m nltk.downloader stopwords punkt punkt_tab"
                    ) from exc

    def check_scripts_before_processing(self, planned):
        """Erro de instalação deve aparecer ANTES do gasto de A/B/C."""
        needed = {"A", "B", "C", "E"}
        if any(extractor == "keybert" for _, extractor, _, _ in planned):
            needed.add("D")
        missing = [self.paths[stage] for stage in needed if not self.paths[stage].is_file()]
        for _, extractor, _, _ in planned:
            if extractor == "keybert":
                continue
            for stage in ("b", "d"):
                if extractor not in self.variants[stage]:
                    missing.append(self.root / "src/Modified_IQLoc" / stage /
                                   f"<script_{extractor}.py>")
        if missing:
            raise FileNotFoundError(
                "Arquivos do pipeline não encontrados antes de iniciar. "
                "Extraia o pacote completo na RAIZ do projeto preservando src/IQLoc/:\n  "
                + "\n  ".join(str(x) for x in missing)
                + "\nDepois execute novamente o orquestrador."
            )

    def b_path(self, prefix):
        return self.out / "Generation" / f"{prefix}_Cached_BR_Keywords.json"

    def c_path(self, prefix):
        return self.out / "Caching" / f"{prefix}_Cached_CE_Scores.json"

    def ensure_keybert_b(self, model, reuse_from=None):
        prefix = self.prefix(model)
        output = self.b_path(prefix)
        exp = f"model_{MODELS[model][0]}"
        if output.exists() and completed_json(output):
            self.reused(exp, "B_KeyBERT", output, "reused_existing")
            return output
        if reuse_from and completed_json(reuse_from):
            link_or_copy(reuse_from, output)
            self.reused(exp, "B_KeyBERT", output, "reused_cross_model")
            return output
        self.ensure_a_alias(prefix)
        self.run(exp, "B_KeyBERT", self.paths["B"], answers=[MODELS[model][2]])
        if not completed_json(output):
            raise RuntimeError(f"B não gerou saída esperada: {output}. Confira CrossEncoderConfig.")
        return output

    def c_provenance(self, model):
        scripts = [self.paths["C"],
                   self.root / "src/IQLoc/CE_Method_Scorer.py",
                   self.root / "src/IQLoc/cross_encoder_selector.py",
                   self.root / "scripts/CrossEncoderConfig.py"]
        return {"input_B": file_signature(self.b_path(self.prefix(model))),
                "model": MODELS[model][1],
                "source_hashes": {str(p.relative_to(self.root)): file_sha(p)
                                  for p in scripts if p.is_file()}}

    def ensure_c(self, model):
        prefix = self.prefix(model)
        output = self.c_path(prefix)
        exp = f"model_{MODELS[model][0]}"
        if output.exists() and not self.force_c:
            if not completed_json(output):
                raise RuntimeError(f"C parece incompleto: {output}. Verifique antes de reutilizar!")
            metadata = Path(str(output) + ".model.json")
            expected = self.c_provenance(model)
            if metadata.exists():
                saved = read_json(metadata)
                if saved.get("model") != MODELS[model][1]:
                    raise RuntimeError(f"Cache C declara modelo incompatível: {metadata}")
                if saved.get("provenance") and saved["provenance"] != expected:
                    raise RuntimeError(f"C existente foi calculada com outros arquivos/configurações: {metadata}. "
                                       "Verifique as diferenças antes de usar --force-c.")
            if not metadata.exists():
                if self.strict_cache:
                    raise RuntimeError(f"C legado sem comprovação de modelo: {output}. "
                                       "Desative --strict-cache somente depois de validar a origem.")
                print(f"[ATENÇÃO] C legado sem manifesto do modelo, usando nome do arquivo: {output.name}")
            self.reused(exp, "C", output, "reused_c_legacy" if not metadata.exists() else "reused")
            return output
        if self.force_c and output.exists():
            # Hardlink de B nunca é modificado aqui; C só lê B e escreve seu próprio arquivo.
            print(f"[FORCE] C será recalculado: {output}")
        self.run(exp, "C", self.paths["C"], answers=[model, MODELS[model][2]])
        # Pode ter havido diferença de alias de nome no script C.
        actual = self.c_path(self.prefix(model))
        if not completed_json(actual):
            raise RuntimeError(f"C não gerou JSON esperado: {actual}. Confira os seletores.")
        write_json(Path(str(actual) + ".model.json"),
                   {"model": MODELS[model][1], "created": now(),
                    "provenance": self.c_provenance(model)})
        return actual

    def build_keyword_map(self, extractor, source, producer_signature=None):
        import ijson
        output = self.b_index_dir / f"{extractor}.json"
        exp = f"B_{extractor}"
        sig = {"input": file_signature(source), "extractor": extractor,
               "map_schema": "bug_identity_v1", "producer": producer_signature}
        meta = Path(str(output) + ".meta.json")
        if output.exists() and meta.exists() and read_json(meta) == sig:
            self.reused(exp, "B_map", output)
            return output
        print(f"[{exp}] Construindo mapa compacto de keywords da etapa B...")
        mapping = {}
        empty_keyword_count = 0
        with Path(source).open("rb") as infile:
            for record in ijson.items(infile, "item", use_float=True):
                if not isinstance(record, dict):
                    raise ValueError(f"B inválido: {source}")
                key = report_key(record)
                kw = record.get("keywords")
                if "keywords" not in record:
                    raise ValueError(f"Sem campo 'keywords' na saída B ({extractor}): {key}")
                if key in mapping:
                    raise ValueError(f"Identidade repetida no cache B: {key}")
                if kw is None:
                    kw = ""
                if not str(kw).strip():
                    empty_keyword_count += 1
                mapping[key] = str(kw)
        if not mapping:
            raise ValueError(f"Nenhum bug na saída B: {source}")
        write_json(output, mapping)
        write_json(meta, sig)
        print(f"[{exp}] Indexados {len(mapping)} bugs; sem keywords: {empty_keyword_count}. "
              "Mapa reutilizável por todas as etapas E")
        return output

    def variant_output(self, stage, before):
        folder = self.out / "Generation/Modified" / ("Bug Reports" if stage == "B" else "Source Code")
        if not folder.exists():
            raise FileNotFoundError(f"Diretório de saída da variante inexistente: {folder}")
        changed = [p for p in folder.glob("*.json")
                   if p not in before or p.stat().st_mtime_ns != before[p]]
        if len(changed) != 1:
            raise RuntimeError(f"Esperava exatamente um JSON alterado em {folder}, "
                               f"encontrei {len(changed)}: {[p.name for p in changed]}. "
                               "Confira o caminho fixo da variante para evitar misturar experimentos.")
        return changed[0]

    def run_variant(self, stage, extractor):
        script = self.variants[stage.lower()].get(extractor)
        if not script:
            raise FileNotFoundError(f"Variante {stage}/{extractor} não encontrada em Modified_IQLoc")
        folder = self.out / "Generation/Modified" / ("Bug Reports" if stage == "B" else "Source Code")
        before = {p: p.stat().st_mtime_ns for p in folder.glob("*.json")} if folder.exists() else {}
        self.run(f"extractor_{extractor}", stage, script, answers=[EXTRACTOR_CHOICES[extractor]])
        source = self.variant_output(stage, before)
        if not completed_json(source):
            raise RuntimeError(f"JSON incompleto na variante {stage}: {source}")
        return source

    def ensure_variant_map(self, extractor):
        output = self.b_index_dir / f"{extractor}.json"
        script = self.variants["b"].get(extractor)
        if not script:
            raise FileNotFoundError(f"Variante B de {extractor} não detectada")
        producer = {"script_sha256": file_sha(script),
                    "a_cache": file_signature(self.ensure_a())}
        source_meta = Path(str(output) + ".meta.json")
        if output.is_file() and source_meta.exists():
            saved = read_json(source_meta)
            if saved.get("producer") == producer and completed_json(output, "dict"):
                self.reused(f"extractor_{extractor}", "B_map", output)
                return output
        if extractor == "rake":
            self.ensure_rake_nltk()
        self.ensure_extractor_a_aliases(extractor)
        source = self.run_variant("B", extractor)
        return self.build_keyword_map(extractor, source, producer)

    def validate_retrieval_index(self):
        """Confirma que E consultará o mesmo índice definido pelo Searcher_RE."""
        src_dir = str(self.root / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        searcher = None
        try:
            # Usa exatamente a classe/configuração utilizada pela etapa E.
            from IR_Reretrieval import Searcher_RE

            searcher = Searcher_RE()
            index_name = getattr(searcher, "index_name", None)
            if not index_name:
                raise RuntimeError(
                    "Searcher_RE não informou o nome do índice configurado."
                )

            if not searcher.es_client.indices.exists(index=index_name):
                raise RuntimeError(
                    f"O índice configurado para E ({index_name!r}) não existe, "
                    "mesmo após Index_Creator.py e Reindexer_FULL.py. "
                    "Confira src/IR_Reretrieval/config/IR_config_2.yaml e os logs "
                    "das duas etapas obrigatórias."
                )

            response = searcher.es_client.count(index=index_name)
            document_count = int(response.get("count", 0))
            if document_count <= 0:
                raise RuntimeError(
                    f"O índice configurado para E ({index_name!r}) existe, "
                    "mas está vazio após a reindexação. A etapa E foi bloqueada."
                )

            print(
                f"[INDEX] Validado para a etapa E: {index_name} "
                f"({document_count:,} documentos)."
            )
            self.log_record({
                "experiment": "shared",
                "stage": "INDEX_VALIDATE",
                "status": "success",
                "index_name": index_name,
                "document_count": document_count,
                "started": now(),
                "finished": now(),
                "duration_seconds": 0,
            })
            return index_name, document_count
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Não foi possível validar a conexão/índice do Elasticsearch usado "
                "pela etapa E. A etapa E não será iniciada."
            ) from exc
        finally:
            if searcher is not None:
                try:
                    searcher.es_client.close()
                except Exception:
                    pass

    def ensure_reindex(self):
        """Prepara o índice de E uma única vez, antes do primeiro experimento.

        A criação e a reindexação são obrigatórias. O parâmetro legado --reindex
        continua aceito por compatibilidade, mas não é mais necessário.
        """
        if self._reindex_done:
            return

        required_scripts = (self.paths["INDEX"], self.paths["REINDEX"])
        missing = [str(path) for path in required_scripts if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "As duas etapas obrigatórias de preparação do índice não foram encontradas:\n"
                + "\n".join(missing)
            )

        print(
            "[INDEX] Preparação obrigatória antes do primeiro experimento: "
            "Index_Creator.py -> Reindexer_FULL.py"
        )

        # Index_Creator usa caminhos de configuração relativos à pasta Indexer.
        self.run(
            "shared", "INDEX", self.paths["INDEX"],
            cwd=self.paths["INDEX"].parent,
        )

        # O Reindexer_FULL mantém o diretório de trabalho na raiz do projeto.
        self.run(
            "shared", "REINDEX", self.paths["REINDEX"],
            cwd=self.root,
        )

        # Não deixa E começar se o Reindexer terminou sem criar/preencher
        # exatamente o índice que Searcher_RE consulta.
        self.validate_retrieval_index()
        self._reindex_done = True

    def validate_d_model(self, source_c, generated_d):
        """Bloqueia reutilização acidental de C de outro Cross-Encoder nas variantes D.

        Compara identidades e scores das primeiras linhas, por posição no JSON.
        """
        import ijson
        def first(path):
            with Path(path).open("rb") as f:
                return next(ijson.items(f, "item", use_float=True), None)
        c = first(source_c)
        d = first(generated_d)
        if c is None or d is None or report_key(c) != report_key(d):
            raise RuntimeError("D foi gerada com bugs diferentes dos presentes no cache C")
        def scored(report):
            pairs = []
            for i, res in enumerate(report.get("Results", [])):
                methods = res.get("methods")
                if isinstance(methods, list):
                    vals = enumerate(methods)
                elif isinstance(methods, dict):
                    vals = methods.items()
                else:
                    continue
                for j, method in vals:
                    if isinstance(method, dict) and isinstance(method.get("ce_score"), (int, float)):
                        pairs.append((str(i) + ":" + str(j), float(method["ce_score"])))
            return dict(pairs)
        a, b = scored(c), scored(d)
        common = a.keys() & b.keys()
        if not common:
            raise RuntimeError("Não encontrei scores compartilhados para verificar a origem de D")
        if any(a[key] != b[key] for key in common):
            raise RuntimeError("Scores de D não correspondem ao cache C escolhido: "
                               "possível mistura de modelos/seletores")

    def experiment(self, exp_id, extractor, model, selection, keyword_map):
        self.ensure_reindex()
        c_path = self.c_path(self.prefix(model))
        if not completed_json(c_path):
            raise RuntimeError(f"C ausente para {model}: {c_path}")
        folder = self.out / "Experiments" / exp_id
        folder.mkdir(parents=True, exist_ok=True)
        d_out = folder / "Cached_Source_Keywords.json"
        loc_out = folder / "Localized_Bugs.json"
        metrics_out = folder / "Localization_Performance.json"
        if extractor == "keybert":
            d_script = self.paths["D"]
            d_args = ("--input", str(c_path), "--output", str(d_out),
                      "--mode", selection, "--top-k", "5", "--threshold", "0.5")
            self.cached_run(exp_id, "D", d_out, [c_path], d_script, args=d_args)
        else:
            # Variantes D antigas NÃO possuem CLI padronizada: preservamos
            # seu filtro 0.5, auditamos os scores e movemos a saída sem copiar GBs.
            if selection != "threshold":
                raise ValueError("Variantes D exigem selection=threshold até adaptação específica")
            sidecar = Path(str(d_out) + ".meta.json")
            script = self.variants["d"].get(extractor)
            if not script:
                raise FileNotFoundError(f"Variante D de {extractor} não detectada")
            sig = {"c": file_signature(c_path), "script_sha256": file_sha(script),
                   "selection": "threshold_0.5", "extractor": extractor}
            if d_out.exists() and sidecar.exists() and read_json(sidecar) == sig:
                self.reused(exp_id, "D", d_out)
            else:
                if d_out.exists():
                    d_out.unlink()
                if extractor == "rake":
                    self.ensure_rake_nltk()
                self.ensure_extractor_c_aliases(extractor, c_path)
                produced = self.run_variant("D", extractor)
                self.validate_d_model(c_path, produced)
                produced.replace(d_out)  # Renomeia: evita copiar JSON grande.
                write_json(sidecar, sig)
        e_args = ("--input", str(d_out), "--keyword-map", str(keyword_map),
                  "--output", str(loc_out), "--metrics", str(metrics_out),
                  "--shortlist-size", "5")
        self.cached_run(exp_id, "E", metrics_out,
                        [d_out, keyword_map], self.paths["E"], args=e_args)
        result = read_json(metrics_out)
        summary = {"experiment": exp_id, "extractor": extractor,
                   "cross_encoder": MODELS[model][1], "selection": selection,
                   "localization_metrics": result, "metrics_file": str(metrics_out)}
        print("[RESULTADO]", json.dumps(summary, ensure_ascii=False))
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--models", default="7,6", help="Ordem dos modelos; padrão ELECTRA,MiniLM")
    parser.add_argument("--extractors", default="keybert,rake,textrank,yake")
    parser.add_argument("--mode", choices=("all", "models", "extractors"), default="all")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--reindex", action="store_true",
        help="Legado/compatibilidade: a criação e a reindexação já são obrigatórias antes do primeiro experimento.",
    )
    parser.add_argument("--force-c", action="store_true", help="Força inferência C de cada modelo escolhido")
    parser.add_argument("--force-a", action="store_true", help="Reexecuta A, preservando backup; recusa se B/C/D existentes dependem de A")
    parser.add_argument("--strict-cache", action="store_true", help="Recusa caches C legados sem manifesto")
    parser.add_argument("--dry-run", action="store_true", help="Mostra apenas o plano, sem executar scripts")
    args = parser.parse_args()
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    extractors = [s.strip().lower() for s in args.extractors.split(",") if s.strip()]
    if not models or any(model not in MODELS for model in models):
        parser.error("Modelos disponíveis: 7 (ELECTRA), 6 (MiniLM)")
    if any(x not in EXTRACTOR_CHOICES for x in extractors):
        parser.error("Extratores disponíveis: keybert,rake,textrank,yake")
    if args.mode == "extractors" and "7" not in models:
        print("[NOTA] Comparação de extratores usa ELECTRA independentemente de --models.")
    planned = []
    if args.mode in ("models", "all"):
        planned += [(f"model_{MODELS[model][0]}_keybert_top5", "keybert", model, "top_k")
                    for model in models]
    if args.mode in ("extractors", "all"):
        planned += [(f"extractor_{extractor}_electra_threshold_0p5", extractor, "7", "threshold")
                    for extractor in extractors]
    print("=== PLANO DE EXPERIMENTOS ===")
    for i, (exp, extractor, model, selection) in enumerate(planned, 1):
        print(f"{i}. {exp} | B/D={extractor}, C={MODELS[model][0]}, seleção={selection}")
    if args.dry_run:
        print("Pré-etapas obrigatórias antes do primeiro experimento: INDEX -> REINDEX -> validação do índice de E.")
        return
    runner = Runner(args.project_root, args.interval, args.dry_run,
                    args.strict_cache, args.force_c, args.reindex, args.force_a)
    # Interrompe antes de A/B/C se o usuário copiou só o orquestrador do ZIP.
    runner.check_scripts_before_processing(planned)
    # Gera o cache da etapa A se ausente; reutiliza nas demais condições.
    runner.ensure_a()
    keybert_maps = {}
    prepared = {}
    results = []
    failures = []
    try:
        for exp, extractor, model, selection in planned:
            try:
                if model not in prepared:
                    prefix = runner.prefix(model)
                    runner.ensure_a_alias(prefix)
                    # O B KeyBERT é independente do Cross-Encoder. Reutiliza B
                    # de um modelo anterior, sem rerodar a extração.
                    first_b = next(iter(prepared.values()), None)
                    source_b = first_b["b"] if first_b else None
                    b = runner.ensure_keybert_b(model, reuse_from=source_b)
                    if "keybert" not in keybert_maps:
                        keybert_maps["keybert"] = runner.build_keyword_map("keybert", b)
                    runner.ensure_c(model)
                    prepared[model] = {"b": b, "c": runner.c_path(runner.prefix(model))}
                if extractor == "keybert":
                    map_path = keybert_maps["keybert"]
                else:
                    map_path = runner.ensure_variant_map(extractor)
                results.append(runner.experiment(exp, extractor, model, selection, map_path))
            except Exception as exc:
                message = f"{exp}: {type(exc).__name__}: {exc}"
                print(f"[FALHA] {message}", file=sys.stderr)
                failures.append({"experiment": exp, "error": message})
                # O próximo experimento independente ainda poderá ser concluído.
    finally:
        write_json(runner.logs / "latest_results.json", results)
        write_json(runner.logs / "latest_failures.json", failures)
    print(f"\nExperimentos concluídos: {len(results)}; falhas: {len(failures)}. "
          f"Resumo: {runner.logs / 'latest_results.json'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
