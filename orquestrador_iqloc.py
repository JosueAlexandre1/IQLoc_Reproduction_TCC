#!/usr/bin/env python3
"""
Orquestrador de experimentos IQLoc.

Salve na raiz do repositório (a pasta que contém src/, Dataset/ e Output/).

Regras experimentais:
- Extratores: B e D usam sempre o mesmo extrator; C usa o modelo original.
- Modelos: B e D usam KeyBERT; C varia o Cross-Encoder.
- Os scripts originais continuam responsáveis pelo processamento e pelas barras tqdm.
- O orquestrador mede tempo e recursos da máquina, e salva logs JSON/CSV.
- --resume-d: reutiliza os arquivos existentes e retoma diretamente em D.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
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
    "1": ("Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-0.6B"),
    "2": ("Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-4B"),
    "4": ("Ettin-Reranker-400M", "cross-encoder/ettin-reranker-400m-v1"),
    "5": ("BGE-Reranker-v2-M3", "BAAI/bge-reranker-v2-m3"),
    "6": ("MiniLM-L6", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "7": ("ELECTRA-original", "cross-encoder/ms-marco-electra-base"),
}

ORIGINAL_MODEL = "7"

# Opções do CrossEncoderConfig.py: usadas para nomear os arquivos.
MODEL_PREFIX_CHOICES = {
    "5": "2",  # BGE
    "6": "3",  # MiniLM
    "7": "1",  # ELECTRA
}

# Prefixos dos experimentos que variam os extratores de B e D.
EXTRACTOR_PREFIX_CHOICES = {
    "keybert": "1",
    "yake": "4",
    "rake": "5",
    "textrank": "6",
}


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def flatten(row):
    result = {k: v for k, v in row.items() if k != "machine"}

    for key, value in row.get("machine", {}).items():
        result[f"machine_{key}"] = value

    return result


def write_csv(path, rows):
    rows = [flatten(r) for r in rows]

    if not rows:
        return

    keys = sorted({k for row in rows for k in row})

    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


class MachineMonitor:
    """Monitora CPU, RAM, disco e, quando disponível, GPU NVIDIA."""

    def __init__(self, interval=1.0):
        self.interval = max(0.25, interval)
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.gpu = None

        if psutil:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        if pynvml:
            try:
                pynvml.nvmlInit()

                if pynvml.nvmlDeviceGetCount():
                    self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

            except Exception:
                self.gpu = None

    def sample(self):
        row = {"timestamp": timestamp()}

        if psutil:
            try:
                vm = psutil.virtual_memory()

                row.update({
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_used_mb": round(vm.used / 1024**2, 2),
                    "ram_available_mb": round(vm.available / 1024**2, 2),
                    "ram_percent": vm.percent,
                })

                disk = psutil.disk_io_counters()

                if disk:
                    row["disk_read_mb_total"] = round(
                        disk.read_bytes / 1024**2, 2
                    )
                    row["disk_write_mb_total"] = round(
                        disk.write_bytes / 1024**2, 2
                    )

            except Exception:
                pass

        if self.gpu is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)

                row.update({
                    "gpu_percent": util.gpu,
                    "gpu_memory_percent": util.memory,
                    "vram_used_mb": round(mem.used / 1024**2, 2),
                    "vram_total_mb": round(mem.total / 1024**2, 2),
                })

                try:
                    row["gpu_temperature_c"] = (
                        pynvml.nvmlDeviceGetTemperature(
                            self.gpu,
                            pynvml.NVML_TEMPERATURE_GPU,
                        )
                    )
                except Exception:
                    pass

                try:
                    row["gpu_power_w"] = round(
                        pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000,
                        2,
                    )
                except Exception:
                    pass

            except Exception:
                pass

        self.samples.append(row)

    def _loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread:
            self.thread.join(timeout=max(2, self.interval * 3))

        summary = {"sample_count": len(self.samples)}

        keys = {
            k
            for sample in self.samples
            for k in sample
            if k != "timestamp"
        }

        for key in sorted(keys):
            values = [
                s[key]
                for s in self.samples
                if isinstance(s.get(key), (int, float))
            ]

            if values:
                summary[key + "_avg"] = round(
                    sum(values) / len(values), 3
                )
                summary[key + "_max"] = round(max(values), 3)

        return summary


class StageCFailure(RuntimeError):
    """Falha específica da etapa C."""

    pass


class IQLocOrchestrator:
    def __init__(
        self,
        root,
        log_dir,
        interval=1.0,
        dry_run=False,
        force_c=False,
        resume_d=False,
    ):
        self.root = Path(root).resolve()
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.interval = interval
        self.dry_run = dry_run
        self.force_c = force_c
        self.resume_d = resume_d

        self.started = time.time()

        self.stage_records = []
        self.experiment_records = []

        self.src = self.root / "src"
        self.prefixes = {
            k: v[1].replace("/", "_")
            for k, v in MODELS.items()
        }

        self.scripts = {
            "A": self.root / "src/IQLoc/a_Cache_ES_Results.py",
            "B_keybert": self.root / "src/IQLoc/b_Generate_BR_Keywords.py",
            "C": self.root / "src/IQLoc/c_Generate_CE_Score.py",
            "D_keybert": self.root / "src/IQLoc/d_Generate_Source_Keywords.py",
            "INDEX_CREATE": (
                self.root
                / "src/IR_Reretrieval/Indexer/Index_Creator.py"
            ),
            "REINDEX": (
                self.root
                / "src/IR_Reretrieval/ReIndexCreator/Reindexer_FULL.py"
            ),
            "E": self.root / "src/IQLoc/e_Localize_Bug.py",
        }

        missing = [
            str(p)
            for p in self.scripts.values()
            if not p.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Não encontrei estes scripts:\n" + "\n".join(missing)
            )

        # Ao retomar, preserva os registros existentes.
        if self.resume_d:
            self.load_previous_records()

        self.b_variants = self.discover_variants("b")
        self.d_variants = self.discover_variants("d")

    def load_previous_records(self):
        """Carrega os registros anteriores para não perder métricas."""

        stage_path = self.log_dir / "stage_records.json"
        experiment_path = self.log_dir / "experiment_records.json"

        if stage_path.exists():
            data = read_json(stage_path)

            if not isinstance(data, list):
                raise ValueError(
                    f"O arquivo {stage_path} não contém uma lista JSON."
                )

            self.stage_records = data
            print(
                f"[orquestrador] Registros anteriores de etapas carregados: "
                f"{len(data)}"
            )

        if experiment_path.exists():
            data = read_json(experiment_path)

            if not isinstance(data, list):
                raise ValueError(
                    f"O arquivo {experiment_path} não contém uma lista JSON."
                )

            self.experiment_records = data
            print(
                f"[orquestrador] Registros anteriores de experimentos "
                f"carregados: {len(data)}"
            )

    def discover_variants(self, stage):
        base = self.root / "src/Modified_IQLoc" / stage
        found = {}

        if not base.exists():
            return found

        for path in sorted(base.rglob("*.py")):
            name = path.stem.lower()

            if name.startswith("__") or any(
                x in name
                for x in (
                    "selector",
                    "config",
                    "helper",
                    "util",
                    "test",
                )
            ):
                continue

            for token, label in (
                ("keybert", "keybert"),
                ("rake", "rake"),
                ("textrank", "textrank"),
                ("text_rank", "textrank"),
                ("yake", "yake"),
            ):
                if token in name:
                    if label != "keybert":
                        found.setdefault(label, path)
                    break

        return found

    def persist(self):
        write_json(
            self.log_dir / "stage_records.json",
            self.stage_records,
        )
        write_csv(
            self.log_dir / "stage_records.csv",
            self.stage_records,
        )
        write_json(
            self.log_dir / "experiment_records.json",
            self.experiment_records,
        )
        write_csv(
            self.log_dir / "experiment_records.csv",
            self.experiment_records,
        )

    def get_log_path(self, exp_id, stage):
        """
        Não sobrescreve logs anteriores.
        Se o nome padrão já existir, cria um novo nome com data e hora.
        """

        base = self.log_dir / f"{slug(exp_id)}_{stage}.log"

        if not base.exists():
            return base

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.log_dir / (
            f"{slug(exp_id)}_{stage}_{stamp}.log"
        )

        counter = 2

        while candidate.exists():
            candidate = self.log_dir / (
                f"{slug(exp_id)}_{stage}_{stamp}_{counter}.log"
            )
            counter += 1

        return candidate

    def record_reused_stage(
        self,
        exp_id,
        stage,
        script,
        model_choice,
        output_file,
    ):
        """Registra explicitamente que uma saída existente foi reutilizada."""

        row = {
            "experiment_id": exp_id,
            "stage": stage,
            "script": str(script),
            "start_time": timestamp(),
            "end_time": timestamp(),
            "duration_seconds": 0,
            "status": "reused_existing_output",
            "output_file": str(output_file),
            "model_choice": model_choice,
        }

        self.stage_records.append(row)
        self.persist()

        print(
            f"[orquestrador] Etapa {stage} reutilizada: {output_file}"
        )

    def record_skipped_stage(
        self,
        exp_id,
        stage,
        script,
        model_choice,
    ):
        """Registra que uma etapa foi intencionalmente ignorada."""

        row = {
            "experiment_id": exp_id,
            "stage": stage,
            "script": str(script),
            "start_time": timestamp(),
            "end_time": timestamp(),
            "duration_seconds": 0,
            "status": "skipped_resume_d",
            "model_choice": model_choice,
        }

        self.stage_records.append(row)
        self.persist()

    def validate_resume_d_inputs(self, model_choice):
        """
        Confere os arquivos que D e as etapas posteriores precisam reutilizar.

        A etapa B já deve ter produzido o JSON canônico de palavras-chave
        dos relatórios. A etapa C já deve ter produzido os scores do modelo.
        """

        prefix = self.model_prefix(model_choice)

        br_keywords = (
            self.root
            / "Output/Generation"
            / f"{prefix}_Cached_BR_Keywords.json"
        )

        ce_scores = (
            self.root
            / "Output/Caching"
            / f"{prefix}_Cached_CE_Scores.json"
        )

        required = [
            ("palavras-chave dos relatórios (B)", br_keywords),
            ("scores do Cross-Encoder (C)", ce_scores),
        ]

        missing = []

        for description, path in required:
            if not path.is_file():
                missing.append(f"{description}: {path}")
            elif path.stat().st_size == 0:
                missing.append(f"{description} (arquivo vazio): {path}")

        if missing:
            raise FileNotFoundError(
                "Não é possível retomar em D. Arquivos necessários "
                "ausentes ou vazios:\n- "
                + "\n- ".join(missing)
                + "\n\nConfira se B e C terminaram e se os arquivos "
                "correspondem ao modelo/experimento desejado."
            )

        print("[orquestrador] Arquivos para retomar em D encontrados:")
        print(f"  B: {br_keywords}")
        print(f"  C: {ce_scores}")

        return br_keywords, ce_scores

    def run_script(
        self,
        exp_id,
        stage,
        script,
        model_choice,
        input_choices=None,
    ):
        start = time.time()
        log_path = self.get_log_path(exp_id, stage)

        row = {
            "experiment_id": exp_id,
            "stage": stage,
            "script": str(script),
            "start_time": timestamp(),
            "status": "running",
            "log_file": str(log_path),
            "model_choice": model_choice,
        }

        print(
            f"\n{'=' * 72}\n"
            f"[{exp_id}] Etapa {stage}: {script.name}\n"
            f"{'=' * 72}"
        )

        if self.dry_run:
            row.update({
                "status": "dry_run",
                "duration_seconds": 0,
                "end_time": timestamp(),
            })

            self.stage_records.append(row)
            self.persist()
            return row

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = (
            str(self.src)
            + os.pathsep
            + str(self.root)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        monitor = MachineMonitor(self.interval)
        monitor.start()
        output_tail = []

        try:
            choices = input_choices or []
            answers = "".join(f"{choice}\n" for choice in choices)

            with log_path.open(
                "w",
                encoding="utf-8",
                errors="replace",
            ) as log:
                proc = subprocess.Popen(
                    [sys.executable, str(script)],
                    cwd=self.root,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
)

                assert proc.stdin is not None
                assert proc.stdout is not None

                try:
                    if answers:
                        proc.stdin.write(answers)
                        proc.stdin.flush()

                    proc.stdin.close()

                except (BrokenPipeError, OSError):
                    pass

                for line in proc.stdout:
                    print(f"[{stage}] {line}", end="")
                    log.write(line)
                    log.flush()
                    output_tail.append(line)

                    if len(output_tail) > 80:
                        output_tail.pop(0)

                code = proc.wait()

            machine = monitor.stop()

            row.update({
                "end_time": timestamp(),
                "duration_seconds": round(time.time() - start, 3),
                "return_code": code,
                "status": "success" if code == 0 else "failed",
                "machine": machine,
            })

            if code:
                row["error_tail"] = "".join(output_tail)[-10000:]

        except Exception as exc:
            row.update({
                "end_time": timestamp(),
                "duration_seconds": round(time.time() - start, 3),
                "status": "failed",
                "error": repr(exc),
                "machine": monitor.stop(),
            })

        self.stage_records.append(row)
        self.persist()

        if row["status"] != "success":
            raise RuntimeError(
                f"Etapa {stage} falhou. Consulte {log_path}"
            )

        return row

    def prefix_choice_for_model(self, choice):
        try:
            return MODEL_PREFIX_CHOICES[choice]

        except KeyError:
            raise ValueError(
                f"O modelo {choice!r} ({MODELS[choice][1]}) não possui "
                "uma opção de prefixo correspondente no CrossEncoderConfig.py. "
                "As opções atualmente compatíveis são 5 (BGE), "
                "6 (MiniLM) e 7 (ELECTRA)."
            )

    def model_prefix(self, choice):
        return self.prefixes[choice]

    def prepare_prefixed_a_cache(self, choice):
        cache_dir = self.root / "Output/Caching"
        unprefixed = cache_dir / "Cached_ES_Results.json"
        prefixed = (
            cache_dir
            / f"{self.model_prefix(choice)}_Cached_ES_Results.json"
        )

        if not prefixed.exists() and unprefixed.exists():
            shutil.copy2(unprefixed, prefixed)

        if not prefixed.exists():
            raise FileNotFoundError(
                f"Cache da etapa A não encontrado: {prefixed}. "
                "Execute o orquestrador com --run-a para gerar o cache."
            )

    def variant_script(self, stage, variant):
        if variant == "keybert":
            return self.scripts[
                "B_keybert" if stage == "B" else "D_keybert"
            ]

        variants = self.b_variants if stage == "B" else self.d_variants

        if variant not in variants:
            raise ValueError(
                f"Não encontrei script da variante {variant!r} para {stage}. "
                f"Detectadas: {', '.join(sorted(variants)) or 'nenhuma'}"
            )

        return variants[variant]

    def run_variant_and_promote_output(
        self,
        exp_id,
        stage,
        variant,
        model_choice,
        prefix_choice,
    ):
        output_root = self.root / "Output/Generation/Modified"

        output_folder = output_root / (
            "Bug Reports" if stage == "B" else "Source Code"
        )

        before = {}

        if output_folder.exists():
            before = {
                p: p.stat().st_mtime_ns
                for p in output_folder.glob("*.json")
            }

        script = self.variant_script(stage, variant)

        self.run_script(
            exp_id,
            stage,
            script,
            model_choice,
            input_choices=[prefix_choice],
        )

        if variant == "keybert":
            return

        candidates = []

        if output_folder.exists():
            for path in output_folder.glob("*.json"):
                try:
                    if (
                        path not in before
                        or path.stat().st_mtime_ns != before[path]
                    ):
                        candidates.append(path)

                except OSError:
                    pass

        if not candidates:
            raise FileNotFoundError(
                f"A etapa {stage} terminou, mas não detectei um JSON "
                f"novo/modificado em {output_folder}. Confira "
                "INPUT_FILE/OUTPUT_DIR/OUTPUT_FILE_NAME no script da variante."
            )

        output = max(
            candidates,
            key=lambda p: p.stat().st_mtime_ns,
        )

        prefix = self.model_prefix(model_choice)

        if stage == "B":
            canonical = (
                self.root
                / "Output/Generation"
                / f"{prefix}_Cached_BR_Keywords.json"
            )
        else:
            canonical = (
                self.root
                / "Output/Generation"
                / f"{prefix}_Cached_Source_Keywords.json"
            )

        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, canonical)

        print(
            f"[orquestrador] Saída da variante {variant} "
            f"encaminhada para {canonical}"
        )

    def read_metrics(self, model_choice):
        path = (
            self.root
            / "Output/Localization"
            / f"{self.model_prefix(model_choice)}_Localization_Performance.json"
        )

        if not path.exists():
            return {
                "metrics_file": str(path),
                "metrics_status": "missing",
            }

        try:
            return {
                "metrics_file": str(path),
                "localization_metrics": read_json(path),
            }

        except Exception as exc:
            return {
                "metrics_file": str(path),
                "metrics_status": "unreadable",
                "metrics_error": repr(exc),
            }

    def run_reindexing_with_retries(
        self,
        exp_id,
        model_choice,
        max_attempts=3,
        retry_delay=5,
    ):
        for attempt in range(1, max_attempts + 1):
            print(
                f"\n[orquestrador] Reindexação: tentativa "
                f"{attempt}/{max_attempts}"
            )

            try:
                self.run_script(
                    exp_id,
                    f"INDEX_CREATE_attempt_{attempt}",
                    self.scripts["INDEX_CREATE"],
                    model_choice,
                )

                self.run_script(
                    exp_id,
                    f"REINDEX_attempt_{attempt}",
                    self.scripts["REINDEX"],
                    model_choice,
                )

                print("[orquestrador] Reindexação concluída.")
                return

            except Exception as exc:
                print(
                    f"[orquestrador] Tentativa {attempt} falhou: {exc}"
                )

                if attempt < max_attempts:
                    print(
                        f"[orquestrador] Tentando novamente em "
                        f"{retry_delay} segundos..."
                    )
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"Criação do índice/reindexação falhou após "
            f"{max_attempts} tentativas. Etapa E não será executada."
        )

    def run_experiment(
        self,
        kind,
        extractor,
        model_choice,
        run_a=False,
    ):
        prefix = self.model_prefix(model_choice)
        model_prefix_choice = self.prefix_choice_for_model(model_choice)

        exp_id = slug(
            f"{kind}_{extractor}_B-D_{MODELS[model_choice][0]}"
        )

        exp_start = time.time()

        row = {
            "experiment_id": exp_id,
            "kind": kind,
            "extractor_B": extractor,
            "extractor_D": extractor,
            "cross_encoder_choice": model_choice,
            "cross_encoder_model": MODELS[model_choice][1],
            "start_time": timestamp(),
            "status": "running",
            "resume_from_stage": "D" if self.resume_d else "A/B/C",
        }

        try:
            # ---------------------------------------------------------
            # Modo de retomada: reutiliza B e C e começa em D.
            # ---------------------------------------------------------
            if self.resume_d:
                br_keywords, ce_scores = self.validate_resume_d_inputs(
                    model_choice
                )

                self.record_skipped_stage(
                    exp_id,
                    "A",
                    self.scripts["A"],
                    model_choice,
                )

                self.record_reused_stage(
                    exp_id,
                    "B",
                    self.variant_script("B", extractor),
                    model_choice,
                    br_keywords,
                )

                self.record_reused_stage(
                    exp_id,
                    "C",
                    self.scripts["C"],
                    model_choice,
                    ce_scores,
                )

                print(
                    "\n[orquestrador] Retomando diretamente na etapa D. "
                    "As etapas A, B e C não serão executadas."
                )

            else:
                if run_a:
                    self.run_script(
                        exp_id,
                        "A",
                        self.scripts["A"],
                        model_choice,
                    )

                    source = (
                        self.root
                        / "Output/Caching/Cached_ES_Results.json"
                    )

                    target = (
                        self.root
                        / "Output/Caching"
                        / f"{prefix}_Cached_ES_Results.json"
                    )

                    if not source.exists():
                        raise FileNotFoundError(
                            f"A terminou, mas não encontrei {source}"
                        )

                    shutil.copy2(source, target)

                self.prepare_prefixed_a_cache(model_choice)

            # Experimentos de extratores: B e D usam o prefixo do extrator.
            # Experimentos de modelos: B e D usam o prefixo do modelo.
            if kind == "extractor":
                bd_prefix_choice = EXTRACTOR_PREFIX_CHOICES.get(extractor)

                if bd_prefix_choice is None:
                    raise ValueError(
                        f"Extrator sem opção de prefixo: {extractor!r}. "
                        f"Disponíveis: {', '.join(EXTRACTOR_PREFIX_CHOICES)}"
                    )

            else:
                bd_prefix_choice = model_prefix_choice

            if not self.resume_d:
                self.run_variant_and_promote_output(
                    exp_id,
                    "B",
                    extractor,
                    model_choice,
                    bd_prefix_choice,
                )

                try:
                    c_output = (
                        self.root
                        / "Output/Caching"
                        / f"{prefix}_Cached_CE_Scores.json"
                    )

                    if c_output.is_file() and not self.force_c:
                        print(
                            f"\n[orquestrador] Etapa C ignorada: "
                            f"o arquivo do experimento já existe: {c_output}"
                        )

                        self.stage_records.append({
                            "experiment_id": exp_id,
                            "stage": "C",
                            "script": str(self.scripts["C"]),
                            "start_time": timestamp(),
                            "end_time": timestamp(),
                            "duration_seconds": 0,
                            "status": "skipped_existing_output",
                            "output_file": str(c_output),
                            "model_choice": model_choice,
                        })

                        self.persist()

                    else:
                        if self.force_c and c_output.is_file():
                            print(
                                f"\n[orquestrador] --force-c informado: "
                                f"recalculando C apesar de existir {c_output}"
                            )

                        self.run_script(
                            exp_id,
                            "C",
                            self.scripts["C"],
                            model_choice,
                            input_choices=[
                                model_choice,
                                model_prefix_choice,
                            ],
                        )

                except Exception as exc:
                    raise StageCFailure(
                        f"Falha na etapa C do experimento {exp_id}: {exc}"
                    ) from exc

            # ---------------------------------------------------------
            # D: etapa de retomada.
            # ---------------------------------------------------------
            self.run_variant_and_promote_output(
                exp_id,
                "D",
                extractor,
                model_choice,
                bd_prefix_choice,
            )

            # Criação do índice e reindexação.
            self.run_reindexing_with_retries(
                exp_id,
                model_choice,
            )

            # E usa o prefixo do modelo para encontrar os resultados.
            self.run_script(
                exp_id,
                "E",
                self.scripts["E"],
                model_choice,
                input_choices=[model_prefix_choice],
            )

            row.update(self.read_metrics(model_choice))
            row["status"] = "success"

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            raise

        finally:
            row["end_time"] = timestamp()
            row["duration_seconds"] = round(
                time.time() - exp_start,
                3,
            )

            self.experiment_records.append(row)
            self.persist()

    def finish(self):
        elapsed = time.time() - self.started

        summary = {
            "start_time": dt.datetime.fromtimestamp(
                self.started
            ).astimezone().isoformat(timespec="seconds"),
            "end_time": timestamp(),
            "total_duration_seconds": round(elapsed, 3),
            "stage_count": len(self.stage_records),
            "stage_success_count": sum(
                r.get("status") == "success"
                for r in self.stage_records
            ),
            "stage_failure_count": sum(
                r.get("status") == "failed"
                for r in self.stage_records
            ),
            "stage_reused_count": sum(
                r.get("status") == "reused_existing_output"
                for r in self.stage_records
            ),
            "experiment_count": len(self.experiment_records),
            "experiment_success_count": sum(
                r.get("status") == "success"
                for r in self.experiment_records
            ),
            "experiment_failure_count": sum(
                r.get("status") == "failed"
                for r in self.experiment_records
            ),
            "resume_d": self.resume_d,
        }

        write_json(self.log_dir / "run_summary.json", summary)
        self.persist()

        print("\n=== RESUMO ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Arquivos de log: {self.log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador do pipeline IQLoc"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )

    parser.add_argument(
        "--mode",
        choices=("extractors", "models", "all"),
        default="extractors",
    )

    parser.add_argument(
        "--extractors",
        default="keybert,rake,textrank,yake",
        help=(
            "Extratores detectáveis em src/Modified_IQLoc/b e /d; "
            "B e D variam juntos."
        ),
    )

    parser.add_argument(
        "--models",
        default="6,7",
        help="Opções de modelo separadas por vírgula; padrão: MiniLM e ELECTRA.",
    )

    parser.add_argument(
        "--run-a",
        action="store_true",
        help="Executa A antes do primeiro experimento e cria o cache prefixado.",
    )

    parser.add_argument(
        "--force-c",
        action="store_true",
        help="Força o recálculo da etapa C mesmo quando os scores já existem.",
    )

    parser.add_argument(
        "--resume-d",
        action="store_true",
        help=(
            "Retoma diretamente na etapa D, reutilizando os arquivos "
            "existentes de B e C."
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Intervalo de amostragem da máquina em segundos.",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe o plano sem executar scripts.",
    )

    args = parser.parse_args()

    if args.resume_d and args.run_a:
        parser.error("--resume-d não pode ser usado junto com --run-a.")

    if args.resume_d and args.force_c:
        parser.error("--resume-d não pode ser usado junto com --force-c.")

    root = args.project_root.resolve()

    log_dir = (
        args.log_dir.resolve()
        if args.log_dir
        else root / "Output/OrchestratorLogs"
    )

    orch = IQLocOrchestrator(
        root,
        log_dir,
        args.interval,
        args.dry_run,
        args.force_c,
        args.resume_d,
    )

    extractors = [
        x.strip().lower()
        for x in args.extractors.split(",")
        if x.strip()
    ]

    choices = [
        x.strip()
        for x in args.models.split(",")
        if x.strip()
    ]

    invalid = [x for x in choices if x not in MODELS]

    if invalid:
        raise ValueError(
            f"Opções de modelo inválidas: {invalid}. "
            f"Use: {', '.join(MODELS)}"
        )

    plan = []

    if args.mode in ("extractors", "all"):
        for extractor in extractors:
            plan.append(("extractor", extractor, ORIGINAL_MODEL))

    if args.mode in ("models", "all"):
        for choice in choices:
            plan.append(("model", "keybert", choice))

    # A retomada precisa identificar inequivocamente qual experimento
    # deve ser continuado, pois os arquivos canônicos podem ser reutilizados.
    if args.resume_d and len(plan) != 1:
        parser.error(
            "--resume-d exige exatamente um experimento. "
            "Informe um único extrator ou modelo, por exemplo "
            "--mode extractors --extractors yake ou "
            "--mode models --models 6."
        )

    print("=== PLANO DE EXPERIMENTOS ===")

    for i, (kind, extractor, model) in enumerate(plan, 1):
        print(
            f"{i}. {kind}: B={extractor}, D={extractor}, "
            f"C={MODELS[model][1]}"
        )

    print(f"Raiz: {root}\nLogs: {log_dir}")

    if args.resume_d:
        print("Retomada ativada: início na etapa D.")

    if args.dry_run:
        print("Dry-run: nenhum script foi executado.")
        return

    run_a_next = args.run_a

    try:
        for kind, extractor, model in plan:
            try:
                orch.run_experiment(
                    kind,
                    extractor,
                    model,
                    run_a=run_a_next,
                )

            except StageCFailure as exc:
                print(
                    f"\n[orquestrador] {exc}\n"
                    "Etapa C falhou. "
                    "Continuando para o próximo experimento.\n"
                )

            finally:
                run_a_next = False

    finally:
        orch.finish()


if __name__ == "__main__":
    main()