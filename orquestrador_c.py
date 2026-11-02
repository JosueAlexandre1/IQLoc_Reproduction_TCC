

# ============================================================

# IQLoc — Stage C: seleção original + monitoramento

# Entrada fixa: Stage B ELECTRA

# Script original preservado

# ============================================================



import os

import sys

import re

import csv

import json

import time

import threading

import statistics

import runpy

from pathlib import Path

from datetime import datetime



# ------------------------------------------------------------

# DEPENDÊNCIAS

# ------------------------------------------------------------



def ensure_package(module, package):

    try:

        __import__(module)

    except ImportError:

        import subprocess

        subprocess.check_call([

            sys.executable, "-m", "pip", "install", "-q", package

        ])



ensure_package("psutil", "psutil")

ensure_package("pynvml", "nvidia-ml-py")



import psutil

import pynvml



# ------------------------------------------------------------

# CAMINHOS

# ------------------------------------------------------------



ROOT = Path(__file__).resolve().parent

SCRIPT = ROOT / "src/IQLoc/c_Generate_CE_Score.py"

TEMP_SCRIPT = ROOT / "src/IQLoc/_c_Generate_CE_Score_temp.py"



INPUT_FILE = (

    ROOT / "Output/Generation/"

    "cross-encoder_ms-marco-electra-base_Cached_BR_Keywords.json"

)



OUTPUT_DIR = ROOT / "Output/Caching"

LOG_DIR = ROOT / "Output/Logs_CE"



OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR.mkdir(parents=True, exist_ok=True)



INTERVAL = 1.0



if not SCRIPT.is_file():

    raise FileNotFoundError(f"Script original não encontrado: {SCRIPT}")



if not INPUT_FILE.is_file():

    raise FileNotFoundError(f"Entrada Stage B não encontrada: {INPUT_FILE}")



if TEMP_SCRIPT.exists():

    TEMP_SCRIPT.unlink()



# ------------------------------------------------------------

# MONITORAMENTO NVIDIA

# ------------------------------------------------------------



GPU_AVAILABLE = False

GPU_HANDLE = None



try:

    pynvml.nvmlInit()

    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

    GPU_AVAILABLE = True

    print("GPU NVIDIA acessível:", GPU_AVAILABLE)

except Exception as exc:

    print("Monitoramento NVIDIA indisponível:", exc)



# ------------------------------------------------------------

# MONITOR DE CPU, RAM, DISCO, GPU E VRAM

# ------------------------------------------------------------



class MachineMonitor:

    def __init__(self):

        self.samples = []

        self.lock = threading.Lock()

        self.stop_event = threading.Event()

        self.thread = None



    def sample(self):

        vm = psutil.virtual_memory()



        row = {

            "timestamp": datetime.now().isoformat(),

            "cpu_percent": psutil.cpu_percent(interval=None),

            "ram_used_mb": vm.used / 1024**2,

            "ram_available_mb": vm.available / 1024**2,

            "ram_percent": vm.percent,

            "disk_used_percent": psutil.disk_usage(str(ROOT)).percent,

        }



        if GPU_AVAILABLE:

            try:

                util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)

                mem = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)



                row.update({

                    "gpu_percent": util.gpu,

                    "gpu_memory_controller_percent": util.memory,

                    "vram_used_mb": mem.used / 1024**2,

                    "vram_total_mb": mem.total / 1024**2,

                    "gpu_memory_percent": (

                        100 * mem.used / mem.total if mem.total else None

                    ),

                })



                try:

                    row["gpu_temperature_c"] = (

                        pynvml.nvmlDeviceGetTemperature(

                            GPU_HANDLE,

                            pynvml.NVML_TEMPERATURE_GPU

                        )

                    )

                except Exception:

                    row["gpu_temperature_c"] = None



                try:

                    row["gpu_power_w"] = (

                        pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000

                    )

                except Exception:

                    row["gpu_power_w"] = None



            except Exception as exc:

                row["gpu_monitor_error"] = str(exc)



        with self.lock:

            self.samples.append(row)



    def run(self):

        psutil.cpu_percent(interval=None)



        while not self.stop_event.is_set():

            try:

                self.sample()

            except Exception as exc:

                print(f"\nFalha na coleta de métricas: {exc}", flush=True)



            self.stop_event.wait(INTERVAL)



    def start(self):

        self.thread = threading.Thread(target=self.run, daemon=True)

        self.thread.start()



    def stop(self):

        self.stop_event.set()



        if self.thread and self.thread.is_alive():

            self.thread.join(timeout=10)



        self.sample()



    def summary(self):

        with self.lock:

            samples = list(self.samples)



        result = {"samples": len(samples)}

        keys = {

            key for row in samples for key in row

            if key != "timestamp"

        }



        for key in sorted(keys):

            values = [

                row[key] for row in samples

                if isinstance(row.get(key), (int, float))

                and not isinstance(row.get(key), bool)

            ]



            if values:

                result[f"{key}_avg"] = statistics.mean(values)

                result[f"{key}_max"] = max(values)



        return result



# ------------------------------------------------------------

# REDIRECIONAMENTO DE SAÍDA PARA NOTEBOOK + LOG

# ------------------------------------------------------------



class Tee:

    def __init__(self, console, log):

        self.console = console

        self.log = log



    def write(self, text):

        self.console.write(text)

        self.log.write(text)

        self.log.flush()

        return len(text)



    def flush(self):

        self.console.flush()

        self.log.flush()



    def isatty(self):

        return self.console.isatty() if hasattr(

            self.console, "isatty"

        ) else False



    def fileno(self):

        return self.console.fileno()



    @property

    def encoding(self):

        return getattr(self.console, "encoding", "utf-8")



    def writable(self):

        return True



# ------------------------------------------------------------

# PREPARAR CÓPIA TEMPORÁRIA COM ENTRADA FIXA

# ------------------------------------------------------------



original_source = SCRIPT.read_text(encoding="utf-8")



pattern = re.compile(r"(?m)^([ \t]*)input_file_path\s*=.*$")

matches = list(pattern.finditer(original_source))



if len(matches) != 1:

    raise RuntimeError(

        "Não foi possível identificar com segurança a atribuição de "

        f"input_file_path: encontrei {len(matches)} ocorrências. "

        "O script original não foi alterado."

    )



patched_source = pattern.sub(

    lambda match: (

        match.group(1)

        + "input_file_path = "

        + repr(str(INPUT_FILE))

    ),

    original_source,

    count=1,

)



TEMP_SCRIPT.write_text(patched_source, encoding="utf-8")



print("Entrada fixa:", INPUT_FILE)

print("Script original preservado:", SCRIPT)

print("Cópia temporária:", TEMP_SCRIPT)

print("Monitoramento GPU:", "disponível" if GPU_AVAILABLE else "indisponível")



# ------------------------------------------------------------

# PREPARAR EXECUÇÃO E LOGS

# ------------------------------------------------------------



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_path = LOG_DIR / f"C_{timestamp}.log"

metrics_path = LOG_DIR / f"C_metrics_{timestamp}.csv"

summary_path = LOG_DIR / f"C_summary_{timestamp}.json"



before_outputs = {}



for path in OUTPUT_DIR.glob("*_Cached_CE_Scores.json"):

    try:

        before_outputs[str(path.resolve())] = path.stat().st_mtime_ns

    except OSError:

        pass



monitor = MachineMonitor()

start_time = time.time()

start_iso = datetime.now().isoformat()



return_code = None

error = None

interrupted = False



old_stdout = sys.stdout

old_stderr = sys.stderr



# Adicionar todos os diretórios necessários aos imports.

# Em especial, cross_encoder_selector.py fica em src/IQLoc.

added_paths = []



for path in [

    ROOT,

    ROOT / "src",

    ROOT / "src" / "IQLoc",

    ROOT / "scripts",

]:

    path_str = str(path)

    if path_str not in sys.path:

        sys.path.insert(0, path_str)

        added_paths.append(path_str)



try:

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:

        sys.stdout = Tee(old_stdout, log)

        sys.stderr = Tee(old_stderr, log)



        try:

            monitor.start()



            print("\n" + "=" * 72)

            print("INICIANDO STAGE C")

            print("Selecione o modelo e o experimento nos menus originais.")

            print("=" * 72 + "\n", flush=True)



            runpy.run_path(str(TEMP_SCRIPT), run_name="__main__")

            return_code = 0



        except KeyboardInterrupt:

            interrupted = True

            error = "Execução interrompida pelo usuário."

            return_code = 130

            print("\nExecução interrompida pelo usuário.", flush=True)



        except Exception as exc:

            return_code = 1

            error = f"{type(exc).__name__}: {exc}"

            print(f"\nERRO NA EXECUÇÃO: {error}", flush=True)



        finally:

            monitor.stop()

            sys.stdout.flush()

            sys.stderr.flush()

            sys.stdout = old_stdout

            sys.stderr = old_stderr



except Exception as exc:

    if error is None:

        error = f"{type(exc).__name__}: {exc}"

    return_code = 1

    sys.stdout = old_stdout

    sys.stderr = old_stderr

    print("Erro ao preparar ou registrar a execução:", error)



finally:

    TEMP_SCRIPT.unlink(missing_ok=True)



# ------------------------------------------------------------

# DETECTAR SAÍDA CRIADA OU MODIFICADA

# ------------------------------------------------------------



changed_outputs = []



for path in OUTPUT_DIR.glob("*_Cached_CE_Scores.json"):

    try:

        resolved = str(path.resolve())

        mtime = path.stat().st_mtime_ns



        if (

            resolved not in before_outputs

            or mtime != before_outputs[resolved]

        ):

            changed_outputs.append(path)

    except OSError:

        continue



output_path = (

    max(changed_outputs, key=lambda path: path.stat().st_mtime_ns)

    if changed_outputs else None

)



output_exists = output_path is not None and output_path.is_file()

output_size = output_path.stat().st_size if output_exists else 0



duration = round(time.time() - start_time, 2)

metrics = monitor.summary()



with monitor.lock:

    samples = list(monitor.samples)



if samples:

    fields = sorted({key for row in samples for key in row})



    with open(metrics_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fields)

        writer.writeheader()

        writer.writerows(samples)



record = {

    "input_file": str(INPUT_FILE),

    "original_script": str(SCRIPT),

    "temporary_script_removed": not TEMP_SCRIPT.exists(),

    "output_file": str(output_path) if output_path else None,

    "output_exists": output_exists,

    "output_size_bytes": output_size,

    "log_file": str(log_path),

    "metrics_file": str(metrics_path),

    "start_time": start_iso,

    "end_time": datetime.now().isoformat(),

    "duration_seconds": duration,

    "return_code": return_code,

    "interrupted": interrupted,

    "result": (

        "success"

        if return_code == 0 and error is None

        and output_exists and output_size > 0

        else "failed_or_output_missing"

    ),

    "error": error,

    **metrics,

}



summary_path.write_text(

    json.dumps(record, indent=2, ensure_ascii=False),

    encoding="utf-8",

)



# ------------------------------------------------------------

# RELATÓRIO FINAL

# ------------------------------------------------------------



print("\n" + "=" * 72)

print("EXECUÇÃO FINALIZADA")

print("Resultado:", record["result"])

print("Código de saída:", return_code)

print("Duração (s):", duration)

print("Amostras de monitoramento:", metrics.get("samples"))

print("CPU média (%):", metrics.get("cpu_percent_avg"))

print("CPU máxima (%):", metrics.get("cpu_percent_max"))

print("RAM média (MB):", metrics.get("ram_used_mb_avg"))

print("RAM máxima (MB):", metrics.get("ram_used_mb_max"))

print("Disco usado máximo (%):", metrics.get("disk_used_percent_max"))

print("GPU média (%):", metrics.get("gpu_percent_avg"))

print("VRAM máxima (MB):", metrics.get("vram_used_mb_max"))

print("Saída detectada:", output_path)

print("Saída existe:", output_exists)

print("Tamanho da saída (bytes):", output_size)

print("Erro:", error)

print("Log:", log_path)

print("Métricas CSV:", metrics_path)

print("Resumo JSON:", summary_path)

print("=" * 72)
