
import gc

import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm


class CrossEncoderScorer:
    def __init__(
        self,
        model_path,
        batch_size=16,
        max_length=None,
        fp16=False,
        empty_cuda_cache_after_result=False,
    ):
        self.model_path = model_path
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.batch_size = batch_size
        self.max_length = max_length
        self.empty_cuda_cache_after_result = (
            empty_cuda_cache_after_result
        )

        model_kwargs = {}

        # Precisão reduzida somente quando solicitada e usando GPU.
        if fp16 and self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        cross_encoder_kwargs = {
            "device": self.device,
            "model_kwargs": model_kwargs,
        }

        if max_length is not None:
            cross_encoder_kwargs["max_length"] = max_length

        print(f"Modelo: {model_path}")
        print(f"Dispositivo: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max length: {self.max_length}")
        print(f"FP16: {bool(model_kwargs)}")

        self.model = CrossEncoder(
            model_path,
            **cross_encoder_kwargs,
        )

    def _predict_batch(self, pairs):
        """
        Executa a inferência. Se houver falta de memória CUDA,
        tenta novamente com lotes progressivamente menores.
        """
        current_batch_size = min(
            self.batch_size,
            len(pairs),
        )

        while True:
            try:
                return self.model.predict(
                    pairs,
                    batch_size=current_batch_size,
                    show_progress_bar=False,
                )

            except torch.cuda.OutOfMemoryError:
                if self.device != "cuda":
                    raise

                if current_batch_size <= 1:
                    torch.cuda.empty_cache()
                    raise

                print(
                    "\nCUDA OOM: reduzindo batch size de "
                    f"{current_batch_size} para "
                    f"{max(1, current_batch_size // 2)}."
                )

                current_batch_size = max(
                    1,
                    current_batch_size // 2,
                )

                gc.collect()
                torch.cuda.empty_cache()

    def _flush_batch(self, batch_pairs, batch_targets, apply_fn):
        if not batch_pairs:
            return

        scores = self._predict_batch(batch_pairs)

        for target, score in zip(batch_targets, scores):
            apply_fn(target, float(score))

        del scores
        batch_pairs.clear()
        batch_targets.clear()

    def ce_score_result(self, result, bug_title, bug_description):
        if not isinstance(result, dict):
            return result

        if "methods" not in result:
            return result

        methods = result["methods"]

        query = (
            str(bug_title or "")
            + " . "
            + str(bug_description or "")
        )

        if isinstance(methods, list):
            batch_pairs = []
            batch_targets = []

            def apply_list(method_obj, score):
                method_obj["ce_score"] = score

            for method in methods:
                if not isinstance(method, dict):
                    continue

                method_body = method.get("method_body")

                if (
                    not isinstance(method_body, str)
                    or not method_body.strip()
                ):
                    continue

                # Mantém a ordem original dos pares.
                batch_pairs.append([method_body, query])
                batch_targets.append(method)

                if len(batch_pairs) >= self.batch_size:
                    self._flush_batch(
                        batch_pairs,
                        batch_targets,
                        apply_list,
                    )

            self._flush_batch(
                batch_pairs,
                batch_targets,
                apply_list,
            )

        elif isinstance(methods, dict):
            batch_pairs = []
            batch_info = []

            def apply_dict(info, score):
                info_type, method_name, original = info

                if info_type == "dict":
                    original["ce_score"] = score
                else:
                    methods[method_name] = {
                        "method_body": original,
                        "ce_score": score,
                    }

            for method_name, method_data in methods.items():
                if isinstance(method_data, dict):
                    method_body = method_data.get("method_body")

                    if (
                        not isinstance(method_body, str)
                        or not method_body.strip()
                    ):
                        continue

                    batch_pairs.append([method_body, query])
                    batch_info.append(
                        ("dict", method_name, method_data)
                    )

                elif isinstance(method_data, str):
                    if not method_data.strip():
                        continue

                    batch_pairs.append([method_data, query])
                    batch_info.append(
                        ("string", method_name, method_data)
                    )

                else:
                    continue

                if len(batch_pairs) >= self.batch_size:
                    self._flush_batch(
                        batch_pairs,
                        batch_info,
                        apply_dict,
                    )

            self._flush_batch(
                batch_pairs,
                batch_info,
                apply_dict,
            )

        gc.collect()

        if (
            self.device == "cuda"
            and self.empty_cuda_cache_after_result
        ):
            torch.cuda.empty_cache()

        return result

    def ce_score(
        self,
        results_dictionary_array,
        bug_title,
        bug_description,
    ):
        for result in tqdm(
            results_dictionary_array,
            total=len(results_dictionary_array),
            desc="Processing Results",
            leave=False,
        ):
            self.ce_score_result(
                result,
                bug_title=bug_title,
                bug_description=bug_description,
            )

        return results_dictionary_array
