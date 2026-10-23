import gc

import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm


class CrossEncoderScorer:
    """
    Encapsula o modelo CrossEncoder usado para pontuar pares (method_body, bug_query).

    A LÓGICA DE SCORING É IDÊNTICA À VERSÃO ORIGINAL:
      - mesmo modelo / mesmo CrossEncoder
      - mesma query (bug_title + " . " + bug_description)
      - mesmo par enviado ao modelo: [method_body, query]
      - mesma ordem de processamento (métodos processados na ordem em que aparecem)
      - mesmo significado de 'ce_score'

    As únicas mudanças em relação à versão anterior são de gerenciamento de memória:
      - batch_size configurável e menor por padrão (16, ao invés de 32)
      - liberação explícita das estruturas temporárias de cada batch
      - gc.collect() e torch.cuda.empty_cache() chamados apenas por Result
        (não por método/batch, para não destruir a performance)
      - no ramo 'dict', iteramos diretamente sobre methods.items() ao invés de
        list(methods.keys()), evitando criar uma lista extra com milhares de chaves
    """

    def __init__(
        self,
        model_path='/home/ams/projects/def-mrdal22/ams/Model_Trained/CrossEncoder/CodeBert',
        batch_size=16,
        empty_cuda_cache_after_result=True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.empty_cuda_cache_after_result = empty_cuda_cache_after_result

        print(f"Usando dispositivo: {self.device}")
        print(f"CrossEncoder batch_size: {self.batch_size}")

        self.model = CrossEncoder(
            model_path,
            device=self.device
        )

    def _predict_batch(self, pairs):
        return self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

    def _flush_batch(self, batch_pairs, batch_targets, apply_fn):
        """
        Executa a inferência sobre o batch atual, aplica os scores via apply_fn
        e em seguida esvazia (in-place) as listas do batch, para que nenhuma
        estrutura proporcional ao total de métodos do Result seja mantida viva.
        """
        if not batch_pairs:
            return

        scores = self._predict_batch(batch_pairs)

        for target, score in zip(batch_targets, scores):
            apply_fn(target, float(score))

        del scores
        batch_pairs.clear()
        batch_targets.clear()

    def ce_score_result(self, result, bug_title, bug_description):
        """
        Processa APENAS UM result por vez.
        Mantém o comportamento/semântica original; muda apenas o gerenciamento de memória.
        """

        if 'methods' not in result:
            return result

        query = bug_title + " . " + bug_description
        methods = result['methods']

        if isinstance(methods, list):

            batch_pairs = []
            batch_methods = []

            def apply_list(method_obj, score):
                method_obj['ce_score'] = score

            for method in methods:

                if not isinstance(method, dict):
                    continue

                method_body = method.get('method_body')

                if not isinstance(method_body, str):
                    continue

                if not method_body.strip():
                    continue

                batch_pairs.append([method_body, query])
                batch_methods.append(method)

                if len(batch_pairs) >= self.batch_size:
                    self._flush_batch(batch_pairs, batch_methods, apply_list)

            # último batch parcial (menor que batch_size)
            self._flush_batch(batch_pairs, batch_methods, apply_list)

        elif isinstance(methods, dict):

            batch_pairs = []
            batch_info = []

            def apply_dict(info, score):
                info_type, method_name, original = info
                if info_type == "dict":
                    original['ce_score'] = score
                else:
                    # Substituir o valor de uma chave já existente NÃO invalida
                    # a iteração de methods.items() (só adicionar/remover chaves
                    # durante a iteração seria inseguro).
                    methods[method_name] = {
                        'method_body': original,
                        'ce_score': score
                    }

            for method_name, method_data in methods.items():

                if isinstance(method_data, dict):

                    method_body = method_data.get('method_body')

                    if not isinstance(method_body, str):
                        continue

                    if not method_body.strip():
                        continue

                    batch_pairs.append([method_body, query])
                    batch_info.append(("dict", method_name, method_data))

                elif isinstance(method_data, str):

                    if not method_data.strip():
                        continue

                    batch_pairs.append([method_data, query])
                    batch_info.append(("string", method_name, method_data))

                else:
                    continue

                if len(batch_pairs) >= self.batch_size:
                    self._flush_batch(batch_pairs, batch_info, apply_dict)

            self._flush_batch(batch_pairs, batch_info, apply_dict)

        # Limpeza: uma vez por Result, nunca por método/batch individual.
        gc.collect()
        if self.device == "cuda" and self.empty_cuda_cache_after_result:
            torch.cuda.empty_cache()

        return result

    def ce_score(self, results_dictionary_array, bug_title, bug_description):
        """
        Mantido por compatibilidade com código que ainda chame ce_score()
        recebendo a lista inteira de Results de um bug. Internamente delega
        para ce_score_result(), processando um Result por vez.
        """
        for result in tqdm(
            results_dictionary_array,
            total=len(results_dictionary_array),
            desc="Processing Results",
            leave=False,
        ):
            self.ce_score_result(
                result,
                bug_title=bug_title,
                bug_description=bug_description
            )

        return results_dictionary_array