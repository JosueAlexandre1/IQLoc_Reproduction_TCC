import json
import requests
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm


class CrossEncoderScorer:
    def __init__(self, model_path='/home/ams/projects/def-mrdal22/ams/Model_Trained/CrossEncoder/CodeBert'):

        self.model = CrossEncoder(model_path)
        # self.model.to(device)

    def ce_score(self, results_dictionary_array, bug_title, bug_description):

        for result in tqdm(results_dictionary_array, total=len(results_dictionary_array), desc="Processing Results"):
            # check if methods key exists
            if 'methods' not in result.keys():
                continue
            methods = result['methods']

            if isinstance(methods, list):
                for method in methods:
                    if not isinstance(method, dict):
                        continue

                    method_body = method.get('method_body')
                    if not isinstance(method_body, str) or method_body.strip() == '':
                        continue

                    scores = self.model.predict([[method_body, bug_title + ' . ' + bug_description]])
                    method['ce_score'] = float(scores[0])

            elif isinstance(methods, dict):
                # iterate over methods dictionary and get key-value pairs
                for method_name, method_body in methods.items():
                    if isinstance(method_body, dict):
                        body_text = method_body.get('method_body')
                        if not isinstance(body_text, str) or body_text.strip() == '':
                            continue

                        scores = self.model.predict([[body_text, bug_title + ' . ' + bug_description]])
                        method_body['ce_score'] = float(scores[0])
                    else:
                        if not isinstance(method_body, str) or method_body.strip() == '':
                            continue

                        scores = self.model.predict([[method_body, bug_title + ' . ' + bug_description]])
                        methods[method_name] = {
                            'method_body': method_body,
                            'ce_score': float(scores[0]),
                        }

        return results_dictionary_array


