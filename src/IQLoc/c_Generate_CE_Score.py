from datetime import datetime
from pathlib import Path
import sys
import re

from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from IQLoc.CE_Method_Scorer import CrossEncoderScorer
from Utils.IO import JSON_File_IO
from cross_encoder_selector import MODEL_NAME
from scripts.CrossEncoderConfig import CROSS_ENCODER_PREFIX

DEFAULT_INPUT_FILE = PROJECT_ROOT / "Output" / "Generation" / f"{CROSS_ENCODER_PREFIX}_Cached_BR_Keywords.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "Caching"
DEFAULT_OUTPUT_FILE_NAME = f"{CROSS_ENCODER_PREFIX}_Cached_CE_Scores.json"
DEFAULT_CE_MODEL = MODEL_NAME


def load_dataframe(file_path):
    return JSON_File_IO.load_JSON_to_Dataframe(file_path)


def calculate_IR_metrices(df, embedding_model=DEFAULT_CE_MODEL):
    ce_reranker = CrossEncoderScorer(model_path=embedding_model)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Scoring rows"):
        bug_title = row['bug_title']
        bug_description = row['bug_description']
        results_dictionary = row['Results']

        results_dictionary = ce_reranker.ce_score(
            results_dictionary,
            bug_title=bug_title,
            bug_description=bug_description)

        df.at[index, 'Results'] = results_dictionary

    return df


def main():
    cur_time = datetime.now()

    input_file_path = str(DEFAULT_INPUT_FILE)
    output_dir = str(DEFAULT_OUTPUT_DIR)
    output_file_name = DEFAULT_OUTPUT_FILE_NAME

    df = load_dataframe(input_file_path)

    updated_df = calculate_IR_metrices(df, embedding_model=DEFAULT_CE_MODEL)

    updated_reports = updated_df.to_dict(orient='records')
    JSON_File_IO.save_Dict_to_JSON(updated_reports, file_path=output_dir, file_name=output_file_name)

    print('Time taken: ', datetime.now() - cur_time)


if __name__ == "__main__":
    # start time
    start = datetime.now()
    print("Start time: ", start)

    main()

    # end time
    end = datetime.now()
    print("End time: ", end)
    # total time taken
    print(f"Runtime of the program is {end - start}")
