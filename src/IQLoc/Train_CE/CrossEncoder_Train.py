import os
import math
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from torch.utils.data import DataLoader
import src.Utils.IO.JSON_File_IO as JSON_File_IO


def load_dataframe(file_path):
    return JSON_File_IO.load_JSON_to_Dataframe(file_path)


def load_diff_content(base_data_path, project, sub_project, bug_id, fixed_files):
    diff_content = ''
    diff_path = os.path.join(base_data_path, 'Diffs_FULL')

    file_path = os.path.join(diff_path, project, sub_project, bug_id, fixed_files[0])
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            diff_content = f.read()
    except Exception as e:
        print(f"Error reading file: {file_path}")
        print(e)
        # continue

    return diff_content


def prepare_train_samples(df, base_data_path):
    samples = []

    for index, row in df.iterrows():
        bug_id = row['bug_id']
        project = row['project']
        sub_project = row['sub_project']
        bug_title = row['bug_title']
        bug_description = row['bug_description']
        fixed_files = row['fixed_files']
        label = row['label']

        diff_content = load_diff_content(base_data_path, project, sub_project, bug_id, fixed_files)

        if diff_content != '':
            samples.append(InputExample(texts=[diff_content, bug_title + ' . ' + bug_description], label=label))

    return samples


def train_cross_encoder(model_path, train_samples, model_save_path, num_epochs=5, train_batch_size=8,
                        eval_samples=None):
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    model = CrossEncoder(model_path, num_labels=1, max_length=512)

    if eval_samples is not None:
        evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(eval_samples)
        model.fit(train_dataloader=train_dataloader,
                  evaluator=evaluator,
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path)
    else:
        model.fit(train_dataloader=train_dataloader,
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path)

    model.save(model_save_path)


if __name__ == "__main__":
    base_data_path = '/home/ams/projects/def-mrdal22/ams/Projects/QueryReformulation_LLM/Dataset/CrossEncoder'
    train_pair_path = os.path.join(base_data_path, 'train_pos_neg_pair_full_Timed.json')

    df = load_dataframe(train_pair_path)
    train_samples = prepare_train_samples(df, base_data_path)


    model_path = 'microsoft/codebert-base'

    model_save_path = '/home/ams/projects/def-mrdal22/ams/Model_Trained/CrossEncoder/CodeBert_Full_DS_Timed'
    number_of_epochs = 5
    batch_size = 4
    train_cross_encoder(model_path, train_samples, model_save_path, number_of_epochs, batch_size)
