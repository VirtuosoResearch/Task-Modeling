from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
)

task_to_keys = {
    # GLUE
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    # SuperGLUE excluding "rte": ("premise", "hypothesis"), since it is the same as in GLUE
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"), 
    "wic": ("sentence1", "sentence2"), 
    "copa": ("premise", "choice1", "choice2", "question"),
    "multirc": ("paragraph", "question", "answer"),
    "record": ("passage", "query", "entities"), # not implemented
    "wsc": ("text", None),
    # Tweet Eval
    'emoji': ('text', None),
    'emotion': ('text', None),
    'hate': ('text', None),
    'irony': ('text', None),
    'offensive': ('text', None),
    'sentiment': ('text', None),
    'stance_abortion': ('text', None),
    'stance_atheism': ('text', None),
    'stance_climate': ('text', None),
    'stance_feminist': ('text', None),
    'stance_hillary': ('text', None),
    # NLI Tasks
    "anli_r1": ('premise', 'hypothesis'),
    "anli_r2": ('premise', 'hypothesis'),
    "anli_r3": ('premise', 'hypothesis'),
    # SQuAD
}

def load_anli_task(task_name, benchmark_name, config, tokenizer, pad_to_max_length=True, max_length=128):
    assert benchmark_name == 'anli'

    raw_datasets = load_dataset('anli')
    if task_name == "anli_r1":
        raw_datasets = {
            "train": raw_datasets["train_r1"],
            "validation": raw_datasets["dev_r1"],
            "test": raw_datasets['test_r1']
        }
    elif task_name == "anli_r2":
        raw_datasets = {
            "train": raw_datasets["train_r2"],
            "validation": raw_datasets["dev_r2"],
            "test": raw_datasets['test_r2']
        }
    elif task_name == "anli_r3":
        raw_datasets = {
            "train": raw_datasets["train_r3"],
            "validation": raw_datasets["dev_r3"],
            "test": raw_datasets['test_r3']
        }
    else:
        print("Non valid task name for ANLI benchmark.")
    num_labels = len(raw_datasets["train"].features["label"].names)

    sentence1_key, sentence2_key = task_to_keys[task_name]
    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        # In all cases, rename the column to labels because the model will expect that.
        result["labels"] = examples["label"]
        return result

    remove_columns = raw_datasets["train"].column_names

    train_dataset = raw_datasets["train"].map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )
    eval_dataset = raw_datasets["validation"].map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )
    test_dataset = raw_datasets["test"].map(
        preprocess_function, batched=True, remove_columns=remove_columns
    ) if 'test' in raw_datasets.keys() else None

    return train_dataset, eval_dataset, test_dataset, num_labels

def load_wsc_task(task_name, benchmark_name, config, tokenizer, pad_to_max_length=True, max_length=128):
    assert task_name == "wsc" and benchmark_name == "super_glue"

    raw_datasets = load_dataset(benchmark_name, task_name)
    num_labels = len(raw_datasets["train"].features["label"].names)

    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        examples_len = len(examples['text'])
        texts = [examples['text'][i].split() for i in range(examples_len)]
        span1_start_char = [
            len(" ".join(text[:examples['span1_index'][i]]))+1 if examples['span1_index'][i] != 0 else 0
            for i, text in enumerate(texts)]
        span1_end_char = [span1_start_char[i] + len(examples['span1_text'][i]) for i in range(examples_len)]

        span2_start_char = [
            len(" ".join(text[:examples['span2_index'][i]]))+1 if examples['span2_index'][i] != 0 else 0
            for i, text in enumerate(texts)]
        span2_end_char = [span2_start_char[i] + len(examples['span2_text'][i]) for i in range(examples_len)]

        texts = [" ".join(text) for text in texts]
        result = tokenizer(texts, padding=padding, max_length=max_length, truncation=True, return_offsets_mapping=True)

        offset_mapping = result["offset_mapping"]
        span1_masks = []
        span2_masks = []
        for i, offset in enumerate(offset_mapping):
            first_start_char = span1_start_char[i]
            first_end_char = span1_end_char[i]

            second_start_char = span2_start_char[i]
            second_end_char = span2_end_char[i]

            input_ids = result['input_ids'][i]
            span1_mask = np.array([0] * len(input_ids))
            span2_mask = np.array([0] * len(input_ids))
                
            context_start = 0
            context_end = 0
            while input_ids[context_end] != 102:
                context_end += 1
            context_end -= 1
            
            first_pos = [0, 0]; second_pos = [0, 0]
            # find first span pos
            idx = context_start
            while idx <= context_end and offset[idx][0] <= first_start_char:
                idx += 1
            first_pos[0] = idx - 1

            idx = context_end
            while idx >= context_start and offset[idx][1] >= first_end_char:
                idx -= 1
            first_pos[1] = idx + 1

            # find second span pos
            idx = context_start
            while idx <= context_end and offset[idx][0] <= second_start_char:
                idx += 1
            second_pos[0] = idx - 1

            idx = context_end
            while idx >= context_start and offset[idx][1] >= second_end_char:
                idx -= 1
            second_pos[1] = idx + 1
            
            span1_mask[first_pos[0]: first_pos[1]+1] = 1
            span2_mask[second_pos[0]: second_pos[1]+1] = 1

            span1_masks.append(span1_mask)
            span2_masks.append(span2_mask)
        result['span1_masks'] = span1_masks
        result['span2_masks'] = span2_masks
        result["labels"] = examples["label"]
        result.pop("offset_mapping")
        return result
    remove_columns = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"] if 'test' in raw_datasets.keys() else None

    return train_dataset, eval_dataset, test_dataset, num_labels

def load_record_task(task_name, benchmark_name, config, tokenizer, pad_to_max_length=True, max_length=512):
    assert task_name == "record" and benchmark_name == "super_glue"
    raw_datasets = load_dataset(benchmark_name, task_name)
    num_labels = -1

    padding = "max_length" if pad_to_max_length else False
    def preprocess_function(examples):
        examples_len = len(examples['query'])
        contexts = [" ".join([examples['passage'][i], tokenizer.sep_token, examples['query'][i]]) for i in range(examples_len)]
        
        num_choices = [len(examples['entities'][i]) for i in range(examples_len)]
        start_idx = [sum(num_choices[:i]) for i in range(len(num_choices))]
        
        first_sentences = [
            [contexts[i]]*num_choices[i] for i in range(examples_len)
        ]
        second_sentences = [
            examples['entities'][i] for i in range(examples_len)
        ]
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(first_sentences, second_sentences, padding=padding, max_length=max_length, truncation=True)
        result = {
            k: [v[start_idx[i]: start_idx[i]+num_choices[i]] 
                for i in range(len(num_choices))] for k, v in tokenized_examples.items()
            }
        result.update({"num_choices": num_choices})
        return result

    remove_columns = raw_datasets["train"].column_names
    remove_columns.remove("idx")

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"] if 'test' in raw_datasets.keys() else None

    return train_dataset, eval_dataset, test_dataset, num_labels

def load_multirc_task(task_name, benchmark_name, config, tokenizer, pad_to_max_length=True, max_length=128):
    assert task_name == "multirc" and benchmark_name == "super_glue"

    raw_datasets = load_dataset(benchmark_name, task_name)
    num_labels = len(raw_datasets["train"].features["label"].names)

    padding = "max_length" if pad_to_max_length else False
    def preprocess_function(examples):
        examples_len = len(examples['answer'])
        texts = [" ".join(
            [examples['paragraph'][i], tokenizer.sep_token, examples['question'][i], tokenizer.sep_token, examples['answer'][i]]
            ) for i in range(examples_len)]

        result = tokenizer(texts, padding=padding, max_length=max_length, truncation=True)
        result["labels"] = examples["label"]
        return result
    
    remove_columns = raw_datasets["train"].column_names
    remove_columns.remove("idx")
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"] if 'test' in raw_datasets.keys() else None

    return train_dataset, eval_dataset, test_dataset, num_labels

def load_copa_task(task_name, benchmark_name, config, tokenizer, pad_to_max_length=True, max_length=128):
    ''' Multiple choice task '''
    assert task_name == "copa" and benchmark_name == "super_glue"

    raw_datasets = load_dataset(benchmark_name, task_name)

    padding = "max_length" if pad_to_max_length else False
    num_labels = len(raw_datasets["train"].features["label"].names)
    
    choices_names = ["choice1", "choice2"]
    def preprocess_function(examples):
        examples_len = len(examples['premise'])
        contexts = [" ".join([examples['question'][i], tokenizer.sep_token, examples['premise'][i]]) for i in range(examples_len)]

        second_sentences = [[context]*2 for context in contexts]
        first_sentences = [[examples[choice_name][i] for choice_name in choices_names] for i in range(examples_len)]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(first_sentences, second_sentences, padding=padding, max_length=max_length, truncation=True)
        result = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
        result["labels"] = examples["label"]
        return result

    remove_columns = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"] if 'test' in raw_datasets.keys() else None

    return train_dataset, eval_dataset, test_dataset, num_labels

def load_wic_task(task_name, benchmark_name, config, tokenizer, pad_to_max_length=True, max_length=128):
    assert task_name == "wic" and benchmark_name == "super_glue"

    raw_datasets = load_dataset(benchmark_name, task_name)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    num_labels = len(raw_datasets["train"].features["label"].names)

    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )

        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True, return_offsets_mapping=True)

        first_start_chars = examples['start1']; first_end_chars = examples['end1']
        second_start_chars = examples['start2']; second_end_chars = examples['end2']
        offset_mapping = result["offset_mapping"]
        token_type_ids = result["token_type_ids"]

        span1_masks = []
        span2_masks = []
        for i, offset in enumerate(offset_mapping):            
            first_start_char = first_start_chars[i]
            first_end_char = first_end_chars[i]
            second_start_char = second_start_chars[i]
            second_end_char = second_end_chars[i]

            cur_token_type_ids = token_type_ids[i]

            input_ids = result['input_ids'][i]
            span1_mask = np.array([0] * len(input_ids))
            span2_mask = np.array([0] * len(input_ids))

            sentence1_start = 0
            sentence1_end = 0
            while cur_token_type_ids[sentence1_end] != 1:
                sentence1_end += 1
            sentence1_end -= 2

            sentence2_start = sentence1_end + 2
            sentence2_end = sentence1_end + 2
            while cur_token_type_ids[sentence2_end] != 0:
                sentence2_end += 1
            sentence2_end -= 2


            first_pos = [0, 0]; second_pos = [0, 0]
            # find first span pos
            idx = sentence1_start
            while idx <= sentence1_end and offset[idx][0] <= first_start_char:
                idx += 1
            first_pos[0] = idx - 1

            idx = sentence1_end
            while idx >= sentence1_start and offset[idx][1] >= first_end_char:
                idx -= 1
            first_pos[1] = idx + 1

            # find second span pos
            idx = sentence2_start
            while idx <= sentence2_end and offset[idx][0] <= second_start_char:
                idx += 1
            second_pos[0] = idx - 1

            idx = sentence2_end
            while idx >= sentence2_start and offset[idx][1] >= second_end_char:
                idx -= 1
            second_pos[1] = idx + 1
            
            span1_mask[first_pos[0]: first_pos[1]+1] = 1
            span2_mask[second_pos[0]: second_pos[1]+1] = 1

            span1_masks.append(span1_mask)
            span2_masks.append(span2_mask)

        result['span1_masks'] = span1_masks
        result['span2_masks'] = span2_masks
        result["labels"] = examples["label"]
        result.pop("offset_mapping")
        return result

    remove_columns = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"] if 'test' in raw_datasets.keys() else None

    return train_dataset, eval_dataset, test_dataset, num_labels

def load_classification_task(task_name, benchmark_name, config, tokenizer, pad_to_max_length=True, max_length=128):

    if task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(benchmark_name, task_name)

    # Labels
    if task_name is not None:
        is_regression = task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

    # Preprocessing the datasets
    if task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[task_name]

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            print(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    remove_columns = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]
    test_dataset = processed_datasets["test"] if 'test' in raw_datasets.keys() else None

    return train_dataset, eval_dataset, test_dataset, num_labels