from wrench.dataset import load_dataset
import wrench.labelmodel as labelmodel

from torch.utils.data.dataloader import Dataset, DataLoader
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
)

ws_methods = {
    "mv": labelmodel.MajorityVoting,
    "ds": labelmodel.DawidSkene,
    "dp": labelmodel.GenerativeModel,
    "metaL": labelmodel.Snorkel,
    "fs": labelmodel.FlyingSquid
}

ws_dataset_dir = "./ws_datasets"


class SpanClassificationDataset(Dataset):

    def __init__(self, ws_dataset, tokenizer, pad_to_max_length=True, max_length=128, label_fn_idx = 0, ws_method = "none", ws_params = {}):
        self.examples = ws_dataset.examples
        self.true_labels = np.array(ws_dataset.labels, dtype=np.long)
        ws_labels = np.array(ws_dataset.weak_labels, dtype=np.long)

        if ws_method == "none" or label_fn_idx != 0:
            self.label_fn_idx = label_fn_idx
            if label_fn_idx == 0:
                self.labels = self.true_labels
            else:
                assert label_fn_idx > 0 and label_fn_idx <= ws_labels.shape[1]
                self.labels = ws_labels[:, label_fn_idx-1]

            non_abstein_mask = (self.labels != -1)
            self.examples = [example for i, example in enumerate(self.examples) if non_abstein_mask[i]]
            self.true_labels = self.true_labels[non_abstein_mask]
            self.labels = self.labels[non_abstein_mask]
            # self.soft_labels = np.zeros_like(self.labels)
            # self.soft_labels[np.arange(self.labels.shape[0]), self.labels] = 1
        else:
            label_model = ws_methods[ws_method](**ws_params)
            label_model.fit(ws_dataset)
            self.labels = label_model.predict_proba(ws_dataset)
            # self.labels = np.argmax(self.soft_labels, axis=1)

        self.tokenizer = tokenizer
        self.padding = "max_length" if pad_to_max_length else False
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example = self.examples[index]
        texts = example['text']

        result = self.tokenizer(texts, padding = self.padding, max_length = self.max_length, truncation=True, return_offsets_mapping=True)
        offset_mapping = result["offset_mapping"]
        attention_mask = result["attention_mask"]

        first_start_char, first_end_char = example['span1']
        second_start_char, second_end_char = example['span2'] 

        sentence_start = 0
        sentence_end = 0
        while attention_mask[sentence_end] == 1:
            sentence_end += 1
        sentence_end -= 2

        first_pos = [0, 0]; second_pos = [0, 0]
        # find first span pos
        idx = sentence_start
        while idx <= sentence_end and offset_mapping[idx][0] <= first_start_char:
            idx += 1
        first_pos[0] = idx - 1

        idx = sentence_end
        while idx >= sentence_start and offset_mapping[idx][1] >= first_end_char:
            idx -= 1
        first_pos[1] = idx + 1

        # find second span pos
        idx = sentence_start
        while idx <= sentence_end and offset_mapping[idx][0] <= second_start_char:
            idx += 1
        second_pos[0] = idx - 1

        idx = sentence_end
        while idx >= sentence_start and offset_mapping[idx][1] >= second_end_char:
            idx -= 1
        second_pos[1] = idx + 1
        span1_mask = np.array([0] * len(attention_mask))
        span2_mask = np.array([0] * len(attention_mask))
        span1_mask[first_pos[0]: first_pos[1]+1] = 1
        span2_mask[second_pos[0]: second_pos[1]+1] = 1
        
        result['span1_masks'] = span1_mask
        result['span2_masks'] = span2_mask
        result["labels"] = self.labels[index]
        result.pop("offset_mapping")
        return result


class TextClassificationDataset(Dataset):

    def __init__(self, ws_dataset, tokenizer, pad_to_max_length=True, max_length=128, label_fn_idx = 0, ws_method = "none", ws_params = {}):
        self.examples = ws_dataset.examples
        self.true_labels = np.array(ws_dataset.labels, dtype=np.long)
        ws_labels = np.array(ws_dataset.weak_labels, dtype=np.long)

        if ws_method == "none" or label_fn_idx != 0:
            self.label_fn_idx = label_fn_idx
            if label_fn_idx == 0:
                self.labels = self.true_labels
            else:
                assert label_fn_idx > 0 and label_fn_idx <= ws_labels.shape[1]
                self.labels = ws_labels[:, label_fn_idx-1]

            non_abstein_mask = (self.labels != -1)
            self.examples = [example for i, example in enumerate(self.examples) if non_abstein_mask[i]]
            self.true_labels = self.true_labels[non_abstein_mask]
            self.labels = self.labels[non_abstein_mask]
            # self.soft_labels = np.zeros_like(self.labels)
            # self.soft_labels[np.arange(self.labels.shape[0]), self.labels] = 1
        else:
            label_model = ws_methods[ws_method](**ws_params)
            label_model.fit(ws_dataset)
            self.labels = label_model.predict_proba(ws_dataset)
            # self.labels = np.argmax(self.soft_labels, axis=1)

        self.tokenizer = tokenizer
        self.padding = "max_length" if pad_to_max_length else False
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        texts = self.examples[index]['text']
        result = self.tokenizer(texts, padding = self.padding, max_length = self.max_length, truncation=True)
        result['labels'] = self.labels[index]
        return result

def load_ws_sentence_classification_task(task_name, tokenizer, pad_to_max_length=True, max_length=128, ws_method = "none", ws_params = {}):
    assert task_name == "youtube" or task_name == "trec" or task_name == "sms"
    train_data, valid_data, test_data = load_dataset(ws_dataset_dir, task_name, extract_feature=False)

    num_labels = len(train_data.id2label)
    num_lfs = len(train_data.weak_labels[0])
    train_datasets = [TextClassificationDataset(train_data, tokenizer, pad_to_max_length=pad_to_max_length, max_length=max_length, label_fn_idx = i, ws_method = ws_method, ws_params = ws_params) for i in range(num_lfs+1)]
    valid_dataset = TextClassificationDataset(valid_data, tokenizer, pad_to_max_length=pad_to_max_length, max_length=max_length, label_fn_idx = 0)
    test_dataset = TextClassificationDataset(test_data, tokenizer, pad_to_max_length=pad_to_max_length, max_length=max_length, label_fn_idx = 0)
    return train_datasets, valid_dataset, test_dataset, num_lfs, num_labels

def load_ws_span_classification_task(task_name, tokenizer, pad_to_max_length=True, max_length=128, ws_method = "none", ws_params = {}):
    assert task_name == "cdr" or task_name == "chemprot" or task_name == "semeval"
    train_data, valid_data, test_data = load_dataset(ws_dataset_dir, task_name, extract_feature=False)

    num_labels = len(train_data.id2label)
    num_lfs = len(train_data.weak_labels[0])
    train_datasets = [SpanClassificationDataset(train_data, tokenizer, pad_to_max_length=pad_to_max_length, max_length=max_length, label_fn_idx = i, ws_method=ws_method, ws_params = ws_params) for i in range(num_lfs+1)]
    valid_dataset = SpanClassificationDataset(valid_data, tokenizer, pad_to_max_length=pad_to_max_length, max_length=max_length, label_fn_idx = 0)
    test_dataset = SpanClassificationDataset(test_data, tokenizer, pad_to_max_length=pad_to_max_length, max_length=max_length, label_fn_idx = 0)
    return train_datasets, valid_dataset, test_dataset, num_lfs, num_labels
