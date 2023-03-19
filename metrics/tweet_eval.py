# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tweet Eval benchmark metric. """

from sklearn.metrics import f1_score, recall_score

import datasets


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_KWARGS_DESCRIPTION = """
"""


def macro_f1(preds, labels, target_labels = None):
    f1_scores = f1_score(y_true=labels, y_pred=preds, average=None)
    if target_labels is None:
        return f1_scores.mean()
    else:
        return f1_scores[target_labels].mean()


def macro_recall(preds, labels, target_labels = None):
    recall_scores = recall_score(y_true=labels, y_pred=preds, average=None)
    if target_labels is None:
        return recall_scores.mean()
    else:
        return recall_scores[target_labels].mean()


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TweetEval(datasets.Metric):
    def _info(self):
        if self.config_name not in [
            'emoji',
            'emotion',
            'hate',
            'irony',
            'offensive',
            'sentiment',
            'stance_abortion',
            'stance_atheism',
            'stance_climate',
            'stance_feminist',
            'stance_hillary',
        ]:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["emoji", "emotion", "hate", "irony", "offensive", "sentiment", '
                '"stance_abortion", "stance_atheism", "stance_climate", "stance_feminist", "stance_hillary"]'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64"),
                    "references": datasets.Value("int64"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references):
        if self.config_name in ["emoji", "emotion", "hate", "offensive"]:
            return {"macro_f1": macro_f1(references, predictions)}
        elif self.config_name == "irony":
            return {"macro_f1i": macro_f1(predictions, references, target_labels=[1])}
        elif self.config_name == "sentiment":
            return {"macro_recall": macro_recall(predictions, references)}
        elif self.config_name in ["stance_abortion", "stance_atheism", "stance_climate", "stance_feminist", "stance_hillary"]:
            return {"macro_f1a_f1f": macro_f1(predictions, references, target_labels=[1,2])}
        else:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["emoji", "emotion", "hate", "irony", "offensive", "sentiment", '
                '"stance_abortion", "stance_atheism", "stance_climate", "stance_feminist", "stance_hillary"]'
            )