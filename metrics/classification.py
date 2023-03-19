
from sklearn.metrics import accuracy_score, f1_score

import datasets
import numpy as np
from numpy import Inf

_DESCRIPTION = """
"""


_KWARGS_DESCRIPTION = """
"""


_CITATION = """
"""

def margin(logits, labels):
    correct_class_logits = logits[np.arange(logits.shape[0]), labels]

    incorrect_logits = np.copy(logits)
    incorrect_logits[np.arange(logits.shape[0]), labels] = -Inf
    highest_incorrect_logits = np.max(incorrect_logits, axis=1)
    margins = (correct_class_logits - highest_incorrect_logits)
    return margins.mean()

def macro_f1(preds, labels, target_labels = None, average = None):
    if average == "binary":
        return f1_score(y_true=labels, y_pred=preds, average=average)
    else:
        f1_scores = f1_score(y_true=labels, y_pred=preds, average=average)
        if target_labels is None:
            return f1_scores.mean()
        else:
            return f1_scores[target_labels].mean()

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SentenceClassification(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references, normalize=True, sample_weight=None, average=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            ),
            "f1": float(macro_f1(labels=references, preds=predictions, average=average))
        }