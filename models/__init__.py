from models.prediction_heads import *

task_to_prediction_heads = {
    # GLUE
    "cola": SequenceClassificationTaskHead,
    "mnli": SequenceClassificationTaskHead,
    "mrpc": SequenceClassificationTaskHead,
    "qnli": SequenceClassificationTaskHead,
    "qqp":  SequenceClassificationTaskHead,
    "rte":  SequenceClassificationTaskHead,
    "sst2": SequenceClassificationTaskHead,
    "stsb": SequenceClassificationTaskHead,
    "wnli": SequenceClassificationTaskHead,
    # SuperGLUE
    "boolq":   SequenceClassificationTaskHead,
    "cb":      SequenceClassificationTaskHead, 
    "wic":     SpanClassificationTaskHead,
    "copa":    MultipleChoiceTaskHead,
    "multirc": MultiRCTaskHead,
    "wsc":     SpanClassificationTaskHead,
    # "record": not_implemented
    # Tweet Eval
    'emoji':     SequenceClassificationTaskHead,
    'emotion':   SequenceClassificationTaskHead,
    'hate':      SequenceClassificationTaskHead,
    'irony':     SequenceClassificationTaskHead,
    'offensive': SequenceClassificationTaskHead,
    'sentiment': SequenceClassificationTaskHead,
    'stance_abortion': SequenceClassificationTaskHead,
    'stance_atheism':  SequenceClassificationTaskHead,
    'stance_climate':  SequenceClassificationTaskHead,
    'stance_feminist': SequenceClassificationTaskHead,
    'stance_hillary':  SequenceClassificationTaskHead,
    # NLI tasks
    "anli_r1": SequenceClassificationTaskHead,
    "anli_r2": SequenceClassificationTaskHead,
    "anli_r3": SequenceClassificationTaskHead,
    # Weak supervision tasks
    "youtube": SequenceClassificationTaskHead,
    "trec": SequenceClassificationTaskHead,
    "sms": SequenceClassificationTaskHead,
    "cdr": SpanClassificationTaskHead,
    "chemprot": SpanClassificationTaskHead,
    "semeval": SpanClassificationTaskHead,
}

# Weak supervision tasks
task_to_prediction_heads.update(
    {f"youtube_{i}": SequenceClassificationTaskHead for i in range(11)}
)
task_to_prediction_heads.update(
    {f"trec_{i}": SequenceClassificationTaskHead for i in range(69)}
)
task_to_prediction_heads.update(
    {f"cdr_{i}": SpanClassificationTaskHead for i in range(34)}
)
task_to_prediction_heads.update(
    {f"chemprot_{i}": SpanClassificationTaskHead for i in range(27)}
)
task_to_prediction_heads.update(
    {f"sms_{i}": SequenceClassificationTaskHead for i in range(74)}
)
task_to_prediction_heads.update(
    {f"semeval_{i}": SpanClassificationTaskHead for i in range(165)}
)