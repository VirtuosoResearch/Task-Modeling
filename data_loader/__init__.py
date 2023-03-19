from data_loader.load_data_fns import *
from data_loader.load_ws_data_fns import *
from data_loader.collators import *
from transformers import default_data_collator

task_to_benchmark = {
    # GLUE
    "cola": "glue",
    "mnli": "glue",
    "mrpc": "glue",
    "qnli": "glue",
    "qqp":  "glue",
    "rte":  "glue",
    "sst2": "glue",
    "stsb": "glue",
    "wnli": "glue",
    # SuperGLUE excluding "rte": ("premise", "hypothesis"), since it is the same as in GLUE
    "boolq":   "super_glue",
    "cb":      "super_glue", 
    "wic":     "super_glue", 
    "copa":    "super_glue",
    "multirc": "super_glue",
    "wsc":     "super_glue",
    # "record": not implemented
    # Tweet Eval
    'emoji':           "tweet_eval",
    'emotion':         "tweet_eval",
    'hate':            "tweet_eval",
    'irony':           "tweet_eval",
    'offensive':       "tweet_eval",
    'sentiment':       "tweet_eval",
    'stance_abortion': "tweet_eval",
    'stance_atheism':  "tweet_eval",
    'stance_climate':  "tweet_eval",
    'stance_feminist': "tweet_eval",
    'stance_hillary':  "tweet_eval",
    # NLI Tasks
    "anli_r1": "anli",
    "anli_r2": "anli",
    "anli_r3": "anli",
}

task_to_collator = {
    # GLUE
    "cola": default_data_collator,
    "mnli": default_data_collator,
    "mrpc": default_data_collator,
    "qnli": default_data_collator,
    "qqp":  default_data_collator,
    "rte":  default_data_collator,
    "sst2": default_data_collator,
    "stsb": default_data_collator,
    "wnli": default_data_collator,
    # SuperGLUE excluding "rte": ("premise", "hypothesis"), since it is the same as in GLUE
    "boolq": default_data_collator,
    "cb":    default_data_collator, 
    "wic":   default_data_collator, 
    "copa":  default_data_collator,
    "multirc": multirc_collator_fn,
    "wsc": default_data_collator,
    # "record": not implemented
    # Tweet Eval
    'emoji':     default_data_collator,
    'emotion':   default_data_collator,
    'hate':      default_data_collator,
    'irony':     default_data_collator,
    'offensive': default_data_collator,
    'sentiment': default_data_collator,
    'stance_abortion': default_data_collator,
    'stance_atheism':  default_data_collator,
    'stance_climate':  default_data_collator,
    'stance_feminist': default_data_collator,
    'stance_hillary':  default_data_collator,
    # NLI Tasks
    "anli_r1": default_data_collator,
    "anli_r2": default_data_collator,
    "anli_r3": default_data_collator,
    # Weak supervision tasks
    "youtube": default_data_collator,
    "trec": default_data_collator,
    "cdr": default_data_collator,
    "chemprot": default_data_collator,
    "sms": default_data_collator,
    "semeval": default_data_collator,
}

# Weak supervision tasks
task_to_collator.update(
    {f"youtube_{i}": default_data_collator for i in range(11)}
)
task_to_collator.update(
    {f"trec_{i}": default_data_collator for i in range(69)}
)

task_to_collator.update(
    {f"cdr_{i}": default_data_collator for i in range(34)}
)
task_to_collator.update(
    {f"chemprot_{i}": default_data_collator for i in range(27)}
)
task_to_collator.update(
    {f"sms_{i}": default_data_collator for i in range(74)}
)
task_to_collator.update(
    {f"semeval_{i}": default_data_collator for i in range(165)}
)

task_to_load_fns = {
    # GLUE
    "cola": load_classification_task,
    "mnli": load_classification_task,
    "mrpc": load_classification_task,
    "qnli": load_classification_task,
    "qqp":  load_classification_task,
    "rte":  load_classification_task,
    "sst2": load_classification_task,
    "stsb": load_classification_task,
    "wnli": load_classification_task,
    # SuperGLUE excluding "rte": ("premise", "hypothesis"), since it is the same as in GLUE
    "boolq": load_classification_task,
    "cb":    load_classification_task, 
    "wic":   load_wic_task, 
    "copa":  load_copa_task,
    "multirc": load_multirc_task,
    "wsc": load_wsc_task,
    # "record": not implemented
    # Tweet Eval
    'emoji':     load_classification_task,
    'emotion':   load_classification_task,
    'hate':      load_classification_task,
    'irony':     load_classification_task,
    'offensive': load_classification_task,
    'sentiment': load_classification_task,
    'stance_abortion': load_classification_task,
    'stance_atheism':  load_classification_task,
    'stance_climate':  load_classification_task,
    'stance_feminist': load_classification_task,
    'stance_hillary':  load_classification_task,
    # NLI Tasks
    "anli_r1": load_anli_task,
    "anli_r2": load_anli_task,
    "anli_r3": load_anli_task,
}

ws_task_to_load_fns = {
    # Weak supervision tasks
    "youtube": load_ws_sentence_classification_task,
    "trec": load_ws_sentence_classification_task,
    "cdr": load_ws_span_classification_task,
    "chemprot": load_ws_span_classification_task,
    "sms": load_ws_sentence_classification_task,
    "semeval": load_ws_span_classification_task,
}