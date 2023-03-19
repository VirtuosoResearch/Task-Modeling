from datasets import load_metric

task_to_metric_name = {
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
    'emoji':     "metrics/tweet_eval.py", 
    'emotion':   "metrics/tweet_eval.py", 
    'hate':      "metrics/tweet_eval.py", 
    'irony':     "metrics/tweet_eval.py", 
    'offensive': "metrics/tweet_eval.py", 
    'sentiment': "metrics/tweet_eval.py", 
    'stance_abortion': "metrics/tweet_eval.py", 
    'stance_atheism':  "metrics/tweet_eval.py", 
    'stance_climate':  "metrics/tweet_eval.py", 
    'stance_feminist': "metrics/tweet_eval.py", 
    'stance_hillary':  "metrics/tweet_eval.py", 
    # NLI Tasks
    "anli_r1": "metrics/anli.py",
    "anli_r2": "metrics/anli.py",
    "anli_r3": "metrics/anli.py",
    # 'esnli': ('premise', 'hypothesis', 'explanation_1', 'explanation_2', 'explanation_3')

    # Weak supervision tasks
    "youtube": "metrics/classification.py",
    "trec": "metrics/classification.py",
    "cdr": "metrics/classification.py",
    "chemprot": "metrics/classification.py",
    "sms": "metrics/classification.py",
    "semeval": "metrics/classification.py",
}