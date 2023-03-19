import os 
os.environ['MKL_THREADING_LAYER'] = "GNU"
import argparse
import numpy as np

def main(args):
    task_list = [
        'cola', 'mrpc', "rte", "sst2", "stsb", "wnli", # GLUE "mnli", "qnli", "qqp",
        "boolq", "cb", "wic", "copa", "multirc", "wsc", # SuperGLUE "record"
        'emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment', # TweetEval
        'stance_abortion', 'stance_atheism', 'stance_climate', 'stance_feminist', 'stance_hillary', # TweetEval
        "anli_r1", "anli_r2", # NLI tasks "anli_r3" 
        ]
    task_list = args.source_tasks if args.source_tasks != "all" else task_list

    target_tasks = args.target_tasks
    other_tasks = [task for task in task_list if task not in target_tasks]

    num_samples = args.num_samples
    max_task_num = args.max_task_num
    min_task_num = args.min_task_num
    for _ in range(num_samples):
        # create a set of trained task combinations
        sampled_task_dir = os.path.join("./sampled_tasks", "{}.txt".format(args.task_set_name))
        if not os.path.exists(sampled_task_dir):
            f = open(sampled_task_dir, "w")
            f.close()
            
        with open(sampled_task_dir, "r") as f:
            sampled_tasks = set()
            for line in f.readlines():
                sampled_tasks.add(line.rstrip("\n"))
            # print(sampled_tasks)

        # train on a new task combination
        with open(sampled_task_dir, "a") as f:
            tmp_target_task_num = np.random.randint(low=1, high=len(target_tasks)+1)
            tmp_sampled_target_tasks = np.random.choice(target_tasks, size=tmp_target_task_num, replace=False)

            tmp_other_task_num = np.random.randint(
                low=max(min_task_num-tmp_target_task_num, 0), 
                high=max_task_num-tmp_target_task_num+1
            )
            tmp_sampled_other_tasks = np.random.choice(other_tasks, size=tmp_other_task_num,replace=False)
            
            tmp_sampled_tasks = np.concatenate([tmp_sampled_target_tasks, tmp_sampled_other_tasks])
            tmp_sampled_tasks.sort()
            tmp_sampled_tasks = " ".join(tmp_sampled_tasks)
            
            if tmp_sampled_tasks in sampled_tasks:
                continue
            print(tmp_sampled_tasks)
            
            os.system("python train_multitask.py --task_names {} --model_name_or_path {}\
                    --device {} --runs {} --save_name {}".format(
                    tmp_sampled_tasks, args.model_name_or_path,
                    args.device, args.runs, args.save_name
            ))
            sampled_tasks.add(tmp_sampled_tasks)
            f.write(tmp_sampled_tasks + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_tasks", nargs='+', type=str, default="all")
    parser.add_argument("--target_tasks", nargs='+', type=str, default=["mrpc"])
    parser.add_argument("--model_name_or_path", type=str, default="prajjwal1/bert-mini")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--min_task_num", type=int, default=3)
    parser.add_argument("--max_task_num", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--task_set_name", type=str, default="sampled_tasks")
    parser.add_argument("--save_name", type=str, default="sampled_tasks")

    args = parser.parse_args()
    main(args)