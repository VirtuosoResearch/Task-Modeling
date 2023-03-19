import os 
os.environ['MKL_THREADING_LAYER'] = "GNU"
import argparse
import numpy as np

task_to_num_lfs = {
    "youtube": 10,
    "trec": 68,
    "cdr": 33,
    "chemprot": 26,
    "sms": 73,
    "semeval": 164
}

def main(args):
    task_list = [str(i) for i in range(1, task_to_num_lfs[args.ws_task_name]+1)]

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
            tmp_task_num = np.random.randint(
                low=max(min_task_num, 0), 
                high=max_task_num+1
            )
            tmp_sampled_tasks = np.random.choice(task_list, size=tmp_task_num, replace=False)            
            tmp_sampled_tasks.sort()
            tmp_sampled_tasks = " ".join(tmp_sampled_tasks)
            
            if tmp_sampled_tasks in sampled_tasks:
                continue
            
            os.system("python train_multitask_ws.py --use_ws_dataset --use_one_predhead\
                    --ws_task_name {} --lf_idxes {} --model_name_or_path {}\
                    --monitor_mode {} --monitor_metric {} \
                    --lr {} --epochs {} --device {} --runs {} --save_name {} --downsample_frac {}".format(
                    args.ws_task_name, tmp_sampled_tasks, args.model_name_or_path,
                    args.monitor_mode, args.monitor_metric,
                    args.lr, args.epochs, args.device, args.runs, args.save_name, args.downsample_frac
            ))
            sampled_tasks.add(tmp_sampled_tasks)
            f.write(tmp_sampled_tasks + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws_task_name", type=str, default="all")
    parser.add_argument("--model_name_or_path", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--min_task_num", type=int, default=3)
    parser.add_argument("--max_task_num", type=int, default=3)
    parser.add_argument("--downsample_frac", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)

    parser.add_argument("--task_set_name", type=str, default="ws_sampled_tasks")
    parser.add_argument("--save_name", type=str, default="ws_sampled_tasks")

    parser.add_argument("--monitor_mode", type=str, default='max', choices=['min', 'max', 'off'])
    parser.add_argument("--monitor_metric", type=str, default='val_youtube_0_f1')
    args = parser.parse_args()
    main(args)