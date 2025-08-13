import numpy as np
import torch
import pandas as pd
import os
import time
import argparse

import toml 
import json
from omegaconf import OmegaConf
from multiprocessing import Pool,Process
import subprocess
import platform
import torch.distributed as dist

from src.trainer.static_trainer import StaticTrainer
from src.trainer.sequential_trainer import SequentialTrainer

class FileParser:
    def __init__(self, filename):
        if filename.endswith(".toml"):
            with open(filename) as f:
                self.kwargs = OmegaConf.load(f)
        elif filename.endswith(".json"):
            with open(filename) as f:
                self.kwargs = OmegaConf.load(f)
        else:
            raise NotImplementedError(f"File type {filename} not supported, currently only toml and json are supported.")
        
    def add_argument(self, *args, **kwargs):

        for arg in args:
            if arg.startswith("--"):
                arg = arg[2:]
            if arg not in self.kwargs:
                if "action" in kwargs:
                    self.kwargs[arg] = False
                else:
                    self.kwargs[arg] = kwargs.get("default", None)
   
    def parse_args(self):
        return argparse.Namespace(**self.kwargs)

def parse_cmd():
    parser = argparse.ArgumentParser()
    return [parse_args(parser)], True

def parse_files():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None, help="config file path")
    parser.add_argument("-f", "--folder", type=str, default=None, help="folder path")
    parser.add_argument("--debug", action="store_true", help="debug mode, to dispalce multiprocessing")
    parser.add_argument("--num_works_per_device", type=int, default=10, help="number of works per device")
    parser.add_argument("--visible_devices", nargs='*', type=int, default=None, help="visible devices")
    args = parser.parse_args()
    assert args.config or args.folder, "Please specify --config or --folder"
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_devices))
    if args.config:
        args.arg_files = [args.config]
    else:
        args.arg_files = []
        for root, dirs, files in os.walk(args.folder):
            for name in files:
                if name.endswith(".toml") or name.endswith(".json"):
                    args.arg_files.append(os.path.join(root, name))
    return args

def prepare_arg(arg):
    # make sure all paths are exist
    basepath = os.path.dirname(os.path.abspath(__file__))
    for _path in ["ckpt_path", "loss_path", "result_path", "database_path"]:
        if os.path.isabs(arg.path[_path]):
            continue
        _abspath = os.path.join(basepath, arg.path[_path])
        _dirpath = os.path.dirname(_abspath)
        # make sure the path directory exist
        if not os.path.exists(_dirpath):
            os.makedirs(_dirpath)
        # turn the relative path to abs path 
        arg.path[_path] = _abspath
    arg.datarow = vars(arg).copy()
    arg.datarow['nbytes'] = -1
    arg.datarow['nparams'] = -1
    arg.datarow['p2r edges'] = -1
    arg.datarow['r2r edges'] = -1
    arg.datarow['r2p edges'] = -1
    arg.datarow['training time'] = np.nan
    arg.datarow['inference time'] = np.nan
    arg.datarow['time']    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    arg.datarow['relative error (direct)'] = np.nan
    arg.datarow['relative error (auto2)'] = np.nan
    arg.datarow['relative error (auto4)'] = np.nan
    
    return arg

def run_arg(arg):
    arg = prepare_arg(arg)

    Trainer = {
        "static": StaticTrainer,  
        "sequential": SequentialTrainer,  
    }[arg.setup["trainer_name"]]
    t = Trainer(arg)
    if arg.setup["train"]:
        if arg.setup["ckpt"]:
            t.load_ckpt()
        t.fit()
    if arg.setup["test"]:
        t.load_ckpt()
        t.test()

    if getattr(arg.setup, "rank", 0) == 0:
        if os.path.exists(arg.path["database_path"]):
            database = pd.read_csv(arg.path["database_path"])
        else:
            database = pd.DataFrame(columns=arg.datarow.keys())
        database.loc[len(database)] = arg.datarow
        database.to_csv(arg.path["database_path"], index=False)

    return t

def run_arg_file_popen_handle(arg_file):
    command = f"python main.py -c {arg_file}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode == 0:
        print(f"Job {arg_file}: {out.decode('utf-8').strip()}")
    else:
        print(f"Job {arg_file} error: {err.decode('utf-8').strip()}")

def run_arg_files(arg_files, is_debug, num_works_per_device=3):
    if len(arg_files) == 1:
        run_arg(FileParser(arg_files[0]).parse_args())
    elif is_debug:
        for arg_file in arg_files:
            print("\n")
            print(arg_file, end="\n\n\n")
            run_arg(parse_args(FileParser(arg_file)))
    elif platform.system() == "Windows":
        processes = []
        for arg_file in arg_files:
            arg = parse_args(FileParser(arg_file))
            p = Process(target=run_arg, args=(arg,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif platform.system() == "Linux":
        num_devices = torch.cuda.device_count()
        processes = {"cpu":[]}
        for i in range(num_devices):
            processes[f"cuda:{i}"] = []
        for arg_file in arg_files:
            arg = parse_args(FileParser(arg_file))
            p = Process(target=run_arg_file_popen_handle, args=(arg_file,))
            if arg.device.startswith("cuda"):
                device_id = int(arg.device[-1])
                processes[f"cuda:{device_id}"].append(p)
            else:
                processes["cpu"].append(p)
        
        max_jobs = max([len(v) for k,v in processes.items()])
        max_runs = (max_jobs + num_works_per_device - 1)  // num_works_per_device
        for i in range(max_runs):
            for k, v in processes.items():
                for p in v[i*num_works_per_device:(i+1)*num_works_per_device]:
                    p.start()
            for k, v in processes.items():
                for p in v[i*num_works_per_device:(i+1)*num_works_per_device]:
                    p.join()
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported")

def init_distributed_mode(arg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        arg.setup.rank = int(os.environ['RANK'])
        arg.setup.world_size = int(os.environ['WORLD_SIZE'])
        arg.setup.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        print('Not using distributed mode')
        arg.setup.distributed = False
        arg.setup.rank = 0
        arg.setup.world_size = 1
        arg.setup.local_rank = 0
        return

    dist.init_process_group(
        backend=arg.setup.backend,
        init_method='env://',
        world_size=arg.setup.world_size,
        rank=arg.setup.rank
    )
    dist.barrier()

if __name__ == '__main__':
    config = parse_files()
    run_arg_files(config.arg_files, config.debug, config.num_works_per_device)
