import os
import argparse
import hashlib
import json

import torch
torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model,evaluate,init_model
from pykt.utils import debug_print,set_seed
from pykt.datasets import init_dataset4train
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

OPERA_ENHANCE_PRO_MODELS = [
    "dkt_enhance_pro",
    "dkvmn_enhance_pro",
    "sakt_enhance_pro",
    "akt_enhance_pro_qid",
    "simplekt_enhance_pro_qid",
    "cgmkt",
]

OPERA_BATCH64_MODELS = [
    "dkvmn_enhance_pro",
    "sakt_enhance_pro",
    "akt_enhance_pro_qid",
    "simplekt_enhance_pro_qid",
]

OPERA_SEQ_LEN_MODELS = [
    "sakt_enhance_pro",
    "simplekt_enhance_pro_qid",
]

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

def main(params):
    if "use_wandb" not in params:
        params['use_wandb'] = 1

    if params['use_wandb']==1:
        import wandb
        wandb.init()

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
        
    debug_print(text = "load config files.",fuc_name="main")
    
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt", "akt_minimax", "akt_interfairauc", "akt_noisyauc", "akt_minimaxfairauc", "akt_dro", "robustkt", "folibikt", "atkt", "lpkt", "skvmn", "dimkt"] + OPERA_BATCH64_MODELS:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ("akt_pgdro", "akt_beshift"):
            train_config["batch_size"] = 64
        if model_name in ["simplekt","stablekt", "datakt", "sparsekt", "mtkt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        if model_name in ["dtransformer"]:
            train_config["batch_size"] = 32 ## because of OOM
        model_config = copy.deepcopy(params)
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']
        if model_name in ("akt_minimax", "akt_interfairauc", "akt_noisyauc", "akt_minimaxfairauc"):
            train_config["optimizer"] = "sgd"
            for key in ("minimax_prefix_length", "minimax_item_col", "init_checkpoint", "batch_size", "num_epochs"):
                model_config.pop(key, None)
        if model_name == "akt_pgdro":
            train_config["optimizer"] = "sgd"
        beshift_config = {}
        if model_name == "akt_beshift":
            for key in params:
                if key.startswith("beshift_"):
                    beshift_config[key[len("beshift_"):]] = model_config.pop(key)
        # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)
    
    debug_print(text="init_dataset",fuc_name="main")
    if model_name in ("akt_minimax", "akt_interfairauc", "akt_noisyauc", "akt_minimaxfairauc"):
        train_loader, valid_loader = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size,
            minimax_config={
                "prefix_length": params.get("minimax_prefix_length", 15),
                "item_col": params.get("minimax_item_col", "concepts"),
            },
        )
    elif model_name in ("akt_pgdro", "akt_beshift"):
        train_loader, valid_loader = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size,
            fairness_config={"seed": params["seed"], **beshift_config},
        )
    elif model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level)

    params_str = "_".join(
        "-".join(map(str, v)) if k == "noisy_gamma" else str(v) for k, v in params.items()
        if k != "other_config" and not (model_name in ("akt_minimax", "akt_interfairauc", "akt_noisyauc", "akt_minimaxfairauc") and k == "init_checkpoint")
    )
    if model_name in ("akt_minimax", "akt_interfairauc", "akt_noisyauc", "akt_minimaxfairauc") and params.get("init_checkpoint"):
        source = os.path.abspath(params["init_checkpoint"]).encode("utf-8")
        params_str += f"_warm_{hashlib.sha256(source).hexdigest()[:10]}"

    print(f"params: {params}, params_str: {params_str}")
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        # if not model_name in ['saint','saint++']:
        params_str = params_str+f"_{ str(uuid.uuid4())}"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    if model_name in ["dimkt"]:
        # del model_config['num_epochs']
        del model_config['weight_decay']

    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2','batch_size','num_epochs','weight_decay']:
        if remove_item in model_config:
            del model_config[remove_item]
    if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt","stablekt", "datakt","folibikt", "mtkt"] + OPERA_SEQ_LEN_MODELS:
        model_config["seq_len"] = seq_len
        
    debug_print(text = "init_model",fuc_name="main")
    print(f"model_name:{model_name}")
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type, dataset_name)
    print(f"model is {model}")
    if model_name in ("akt_minimax", "akt_interfairauc", "akt_noisyauc", "akt_minimaxfairauc"):
        if params.get("init_checkpoint"):
            state = torch.load(params["init_checkpoint"], map_location=device)
            state["log_pair_weights"] = model.log_pair_weights
            if model_name == "akt_noisyauc":
                state["noisy_radii"] = model.noisy_radii
                state["noisy_dual"] = model.noisy_dual
            model.load_state_dict(state)
        opt = SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif model_name == "akt_pgdro":
        opt = SGD(model.parameters(), lr=learning_rate)
    elif model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        print(f"dtransformer weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
    elif model_name == "cdfkt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=params.get("weight_decay", 1e-5))
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            if model_name in OPERA_ENHANCE_PRO_MODELS:
                opt = Adam(model.parameters(), learning_rate, weight_decay=1e-5)
            else:
                opt = Adam(model.parameters(), learning_rate)
   
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    
    debug_print(text = "train model",fuc_name="main")
    
    if model_name == "rkt":
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = \
            train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, data_config[dataset_name], fold)
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model)
    
    if save_model:
        best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type, dataset_name)
        net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"), map_location=device)
        best_model.load_state_dict(net)

    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
    model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
    print(f"end:{datetime.datetime.now()}")
    
    if params['use_wandb']==1:
        metrics = {"validauc": validauc, "validacc": validacc, "best_epoch": best_epoch, "model_save_path": model_save_path}
        if model_name == "akt_noisyauc":
            metrics["valid_noisy_violation"], metrics["valid_noisy_ranking_loss"] = model.best_minimax_loss
        elif model_name in ("akt_minimax", "akt_interfairauc", "akt_minimaxfairauc"):
            metric = "valid_interfair_loss" if model_name == "akt_interfairauc" else "valid_minimax_loss"
            metrics[metric] = model.best_minimax_loss
        wandb.log(metrics)
