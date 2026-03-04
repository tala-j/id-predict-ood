"""
Training (wandb removed)
Last Updated: 8/18
"""

import yaml, os, argparse, uuid, itertools, fcntl
import random
import numpy as np
import pandas as pd
import torch as t
t.set_float32_matmul_precision('high')
from torch.optim import SGD, Adam
from utils.data import BracketsDataset
from utils.model import get_transformer, GPT, N_CLASSES

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
CUTOFF = 0.9

class SimpleRun:
    def __init__(self, sweep_id=None):
        self.id = uuid.uuid4().hex[:8]
        self.sweep_id = sweep_id

class Trainer:
    def __init__(self, save_csv: str, config: dict, sweep_id=None,
                 train_inputs=None, train_seq_lens=None, train_targets=None,
                 indist_inputs=None, indist_seq_lens=None, indist_targets=None,
                 ood_inputs=None, ood_seq_lens=None, ood_targets=None):
        assert t.cuda.is_available(), "!!!CUDA not available!!!"

        self.run = SimpleRun(sweep_id=sweep_id)
        self.config = config

        # All data already on GPU
        self.train_inputs = train_inputs
        self.train_seq_lens = train_seq_lens
        self.train_targets = train_targets
        self.indist_inputs = indist_inputs
        self.indist_seq_lens = indist_seq_lens
        self.indist_targets = indist_targets
        self.ood_inputs = ood_inputs
        self.ood_seq_lens = ood_seq_lens
        self.ood_targets = ood_targets

        # set random seed for reproducible initialization
        if self.config.get("rdm_seed") is not None:
            self._set_seed(self.config["rdm_seed"])

        self.batch_size = self.config["batch_size"]
        self.loss_fn = t.nn.CrossEntropyLoss().cuda()

        self.model = self._initialize_model()
        self._get_optimizer()

        self.save_at = self._determine_save_points(len(train_inputs))
        self.model_dir = self._setup_model_dir(sweep_id is not None)
        self.save_csv = save_csv

    def _initialize_model(self):
        model = get_transformer(
            n_layer=self.config["n_layer"],
            n_head=self.config["n_head"],
            n_embd=self.config["n_embd"],
            embd_pdrop=self.config["embd_pdrop"],
            resid_pdrop=self.config["resid_pdrop"],
            attn_pdrop=self.config["attn_pdrop"])
        return t.compile(model)

    def _get_optimizer(self):
        if self.config["opt"] == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=self.config["lr"],
                                 weight_decay=self.config["wd"])
        elif self.config["opt"] == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config["lr"],
                                  weight_decay=self.config["wd"])
        else:
            raise ValueError("Optimizer not supported or specified")

    def _set_seed(self, rdm_seed: int):
        random.seed(rdm_seed)
        np.random.seed(rdm_seed)
        t.manual_seed(rdm_seed)
        t.cuda.manual_seed(rdm_seed)
        t.cuda.manual_seed_all(rdm_seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False

    def _determine_save_points(self, train_size):
        save_at = sorted(set([0] + [(i+1)*10000 for i in range(train_size//10000)] + [train_size]))
        return save_at

    def _setup_model_dir(self, is_sweep):
        model_dir = os.path.join(LOGS_DIR, "models", f"sweep_{self.run.sweep_id}" if is_sweep else "",
                                 f"run_{self.run.id}")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _get_preds(self, inputs: t.Tensor, seq_lens: t.Tensor) -> t.Tensor:
        outputs = self.model(inputs)[0]
        last_indices = (seq_lens - 1).unsqueeze(1).repeat(1, N_CLASSES)
        preds = outputs.gather(1, last_indices.unsqueeze(1))[:, 0, :]
        return preds

    def train_epoch(self, start: int, end: int) -> tuple[float, float]:
        self.model.train()
        total_loss = t.zeros(1, device='cuda')
        total_correct = t.zeros(1, device='cuda')
        total_samples = end - start

        for b in range(start, end, self.batch_size):
            b_end = min(b + self.batch_size, end)
            inputs = self.train_inputs[b:b_end]
            seq_lens = self.train_seq_lens[b:b_end]
            targets = self.train_targets[b:b_end].long()

            self.optimizer.zero_grad()
            preds = self._get_preds(inputs, seq_lens)
            loss = self.loss_fn(preds, targets)
            loss.backward()
            self.optimizer.step()

            with t.no_grad():
                total_loss += loss * (b_end - b)
                total_correct += ((preds[:, 1] > preds[:, 0]) == targets.bool()).sum()

        return (total_loss / total_samples).item(), (total_correct / total_samples).item()

    def train_model(self):
        self.model.cuda().train()

        print("progress,datapoints_seen,train_loss,train_acc,indist_acc,ood_acc", flush=True)
        for i in range(1, len(self.save_at)):
            train_loss, train_acc = self.train_epoch(self.save_at[i-1], self.save_at[i])
            _, indist_test_acc = self.evaluate(self.indist_inputs, self.indist_seq_lens, self.indist_targets)
            _, ood_test_acc = self.evaluate(self.ood_inputs, self.ood_seq_lens, self.ood_targets)

            print(f"[{i} of {len(self.save_at) - 1}],{self.save_at[i]},{train_loss:.3f},{train_acc:.3f},{indist_test_acc:.3f},{ood_test_acc:.3f}", flush=True)
            self._save_checkpoint(self.save_at[i], train_acc, indist_test_acc, ood_test_acc)

        t.save(self.model.state_dict(), os.path.join(self.model_dir, f"run_{self.run.id}_final.pt"))

    def _save_checkpoint(self, datapoints_seen, train_acc, indist_acc, ood_acc):
        to_save = pd.DataFrame([{"sweep_id": self.run.sweep_id, "run_id": self.run.id,
                                 **self.config, "datapoints_seen": datapoints_seen,
                                 "train_acc": train_acc, "indist_acc": indist_acc, "ood_acc": ood_acc}])
        df_path = os.path.join(LOGS_DIR, f"{self.save_csv}.csv")
        lock_path = df_path + ".lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if os.path.exists(df_path):
                df = pd.read_csv(df_path)
            else:
                df = pd.DataFrame()
            df = pd.concat([df, to_save], ignore_index=True)
            df.to_csv(df_path, index=False)

    def evaluate(self, inputs: t.Tensor, seq_lens: t.Tensor, targets: t.Tensor) -> tuple[float, float]:
        self.model.eval()
        with t.no_grad():
            preds = self._get_preds(inputs, seq_lens)
            tgts = targets.long()
            loss = self.loss_fn(preds, tgts).item()
            acc = ((preds[:, 1] > preds[:, 0]) == tgts.bool()).float().mean().item()
        return loss, acc


_HERE = os.path.dirname(os.path.abspath(__file__))

TRANSFORMER_SWEEP_CFG = os.path.join(_HERE, "transformer_sweep.yaml")
TRANSFORMER_CFG = os.path.join(_HERE, "transformer.yaml")

TRAIN_PATH = os.path.join(_HERE, "data/train_binomial(40,0.5).csv")
INDIST_PATH = os.path.join(_HERE, "data/in_dist_test_binomial(40,0.5).csv")
OOD_PATH = os.path.join(_HERE, "data/test_binomial(40,0.5).csv")
SAVE_CSV = "1Mexp2_bin_40_05_Transformer"


def dataset_to_gpu(dataset: BracketsDataset, device="cuda"):
    inputs = dataset.toks.to(device)
    seq_lens = t.tensor([len(s) + 2 for s in dataset.strs], device=device)
    targets = dataset.ylabels.to(device)
    return inputs, seq_lens, targets


def grid_configs(sweep_cfg: dict) -> list[dict]:
    params = sweep_cfg["parameters"]
    keys = list(params.keys())
    values = [params[k]["values"] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main(sweep: bool, sweep_id: str | None, worker_id: int, num_workers: int):
    print("Loading datasets...", flush=True)
    train_inputs, train_seq_lens, train_targets = dataset_to_gpu(BracketsDataset(pd.read_csv(TRAIN_PATH)))
    indist_inputs, indist_seq_lens, indist_targets = dataset_to_gpu(BracketsDataset(pd.read_csv(INDIST_PATH)))
    ood_inputs, ood_seq_lens, ood_targets = dataset_to_gpu(BracketsDataset(pd.read_csv(OOD_PATH)))
    print("Datasets loaded to GPU.", flush=True)

    if sweep:
        sweep_cfg = yaml.safe_load(open(TRANSFORMER_SWEEP_CFG, "r"))
        all_configs = grid_configs(sweep_cfg)
        my_configs = [c for i, c in enumerate(all_configs) if i % num_workers == worker_id]
        print(f"Worker {worker_id}/{num_workers}: sweep {sweep_id}, {len(my_configs)}/{len(all_configs)} runs", flush=True)
        for i, config in enumerate(my_configs):
            print(f"\n[worker {worker_id}] Run {i+1}/{len(my_configs)} | config: {config}", flush=True)
            Trainer(save_csv=SAVE_CSV, config=config, sweep_id=sweep_id,
                    train_inputs=train_inputs, train_seq_lens=train_seq_lens, train_targets=train_targets,
                    indist_inputs=indist_inputs, indist_seq_lens=indist_seq_lens, indist_targets=indist_targets,
                    ood_inputs=ood_inputs, ood_seq_lens=ood_seq_lens, ood_targets=ood_targets).train_model()
    else:
        config = yaml.safe_load(open(TRANSFORMER_CFG, "r"))
        Trainer(save_csv=SAVE_CSV, config=config,
                train_inputs=train_inputs, train_seq_lens=train_seq_lens, train_targets=train_targets,
                indist_inputs=indist_inputs, indist_seq_lens=indist_seq_lens, indist_targets=indist_targets,
                ood_inputs=ood_inputs, ood_seq_lens=ood_seq_lens, ood_targets=ood_targets).train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run transformer training with specified parameters.")
    parser.add_argument("--sweep", default=False, action="store_true", help="run a sweep")
    parser.add_argument("--sweep_id", type=str, default=None, help="shared sweep ID")
    parser.add_argument("--worker_id", type=int, default=0, help="index of this worker")
    parser.add_argument("--num_workers", type=int, default=1, help="total number of parallel workers")
    args = parser.parse_args()

    main(sweep=args.sweep, sweep_id=args.sweep_id, worker_id=args.worker_id, num_workers=args.num_workers)
