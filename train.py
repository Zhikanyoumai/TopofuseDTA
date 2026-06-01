import os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_default_config, parse_args
from models import LitTopoFuseDTA
from data import DTADataModule
from utils import save_test_results


def main():
    args = parse_args()
    cfg = get_default_config()
    cfg["dataset_name"] = args.dataset
    cfg["batch_size"] = args.batch_size
    cfg["lr"] = args.lr
    cfg["criterion"] = nn.MSELoss()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(
        save_dir=log_dir, name="topofuse", version=0,
        default_hp_metric=False)

    ckpt_dir = logger.log_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="valid_mse", dirpath=ckpt_dir,
        filename="topofuse-{epoch:03d}-{valid_mse:.4f}"
                 "-{valid_pearsonr:.4f}-{valid_r2:.4f}-{valid_ci:.4f}",
        save_top_k=3, mode="min", save_last=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    gpu_available = torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if gpu_available else "cpu",
        devices=[args.gpu] if gpu_available else None,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=10)

    model = LitTopoFuseDTA(**cfg)
    dm = DTADataModule(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size)
    dm.setup()

    resume = os.path.join(ckpt_dir, "last.ckpt")
    can_resume = False
    if os.path.exists(resume):
        try:
            ckpt_state = torch.load(resume, map_location="cpu")
            ck = set(ckpt_state.get("state_dict", {}).keys())
            mk = set(model.state_dict().keys())
            miss = mk - ck
            extra = ck - mk
            mismatch = [
                k for k in mk & ck
                if model.state_dict()[k].shape
                != ckpt_state["state_dict"][k].shape]
            if miss or extra or mismatch:
                os.remove(resume)
                for f in os.listdir(ckpt_dir):
                    if f.endswith(".ckpt") and f != "last.ckpt":
                        os.remove(os.path.join(ckpt_dir, f))
            else:
                can_resume = True
            del ckpt_state
        except Exception:
            pass

    if can_resume:
        trainer.fit(
            model, dm.train_dataloader(), dm.val_dataloader(),
            ckpt_path=resume)
    else:
        trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    best = ckpt_cb.best_model_path
    print(f"Best checkpoint: {best}")
    model = LitTopoFuseDTA.load_from_checkpoint(best, **cfg)

    pl.Trainer(
        accelerator="gpu" if gpu_available else "cpu",
        devices=[args.gpu] if gpu_available else None,
        logger=logger
    ).test(model, dm.test_dataloader())

    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in dm.test_dataloader():
            y, fp, lg, pg, ds, ps = batch
            if gpu_available:
                fp = fp.cuda(args.gpu)
                lg = lg.cuda(args.gpu)
                pg = pg.cuda(args.gpu)
                ds = ds.cuda(args.gpu)
                ps = ps.cuda(args.gpu)
            preds.extend(
                model(fp, lg, pg, ds, ps).cpu().numpy().flatten())
            targets.extend(y.numpy().flatten())

    save_test_results(
        np.array(preds), np.array(targets),
        args.dataset,
        os.path.join(logger.log_dir, "test_results"))


if __name__ == "__main__":
    main()
