# ------------------------------------------------------------------------
# Methods used for training inertial-based models
# ------------------------------------------------------------------------
# Adapted for Gamma-Quant + continuous conv + per-class metric logging
# ------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils.data_utils import convert_samples_to_segments, unwindow_inertial_data
from utils.torch_utils import (
    init_weights,
    save_checkpoint,
    worker_init_reset_seed,
    InertialDataset,
)
from utils.os_utils import mkdir_if_missing
from models.DeepConvLSTM import DeepConvLSTM
from models.map_metric import ANETdetection


def run_inertial_network(
    train_sbjs,
    val_sbjs,
    cfg,
    ckpt_folder,
    ckpt_freq,
    resume,
    rng_generator,
    run,
    gamma_quant=None,
    quant_bits=4,
    gamma_type="id",
    apply_gamma="global",
):
    """
    Train and validate an inertial-based network (DeepConvLSTM).

    This version:
      * Works with standard and continuous conv DeepConvLSTM variants
      * Supports Gamma-Quant settings
      * Logs per-epoch metrics to stdout (and Neptune if provided)
      * Saves a *single* best-F1 checkpoint per split:
            logs/deepconvlstm/<freq_tag>/<split_name>/best_<split_name>.pth.tar
      * Saves per-class metrics (acc/prec/rec/F1) of the best-F1 epoch:
            logs/deepconvlstm/<freq_tag>/<split_name>/per_class_metrics_<split_name>.csv
      * Computes detection mAP using ANETdetection on sample-wise segments

    Returns
    -------
    t_losses : list[float]
        Training loss per epoch.
    v_losses : list[float]
        Validation loss per epoch.
    v_mAP_best : np.ndarray
        Vector of mAP values (one per tIoU threshold) for the best-F1 epoch.
    v_preds_out : np.ndarray
        Window-wise predictions on the validation set (last epoch).
    v_gt_out : np.ndarray
        Window-wise ground truth labels on the validation set (last epoch).
    net : nn.Module
        Trained network (after final epoch).
    """

    # --------------------------------------------------------------
    # Derive split name and frequency-specific log directory
    # --------------------------------------------------------------
    split_name = cfg["dataset"]["json_anno"].split("/")[-1].split(".")[0]
    sampling_rate = float(cfg["dataset"]["sampling_rate"])

    if abs(sampling_rate - 50.0) < 1e-3:
        freq_tag = "50hz"
    elif abs(sampling_rate - 25.0) < 1e-3:
        freq_tag = "25hz"
    elif abs(sampling_rate - 12.5) < 1e-3:
        freq_tag = "12_5hz"
    elif abs(sampling_rate - 6.0) < 1e-3:
        freq_tag = "6hz"
    else:
        freq_tag = f"{sampling_rate:g}hz"

    ckpt_folder = os.path.join("logs", "deepconvlstm", freq_tag, split_name)
    mkdir_if_missing(ckpt_folder)

    # --------------------------------------------------------------
    # Load inertial CSVs for train/val subjects
    # --------------------------------------------------------------
    train_data = np.empty((0, cfg["dataset"]["input_dim"] + 2))
    val_data = np.empty((0, cfg["dataset"]["input_dim"] + 2))

    for t_sbj in train_sbjs:
        t_path = os.path.join(cfg["dataset"]["sens_folder"], t_sbj + ".csv")
        t_df = pd.read_csv(t_path, index_col=False, low_memory=False)
        t_df = t_df.replace({"label": cfg["label_dict"]}).fillna(0)
        t_arr = t_df.to_numpy()
        train_data = np.append(train_data, t_arr, axis=0)

    for v_sbj in val_sbjs:
        v_path = os.path.join(cfg["dataset"]["sens_folder"], v_sbj + ".csv")
        v_df = pd.read_csv(v_path, index_col=False, low_memory=False)
        v_df = v_df.replace({"label": cfg["label_dict"]}).fillna(0)
        v_arr = v_df.to_numpy()
        val_data = np.append(val_data, v_arr, axis=0)

    # --------------------------------------------------------------
    # Normalization (based on training data) to [-1, 1]
    # --------------------------------------------------------------
    min_train = np.min(train_data[:, 1:-1])
    max_train = np.max(train_data[:, 1:-1])

    train_data[:, 1:-1] = 2 * (train_data[:, 1:-1] - min_train) / (max_train - min_train) - 1
    val_data[:, 1:-1] = 2 * (val_data[:, 1:-1] - min_train) / (max_train - min_train) - 1

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print("Min/ Max values in training data: [{:.2f}, {:.2f}]".format(
        np.min(train_data[:, 1:-1]), np.max(train_data[:, 1:-1])
    ))
    print("Min/ Max values in validation data: [{:.2f}, {:.2f}]".format(
        np.min(val_data[:, 1:-1]), np.max(val_data[:, 1:-1])
    ))

    # --------------------------------------------------------------
    # Datasets and dataloaders
    # --------------------------------------------------------------
    train_dataset = InertialDataset(
        train_data,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
        model=cfg["name"],
    )
    test_dataset = InertialDataset(
        val_data,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
        model=cfg["name"],
    )

    # Map class_id -> class_name for later CSV
    label_dict = cfg.get("label_dict", None)
    if label_dict is not None:
        class_id2name = {v: k for k, v in label_dict.items()}
    else:
        class_id2name = None

    if cfg["name"] in ["shallow_deepconvlstm", "deepconvcontext"]:
        print("DID NOT SHUFFLE (model {})".format(cfg["name"]))
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["loader"]["train_batch_size"],
            shuffle=False,
            num_workers=4,
            worker_init_fn=worker_init_reset_seed,
            generator=rng_generator,
            persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["loader"]["train_batch_size"],
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_reset_seed,
            generator=rng_generator,
            persistent_workers=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["loader"]["test_batch_size"],
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_reset_seed,
        generator=rng_generator,
        persistent_workers=True,
    )

    # --------------------------------------------------------------
    # Define network (DeepConvLSTM with optional continuous conv)
    # --------------------------------------------------------------
    if cfg["name"] != "deepconvlstm":
        raise ValueError(f"Unsupported model name '{cfg['name']}'")

    conv_type = cfg["model"].get("conv_type", "standard")
    conv_rank = cfg["model"].get("conv_rank", 8)
    conv_mlp_hidden_dim = cfg["model"].get("conv_mlp_hidden_dim", 32)

    net = DeepConvLSTM(
        train_dataset.channels,
        train_dataset.classes,
        train_dataset.window_size,
        cfg["model"]["conv_kernels"],
        cfg["model"]["conv_kernel_size"],
        cfg["model"]["lstm_units"],
        cfg["model"]["lstm_layers"],
        cfg["model"]["dropout"],
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
        conv_type=conv_type,
        conv_rank=conv_rank,
        conv_mlp_hidden_dim=conv_mlp_hidden_dim,
    )

    print(
        "Number of learnable parameters for DeepConvLSTM: {}".format(
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        )
    )
    print(net)

    # --------------------------------------------------------------
    # Device, criterion, optimizer, scheduler
    # --------------------------------------------------------------
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    if gpu is not None:
        net = net.to(gpu)

    # Base criterion
    criterion = nn.CrossEntropyLoss()

    # Optional class weighting
    if cfg["train_cfg"].get("weighted_loss", False):
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(train_dataset.labels), y=train_dataset.labels
        )
        print("Class weights: {}".format(class_weights))
        w_tensor = torch.FloatTensor(class_weights)
        if gpu is not None:
            w_tensor = w_tensor.to(gpu)
        criterion = nn.CrossEntropyLoss(weight=w_tensor)

    # Optimizer and LR schedule
    opt = torch.optim.Adam(
        net.parameters(),
        lr=cfg["train_cfg"]["lr"],
        weight_decay=cfg["train_cfg"]["weight_decay"],
    )
    scheduler = None
    if cfg["train_cfg"]["lr_step"] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=cfg["train_cfg"]["lr_step"],
            gamma=cfg["train_cfg"]["lr_decay"],
        )

    # --------------------------------------------------------------
    # Resume or initialise weights
    # --------------------------------------------------------------
    start_epoch = 0
    best_f1 = -1.0
    best_epoch = -1
    best_v_mAP_vec = None

    if resume is not None and resume != "":
        if os.path.isfile(resume):
            print(f"Loading checkpoint '{resume}'")
            checkpoint = torch.load(resume, map_location="cpu")
            start_epoch = checkpoint.get("epoch", 0)
            net.load_state_dict(checkpoint["state_dict"])
            opt.load_state_dict(checkpoint["optimizer"])
            best_f1 = checkpoint.get("best_f1", -1.0)
            print(f"Loaded checkpoint (epoch {start_epoch}, best_f1 {best_f1:.2f})")
        else:
            print(f"No checkpoint found at '{resume}'")
    else:
        # fresh init
        net = init_weights(net, cfg["train_cfg"]["weight_init"])

    # --------------------------------------------------------------
    # Detection evaluator (ActivityNet-style)
    # --------------------------------------------------------------
    det_eval = ANETdetection(
        cfg["dataset"]["json_anno"],
        "validation",
        tiou_thresholds=cfg["dataset"]["tiou_thresholds"],
    )

    t_losses = []
    v_losses = []
    v_mAP_all = []

    v_preds_out = None
    v_gt_out = None

    # --------------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------------
    for epoch in range(start_epoch, cfg["train_cfg"]["epochs"]):
        # ---------------- Training ----------------
        net.train()
        train_epoch_losses = []

        for i, (inputs, targets) in enumerate(train_loader):
            opt.zero_grad()
            if gpu is not None:
                inputs = inputs.to(gpu)
                targets = targets.to(gpu)

            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            opt.step()

            train_epoch_losses.append(loss.item())

        t_losses.append(float(np.nanmean(train_epoch_losses)))

        # ---------------- Validation ----------------
        net.eval()
        with torch.no_grad():
            val_epoch_losses = []
            preds = np.array([], dtype=np.int64)
            gt = np.array([], dtype=np.int64)

            for i, (inputs, targets) in enumerate(test_loader):
                if gpu is not None:
                    inputs = inputs.to(gpu)
                    targets = targets.to(gpu)

                outputs = net(inputs)
                loss = criterion(outputs, targets.long())
                val_epoch_losses.append(loss.item())

                batch_preds = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
                batch_gt = targets.cpu().numpy().flatten()

                preds = np.concatenate((preds, batch_preds))
                gt = np.concatenate((gt, batch_gt))

        v_losses.append(float(np.nanmean(val_epoch_losses)))
        v_preds_out = preds
        v_gt_out = gt

        # Classification metrics (window-wise)
        conf_mat = confusion_matrix(v_gt_out, v_preds_out, normalize="true")
        v_acc = conf_mat.diagonal()

        v_prec = precision_score(v_gt_out, v_preds_out, average=None, zero_division=1)
        v_rec = recall_score(v_gt_out, v_preds_out, average=None, zero_division=1)
        v_f1 = f1_score(v_gt_out, v_preds_out, average=None, zero_division=1)

        avg_acc = float(np.nanmean(v_acc) * 100.0)
        avg_prec = float(np.nanmean(v_prec) * 100.0)
        avg_rec = float(np.nanmean(v_rec) * 100.0)
        avg_f1 = float(np.nanmean(v_f1) * 100.0)

        # ---------------- Detection mAP (sample-wise segments) ----------------
        # Unwindow predictions to sample-wise
        unseg_preds, _ = unwindow_inertial_data(
            val_data,
            test_dataset.ids,
            v_preds_out,
            cfg["dataset"]["window_size"],
            cfg["dataset"]["window_overlap"],
        )
        v_segments = convert_samples_to_segments(
            val_data[:, 0], unseg_preds, cfg["dataset"]["sampling_rate"]
        )

        # Save raw segment results at final epoch (for later analysis)
        if epoch == (cfg["train_cfg"]["epochs"] - 1):
            v_results = pd.DataFrame(
                {
                    "video_id": v_segments["video-id"],
                    "t_start": v_segments["t-start"].tolist(),
                    "t_end": v_segments["t-end"].tolist(),
                    "label": v_segments["label"].tolist(),
                    "score": v_segments["score"].tolist(),
                }
            )
            unproc_dir = os.path.join(ckpt_folder, "unprocessed_results")
            mkdir_if_missing(unproc_dir)
            np.save(os.path.join(unproc_dir, f"v_preds_{split_name}.npy"), v_preds_out)
            np.save(os.path.join(unproc_dir, f"v_gt_{split_name}.npy"), v_gt_out)
            v_results.to_csv(
                os.path.join(unproc_dir, f"v_seg_{split_name}.csv"),
                index=False,
            )

        # Compute mAP for this epoch
        v_mAP_vec, _ = det_eval.evaluate(v_segments)
        v_mAP_all.append(v_mAP_vec)

        # ---------------- Logging ----------------
        block1 = "Epoch: [{:03d}/{:03d}]".format(
            epoch + 1, cfg["train_cfg"]["epochs"]
        )
        block2 = "TRAINING:\tavg. loss {:4.4f}".format(t_losses[-1])
        block3 = "VALIDATION:\tavg. loss {:4.4f}".format(v_losses[-1])

        if v_mAP_vec is None or len(v_mAP_vec) == 0 or np.all(np.isnan(v_mAP_vec)):
            block4 = "\t\tAvg. mAP    n/a"
        else:
            avg_map = float(np.nanmean(v_mAP_vec) * 100.0)
            block4 = "\t\tAvg. mAP {:4.2f} (%)".format(avg_map)
            for tiou, tiou_mAP in zip(
                cfg["dataset"]["tiou_thresholds"], v_mAP_vec
            ):
                block4 += " mAP@{:.1f} {:4.2f} (%)".format(
                    float(tiou), float(tiou_mAP * 100.0)
                )

        block4 += " Acc {:4.2f} (%) Prec {:4.2f} (%) Rec {:4.2f} (%) F1 {:4.2f} (%)".format(
            avg_acc, avg_prec, avg_rec, avg_f1
        )

        print(block1)
        print(block2)
        print(block3)
        print(block4)

        # Neptune logging
        if run is not None:
            run["train/loss"].log(t_losses[-1])
            run["val/loss"].log(v_losses[-1])
            if v_mAP_vec is not None and len(v_mAP_vec) > 0 and not np.all(
                np.isnan(v_mAP_vec)
            ):
                run["val/mAP"].log(float(np.nanmean(v_mAP_vec) * 100.0))
            run["val/acc_mean"].log(avg_acc)
            run["val/prec_mean"].log(avg_prec)
            run["val/rec_mean"].log(avg_rec)
            run["val/f1_mean"].log(avg_f1)

        # ---------------- Checkpoint: save best-F1 model + per-class CSV ----------------
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch + 1
            best_v_mAP_vec = v_mAP_vec

            ckpt_name = f"best_{split_name}.pth.tar"
            print(
                f"New best F1 {best_f1:.2f}% at epoch {best_epoch}, "
                f"saving checkpoint to {os.path.join(ckpt_folder, ckpt_name)}"
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_f1": best_f1,
                },
                False,
                file_folder=ckpt_folder,
                file_name=ckpt_name,
            )

            # Save per-class metrics for this best epoch
            num_classes = len(v_acc)
            if class_id2name is not None and len(class_id2name) == num_classes:
                class_names = [class_id2name[i] for i in range(num_classes)]
            else:
                class_names = [f"class_{i}" for i in range(num_classes)]

            per_class_df = pd.DataFrame(
                {
                    "split": [split_name] * num_classes,
                    "sampling_rate": [sampling_rate] * num_classes,
                    "freq_tag": [freq_tag] * num_classes,
                    "epoch": [epoch + 1] * num_classes,
                    "class_id": list(range(num_classes)),
                    "class_name": class_names,
                    "acc": v_acc,
                    "prec": v_prec,
                    "rec": v_rec,
                    "f1": v_f1,
                }
            )
            per_class_path = os.path.join(
                ckpt_folder, f"per_class_metrics_{split_name}.csv"
            )
            per_class_df.to_csv(per_class_path, index=False)

        # LR scheduler step
        if scheduler is not None:
            scheduler.step()

    # Prepare mAP output for main.py
    if best_v_mAP_vec is None:
        v_mAP_best = np.array([], dtype=float)
    else:
        v_mAP_best = np.array(best_v_mAP_vec, dtype=float)

    return t_losses, v_losses, v_mAP_best, v_preds_out, v_gt_out, net


def validate_one_epoch(loader, network, criterion, gpu=None):
    """
    Validate the network for one epoch.

    Parameters
    ----------
    loader : DataLoader
        Validation dataloader.
    network : nn.Module
        Model.
    criterion : loss
        Loss function.
    gpu : torch.device or None
        Optional CUDA device.

    Returns
    -------
    losses : list[float]
    preds : np.ndarray
    gt : np.ndarray
    """
    network.eval()
    losses = []
    preds = np.array([], dtype=np.int64)
    gt = np.array([], dtype=np.int64)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if gpu is not None:
                inputs = inputs.to(gpu)
                targets = targets.to(gpu)

            outputs = network(inputs)
            batch_loss = criterion(outputs, targets.long())
            losses.append(batch_loss.item())

            batch_preds = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
            batch_gt = targets.cpu().numpy().flatten()

            preds = np.concatenate((preds, batch_preds))
            gt = np.concatenate((gt, batch_gt))

    return losses, preds, gt
