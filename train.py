# coding: utf-8
import os
import sys
import warnings
from typing import Dict, Tuple, List
from collections import OrderedDict

import yaml
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.append("src")
import dataset
import my_model
from utils import set_seed, parse_args
from ml_logger import MlLogger

warnings.simplefilter('ignore')


def init() -> Tuple[Dict, MlLogger]:
    """実験の初期化処理を行う

    Returns:
        Tuple[Dict, MlLogger]: 設定情報, ロガーインスタンス
    """
    # 設定ファイルの読み込み
    args = parse_args()
    with open(args.config_path) as f:
        settings = yaml.load(f)
    set_seed(settings['globals']['seed'])

    # Mlflowの用意
    logger = MlLogger(args.exp_name, {"fold": settings['globals']['use_fold']})
    logger.logging_params(settings)

    # もとになるCSVファイルを保存
    logger.logging_file(settings['globals']['csv_path'])

    # 一時ファイル保存ディレクトリを作成
    if not os.path.exists(settings['globals']['tmp_dir']):
        os.makedirs(settings['globals']['tmp_dir'])

    return settings, logger


def get_file_list(settings: Dict) -> Tuple[List, List, List]:
    """csvファイルからtrain,validファイルリストを作成する

    Args:
        settings (Dict): 設定情報

    Returns:
        Tuple[List, List, List]: Trainファイルリスト, Validファイルリスト, Labelリスト
    """
    train_df = pd.read_csv(settings['globals']['csv_path'])
    skf = StratifiedKFold(**settings["split"]["params"])
    train_df["fold"] = -1
    for fold_id, (train_index, val_index) in enumerate(skf.split(train_df, train_df["label_code"])):
        train_df.iloc[val_index, -1] = fold_id

    # check the propotion
    use_fold = settings["globals"]["use_fold"]
    train_file_list = train_df.query(
        "fold != @use_fold")[["file_path", "label_code"]].values.tolist()
    val_file_list = train_df.query(
        "fold == @use_fold")[["file_path", "label_code"]].values.tolist()

    print("[fold {}] train: {}, val: {}".format(
        use_fold, len(train_file_list), len(val_file_list)))

    labels = {}
    for i, label_code in enumerate(list(train_df['label_code'].unique())):
        labels[label_code] = i
    return train_file_list, val_file_list, labels


def get_loader(dataset_cfg: Dict, loader_cfg: Dict, train_file_list: List, valid_file_list: List, label_code: Dict) -> Tuple[data.DataLoader, data.DataLoader]:
    """DataLoaderを作成する

    Args:
        dataset_cfg (Dict): datasetの設定
        loader_cfg (Dict): dataloaderの設定
        train_file_list (List): Trainファイルリスト
        valid_file_list (List): Validファイルリスト
        label_code (Dict): ラベル情報

    Returns:
        Tuple[data.DataLoader, data.DataLoader]: Train用DataLoader, Valid用DataLoader
    """
    try:
        train_dataset = getattr(dataset, dataset_cfg['name'])(
            train_file_list, dataset_cfg, label_code,
            waveform_transforms=dataset_cfg['augment']['signal'],
            spectrogram_transforms=dataset_cfg['augment']['spectrogram'])

        valid_dataset = getattr(dataset, dataset_cfg['name'])(
            valid_file_list, dataset_cfg, label_code)
    except AttributeError as e:
        print(f"Dataset {dataset_cfg['name']} is None. {e}")
        exit(1)

    train_loader = data.DataLoader(train_dataset, **loader_cfg["train"])
    val_loader = data.DataLoader(valid_dataset, **loader_cfg["val"])
    return train_loader, val_loader


def get_model(model_cfg: Dict):
    try:
        model = getattr(my_model, model_cfg['name'])(**model_cfg['params'])
    except AttributeError as e:
        print(f"Model {model_cfg['name']} is None. {e}")
        exit(1)
    return model


def get_metrics(preds: np.ndarray, targets: np.ndarray) -> Tuple[np.float, np.float, np.float, np.float]:
    tmp_pred = []
    tmp_tar = []
    for pred, target in zip(preds, targets):
        tmp_pred.append(np.argmax(pred))
        tmp_tar.append(np.argmax(target))

    acc = accuracy_score(tmp_tar, tmp_pred)
    prec = precision_score(tmp_tar, tmp_pred, average='macro')
    recall = recall_score(tmp_tar, tmp_pred, average='macro')
    f1 = f1_score(tmp_tar, tmp_pred, average='macro')
    return acc, prec, recall, f1


def train(model, optimizer, loss_func, train_loader, scheduler, device) -> Tuple[np.float, np.float, np.float, np.float, np.float]:
    losses = []
    probs, targets = [], []

    with tqdm(train_loader) as pbar:
        for batch_idx, (image, target) in enumerate(pbar):
            tmp_target = target

            image, target = image.to(device).float(), target.to(device)
            optimizer.zero_grad()
            batch_output_dict = model(image, None)
            loss = loss_func(batch_output_dict['logits'], target)
            losses.append(loss.item())

            # metrics計算
            proba = batch_output_dict['multilabel_proba'].detach(
            ).cpu().numpy()
            acc, prec, recall, f1 = get_metrics(proba, tmp_target)

            # 予測値と真値を保存
            probs.append(proba)
            targets.append(tmp_target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # batchでの結果を表示
            pbar.set_postfix(OrderedDict(
                loss="{:.5f}".format(np.average(losses)),
                acc="{:.3f}".format(np.average(acc)),
                prec="{:.3f}".format(np.average(prec)),
                recall="{:.3f}".format(np.average(recall)),
                f1="{:.3f}".format(np.average(f1))
            ))
    acc, prec, recall, f1 = get_metrics(
        np.concatenate(probs, axis=0), np.concatenate(targets, axis=0)
    )
    return np.average(losses), acc, prec, recall, f1


def validation(model, loss_func, valid_loader, device) -> Tuple[np.float, np.float, np.float, np.float, np.float]:
    losses = []
    probs, targets = [], []
    with tqdm(valid_loader) as pbar:
        for batch_idx, (image, target) in enumerate(pbar):
            tmp_target = target
            image, target = image.to(device).float(), target.to(device)
            batch_output_dict = model(image)
            loss = loss_func(batch_output_dict['logits'], target)
            losses.append(loss.item())

            # metrics計算
            proba = batch_output_dict['multilabel_proba'].detach(
            ).cpu().numpy()
            acc, prec, recall, f1 = get_metrics(proba, tmp_target)

            # 予測値と真値を保存
            probs.append(proba)
            targets.append(tmp_target)

            # batchでの結果を表示
            pbar.set_postfix(OrderedDict(
                loss="{:.5f}".format(np.average(losses)),
                acc="{:.3f}".format(np.average(acc)),
                prec="{:.3f}".format(np.average(prec)),
                recall="{:.3f}".format(np.average(recall)),
                f1="{:.3f}".format(np.average(f1))
            ))
    acc, prec, recall, f1 = get_metrics(
        np.concatenate(probs, axis=0), np.concatenate(targets, axis=0)
    )
    return np.average(losses), acc, prec, recall, f1


def main():
    # 実験の初期化
    settings, logger = init()

    # ファイルリストを用意
    train_file_list, valid_file_list, labels = get_file_list(settings)

    # データローダーを用意
    train_loader, valid_loader = get_loader(
        settings["dataset"], settings["loader"],
        train_file_list, valid_file_list
    )

    device = torch.device(settings["globals"]["device"])

    # Modelの準備
    model = get_model(settings['model'])

    try:
        # Optimizerの準備
        optimizer = getattr(
            torch.optim, settings["optimizer"]["name"]
        )(model.parameters(), **settings["optimizer"]["params"])

        # Schedulerの準備
        scheduler = getattr(
            torch.optim.lr_scheduler, settings["scheduler"]["name"]
        )(optimizer, **settings["scheduler"]["params"])

        # Lossの準備
        loss_func = getattr(torch.nn, settings["loss"]["name"])(
            **settings["loss"]["params"])
    except Exception as e:
        print(f"{e}")
        exit(1)

    model.to(device)

    # Train Loop
    best_f1 = 0.0
    best_val_loss = 1e+10
    try:
        for epoch in range(settings['globals']['num_epochs']):
            # TRAIN
            model.train()
            t_loss, t_acc, t_prec, t_recall, t_f1 = train(
                model, optimizer, loss_func, train_loader, scheduler, device)

            # VALIDATION
            model.eval()
            v_loss, v_acc, v_prec, v_recall, v_f1 = validation(
                model, loss_func, valid_loader, device)

            # mlflowにロギング
            logger.logging_step_metric(
                epoch,
                {
                    'train_loss': t_loss,
                    'train_acc': t_acc,
                    'train_prec': t_prec,
                    'train_recall': t_recall,
                    'train_f1': t_f1,
                    'valid_loss': v_loss,
                    'valid_acc': v_acc,
                    'valid_prec': v_prec,
                    'valid_recall': v_recall,
                    'valid_f1': v_f1,
                }
            )

            # プロンプトに出力
            print(f"\n[EPOCH] {epoch}")
            print(
                f"[TRAIN] loss: {t_loss:.5f}, acc: {t_acc:.3f}, prec: {t_prec:.3f}, recall: {t_recall:.3f}, f1: {t_f1:.3f}")
            print(
                f"[VALID] loss: {v_loss:.5f}, acc: {v_acc:.3f}, prec: {v_prec:.3f}, recall: {v_recall:.3f}, f1: {v_f1:.3f}")

            # Best F1 と Best Lossのモデルを保存する
            if v_loss < best_val_loss:
                print(
                    f'BEST VALID_LOSS={best_val_loss:.5f} -> {v_loss:.5f}')
                best_val_loss = v_loss
                checkpoint = {
                    'model': model.state_dict()
                }
                checkpoint_path = os.path.join(
                    settings['globals']['tmp_dir'], 'best_loss.pth')
                torch.save(checkpoint, checkpoint_path)
            if v_f1 > best_f1:
                print(
                    f'BEST VALID_F1={best_f1:.5f} -> {v_f1:.5f}')
                best_f1 = v_f1
                checkpoint = {
                    'model': model.state_dict()
                }
                checkpoint_path = os.path.join(
                    settings['globals']['tmp_dir'], 'best_f1.pth')
                torch.save(checkpoint, checkpoint_path)

            # 最新版モデルを保存
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            checkpoint_path = os.path.join(
                settings['globals']['tmp_dir'], 'last.pth')
            torch.save(checkpoint, checkpoint_path)
    except KeyboardInterrupt:
        pass

    logger.logging_metrics(
        {
            'best_val_loss': best_val_loss,
            'best_val_F1': best_f1
        }
    )
    logger.logging_files("./tmp")


if __name__ == "__main__":
    main()
