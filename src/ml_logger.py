# coding: utf-8

import mlflow
from typing import Dict


class MlLogger(object):
    def __init__(self, exp_name: str, tag: Dict):
        """mlflowの初期化

        Args:
            exp_name (str): 実験名
            tag (Dict): タグ情報
        """
        mlflow.set_experiment(exp_name)
        mlflow.set_tags(tag)

    def logging_params(self, params: Dict) -> None:
        """パラメータを保存する

        Args:
            params (Dict): パラメータ
        """
        mlflow.log_params(params)

    def logging_step_metric(self, step: int, metrics: Dict) -> None:
        """段階的にメトリクスを保存する

        Args:
            step (int): ステップ数
            metrics (Dict): メトリクス
        """
        mlflow.log_metrics(metrics, step=step)

    def logging_metrics(self, metrics: Dict) -> None:
        """メトリクスを保存する

        Args:
            metrics (Dict): メトリクス
        """
        mlflow.log_metrics(metrics)

    def logging_files(self, dir_path: str) -> None:
        """指定されたディレクトリ配下のファイルを保存する

        Args:
            dir_path (str): ディレクトリパス
        """
        mlflow.log_artifacts(dir_path)

    def logging_file(self, file_path: str) -> None:
        """指定されたファイルを保存する

        Args:
            dir_path (str): ファイルパス
        """
        mlflow.log_artifact(file_path)
