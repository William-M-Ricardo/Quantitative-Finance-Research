import numpy as np
import polars as pl
import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go

from vnpy.alpha import (
    AlphaModel,
    AlphaDataset,
    Segment
)


class XgbModel(AlphaModel):
    """XGBoost集成学习算法"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        min_split_loss: float = 0,
        max_depth: int = 12,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 1,
        seed: int = None
    ):
        """
        Parameters
        ----------
        learning_rate : float
            学习率
        min_split_loss : float
            分裂损失阈值
        max_depth : int
            树的最大深度
        reg_alpha : float
            L1正则化系数
        reg_lambda : float
            L2正则化系数
        num_boost_round : int
            最大训练轮数
        early_stopping_rounds : int
            提前停止训练的轮数
        verbose_eval : int
            打印训练日志的间隔轮数
        seed : int
            随机种子
        """
        self.params: dict = {
            "objective": "reg:squarederror",
            "learning_rate": learning_rate,
            "min_split_loss": min_split_loss,
            "max_depth": max_depth,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "seed": seed
        }

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds
        self.verbose_eval: int = verbose_eval

        self.model: xgb.Booster = None

        self.evals_result: dict = {}
        self.feature_names: list[str] = []

    def fit(self, dataset: AlphaDataset) -> None:
        """使用数据拟合模型"""
        # 获取训练数据
        df_train: pl.DataFrame = dataset.fetch_learn(Segment.TRAIN)
        df_train = df_train.sort(["datetime", "vt_symbol"])

        df_valid: pl.DataFrame = dataset.fetch_learn(Segment.VALID)
        df_valid = df_valid.sort(["datetime", "vt_symbol"])

        # 提取特征名称
        self.feature_names = df_train.columns[2:-1]

        # 转换为DMatrix
        dtrain: xgb.DMatrix = xgb.DMatrix(df_train.select(df_train.columns[2: -1]).to_numpy(), label=df_train["label"])
        dvalid: xgb.DMatrix = xgb.DMatrix(df_valid.select(df_valid.columns[2: -1]).to_numpy(), label=df_valid["label"])

        # 执行模型训练
        self.model = xgb.train(
            self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
            evals_result=self.evals_result,
        )

        # 转换评估结果
        self.evals_result["train"] = list(self.evals_result["train"].values())[0]
        self.evals_result["valid"] = list(self.evals_result["valid"].values())[0]

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """使用模型进行预测"""
        # 检查模型存在
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        # 获取预测用数据
        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        # 转换为Data
        data = df.select(df.columns[2: -1]).to_pandas()

        # 返回预测结果
        return self.model.predict(xgb.DMatrix(data))

    def detail(self) -> dict:
        """获取模型细节"""
        for importance_type in ["weight", "gain"]:
            # 获取特征分数
            score_data: list[float] = list(self.model.get_score(importance_type=importance_type).values())

            feature_importance: pd.Series = pd.Series(
                score_data,
                index=self.feature_names
            )

            # 特征分数排序
            feature_importance = feature_importance.sort_values(ascending=True)

            # 绘制柱形图
            fig = go.Figure(go.Bar(
                x=feature_importance.values,
                y=feature_importance.index,
                orientation='h'
            ))

            fig.update_layout(
                title=f"Feature Importance ({importance_type})",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=1800,
                width=800,
                showlegend=False
            )
            fig.show()
