import pandas as pd
import mlflow
import sys
import csv
import numpy as np
from sklearn.metrics import mean_squared_log_error
from runner import Runner
from util import Util

# 定数
shap_sampling = 10000
corr_sampling = 10000


class youtube_Runner(Runner):
    def __init__(self, run_name, model_cls, features, setting, params, cv):
        super().__init__(run_name, model_cls, features, setting, params, cv)
        self.metrics = mean_squared_log_error
        self.category_score_list = []
        # self.categoricals = ['categoryId', 'channelId']  # カテゴリ変数を指定する場合に使用する

    def train_fold(self, i_fold):
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != 'all'
        train_x = self.train_x.copy()
        train_y = self.train_y.copy()

        if validation:

            # 学習データ・バリデーションデータのindexを取得
            if self.cv_method == 'KFold':
                tr_idx, va_idx = self.load_index_k_fold(i_fold)
            elif self.cv_method == 'StratifiedKFold':
                tr_idx, va_idx = self.load_index_sk_fold(i_fold)
            elif self.cv_method == 'GroupKFold':
                tr_idx, va_idx = self.load_index_gk_fold_shuffle(i_fold)
            elif self.cv_method == 'StratifiedGroupKFold':
                tr_idx, va_idx = self.load_index_sgk_fold(i_fold)
            elif self.cv_method == 'TrainTestSplit':
                tr_idx, va_idx = self.load_index_train_test_split()
            elif self.cv_method == 'CustomTimeSeriesSplitter':
                tr_idx, va_idx = self.load_index_custom_ts_fold(i_fold)
            else:
                print('CVメソッドが正しくないため終了します')
                sys.exit(0)

            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # =================================
            """
            # pseudo labelingデータを追加する
            pseudo_df = pd.read_pickle(self.feature_dir_name + 'test_pseudo_06231123.pkl')
            pseudo_df_x = pseudo_df.drop('y', axis=1)[self.features]
            pseudo_df_y = np.log1p(pseudo_df[self.target])
            # 結合
            tr_x = pd.concat([tr_x, pseudo_df_x], axis=0)
            tr_y = pd.concat([tr_y, pseudo_df_y], axis=0)

            # pseudo labelingデータを追加する
            pseudo_df = pd.read_pickle(self.feature_dir_name + 'test_pseudo_06231738.pkl')
            pseudo_df_x = pseudo_df.drop('y', axis=1)[self.features]
            pseudo_df_y = np.log1p(pseudo_df[self.target])
            # 結合
            tr_x = pd.concat([tr_x, pseudo_df_x], axis=0)
            tr_y = pd.concat([tr_y, pseudo_df_y], axis=0)
            """
            # =================================

            # 学習を行う
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # バリデーションデータへの予測・評価を行う
            if self.calc_shap:
                va_pred, self.shap_values[va_idx[:shap_sampling]] = model.predict_and_shap(va_x, shap_sampling)
            else:
                va_pred = model.predict(va_x)

            # log1pで学習させたので、それを戻す
            va_pred = np.expm1(va_pred)
            va_pred = np.where(va_pred < 0, 0, va_pred)
            score = np.sqrt(self.metrics(np.expm1(va_y), va_pred))
            # foldごとのスコアをリストに追加
            self.fold_score_list.append([f'fold{i_fold}', round(score, 4)])

            # =================================
            # 特別仕様: groupごとのスコアを算出
            _temp_df = self.train_x[['categoryId']].copy()
            _temp_df = _temp_df.iloc[va_idx].reset_index(drop=True)
            _temp_df = pd.concat([_temp_df, va_y.reset_index(drop=True), pd.Series(va_pred, name='pred')], axis=1)

            for i in sorted(_temp_df['categoryId'].unique().tolist()):
                category_df = _temp_df.query('categoryId == @i')
                category_y = category_df['y']
                category_pred = category_df['pred']
                category_score = np.sqrt(self.metrics(np.expm1(category_y), category_pred))
                self.category_score_list.append([i, round(category_score, 4)])
            # =================================

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで学習を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, None, None, None

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        self.logger.info(f'{self.run_name} - start training cv')
        if self.cv_method in ['KFold', 'TrainTestSplit', 'CustomTimeSeriesSplitter']:
            self.logger.info(f'{self.run_name} - cv method: {self.cv_method}')
        else:
            self.logger.info(f'{self.run_name} - cv method: {self.cv_method} - group: {self.cv_target_gr_column} - stratify: {self.cv_target_sf_column}')

        scores = []  # 各foldのscoreを保存
        va_idxes = []  # 各foldのvalidationデータのindexを保存
        preds = []  # 各foldの推論結果を保存

        # 各foldで学習を行う
        for i_fold in range(self.n_splits):
            # 学習を行う
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            self.logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # mlflowでトラッキング
            # mlflow.log_metric(f'fold {i_fold}', score)

            # モデルを保存する
            model.save_model(self.out_dir_name)

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        # 全体のスコアを算出
        if self.cv_method not in ['TrainTestSplit', 'CustomTimeSeriesSplitter']:
            score_all_data = np.sqrt(self.metrics(np.expm1(self.train_y), preds))
        else:
            score_all_data = None

        # mlflowでトラッキング
        # mlflow.log_metric('score_all_data', score_all_data)
        # mlflow.log_metric('score_fold_mean', np.mean(scores))

        # oofデータに対するfoldごとのscoreをcsvに書き込む（foldごとに分析する用）
        self.score_list.append(['score_all_data', score_all_data])
        self.score_list.append(['score_fold_mean', np.mean(scores)])
        for i in self.fold_score_list:
            self.score_list.append(i)
        for i in self.category_score_list:
            self.score_list.append(i)
        with open(self.out_dir_name + f'{self.run_name}_score.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(self.score_list)

        # カテゴリごとのスコアもmlflowでトラッキングする
        def score_mean(df):
            df = df.groupby('run_name').mean().round(4).reset_index().sort_values('run_name')
            return df
        _score_df = pd.read_csv(self.out_dir_name + f'{self.run_name}_score.csv')
        _score_df = score_mean(_score_df)
        _score_df = _score_df.T
        _score_df.columns = _score_df.iloc[0]
        _score_df = _score_df.drop(_score_df.index[0])
        for col in _score_df.columns.tolist():
            mlflow.log_metric(col, _score_df[col].values[0])

        # 学習データでの予測結果の保存
        if self.save_train_pred:
            Util.dump_df_pickle(pd.DataFrame(preds), self.out_dir_name + f'{self.run_name}_train.pkl')

        # 評価結果の保存
        self.logger.result_scores(self.run_name, scores, score_all_data)

        # shap feature importanceデータの保存
        if self.calc_shap:
            self.shap_feature_importance()

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        test_x = self.load_x_test()
        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_splits):
            self.logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model(self.out_dir_name)
            pred = np.expm1(model.predict(test_x))
            preds.append(pred)
            self.logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 推論結果の保存（submit対象データ）
        Util.dump_df_pickle(pd.DataFrame(pred_avg), self.out_dir_name + f'{self.run_name}_pred.pkl')

        self.logger.info(f'{self.run_name} - end prediction cv')

    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む
        対数変換や使用するデータを削除する場合には、このメソッドの修正が必要
        :return: 学習データの目的変数
        """
        df = pd.read_pickle(self.feature_dir_name + f'{self.train_file_name}')

        # 特定のサンプルを除外して学習させる場合 -------------
        # df = df.drop(index=self.remove_train_index)
        # -----------------------------------------

        return pd.Series(np.log1p(df[self.target]))
