import sys
import os
import datetime
import yaml
import json
import collections as cl
import warnings
import fire
import traceback
import mlflow
from model_lgb import ModelLGB
from model_cb import ModelCB
from youtube_runner import youtube_Runner
from util import Submission

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

key_list = ['use_features', 'model_params', 'cv', 'setting']

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
EXPERIMENT_NAME = yml['SETTING']['EXPERIMENT_NAME']
TRACKING_DIR = yml['SETTING']['TRACKING_DIR']


def exist_check(path, run_name) -> None:
    """モデルディレクトリの存在チェック

    Args:
        path (str): モデルディレクトリのpath
        run_name (str): チェックするrun名

    """
    dir_list = []
    for d in os.listdir(path):
        dir_list.append(d.split('-')[-1])

    if run_name in dir_list:
        print('同名のrunが実行済みです。再実行しますか？[Y/n]')
        x = input('>> ')
        if x != 'Y':
            print('終了します')
            sys.exit(0)
    return None


def my_makedirs(path) -> None:
    """引数のpathディレクトリが存在しなければ、新規で作成する

    Args:
        path (str): 作成するディレクトリ名

    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return None


def save_model_config(key_list, value_list, dir_name, run_name) -> None:
    """学習のjsonファイル生成

    どんなパラメータ/特徴量で学習させたモデルかを管理するjsonファイルを出力する

    """
    def set_default(obj):
        """json出力の際にset型のオブジェクトをリストに変更する"""
        if isinstance(obj, set):
            return list(obj)
        raise TypeError

    ys = cl.OrderedDict()
    for i, v in enumerate(key_list):
        data = cl.OrderedDict()
        data = value_list[i]
        ys[v] = data
        mlflow.log_param(v, data)
    fw = open(dir_name + run_name + '_param.json', 'w')
    json.dump(ys, fw, indent=4, default=set_default)
    return None


def get_cv_info() -> dict:
    """CVの情報を設定する

    methodは[KFold, StratifiedKFold ,GroupKFold, StratifiedGroupKFold, CustomTimeSeriesSplitter, TrainTestSplit]から選択可能
    CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    StratifiedKFold or GroupKFold or StratifiedGroupKFold の場合はcv_target_gr, cv_target_sfに対象カラム名を設定する

    Returns:
        dict: cvの辞書

    """
    return yml['SETTING']['CV']


def get_run_name(cv, model_type):
    """run名を設定する
    """
    run_name = model_type
    suffix = '_' + datetime.datetime.now().strftime("%m%d%H%M")
    model_info = ''
    run_name = run_name + '_' + cv.get('method') + model_info + suffix
    return run_name


def get_setting_info():
    """setting情報を設定する
    """
    setting = {
        'feature_directory': yml['SETTING']['FEATURE_DIR_NAME'],  # 特徴量の読み込み先ディレクトリ
        'model_directory': MODEL_DIR_NAME,  # モデルの保存先ディレクトリ
        'train_file_name': yml['SETTING']['TRAIN_FILE_NAME'],
        'test_file_name': yml['SETTING']['TEST_FILE_NAME'],
        'target': yml['SETTING']['TARGET_COL'],  # 目的変数
        'calc_shap': yml['SETTING']['CALC_SHAP'],  # shap値を計算するか否か
        'save_train_pred': yml['SETTING']['SAVE_TRAIN_PRED']  # trainデータでの推論値を特徴量として加えたい場合はTrueに設定する
    }
    return setting


def main(model_type='lgb') -> str:
    """トレーニングのmain関数

    model_typeによって学習するモデルを変更する
    → lgb, cb, xgb, nnが標準で用意されている

    Args:
        model_type (str, optional): どのモデルで学習させるかを指定. Defaults to 'lgb'.

    Returns:
        str: [description]

    Examples:
        >>> python hoge.py --model_type="lgb"
        >>> python hoge.py lgb

    """

    cv = get_cv_info()  # CVの情報辞書
    run_name = get_run_name(cv, model_type)  # run名
    dir_name = MODEL_DIR_NAME + run_name + '/'  # 学習に使用するディレクトリ
    setting = get_setting_info()  # 諸々の設定ファイル辞書

    # すでに実行済みのrun名がないかチェックし、ディレクトリを作成する
    exist_check(MODEL_DIR_NAME, run_name)
    my_makedirs(dir_name)

    # モデルに合わせてパラメータを読み込む
    model_cls = None
    if model_type == 'lgb':
        model_params = yml['MODEL_LGB']['PARAM']
        model_cls = ModelLGB
    elif model_type == 'cb':
        model_params = yml['MODEL_CB']['PARAM']
        model_cls = ModelCB
    elif model_type == 'xgb':
        pass
    elif model_type == 'nn':
        pass
    else:
        print('model_typeが不正なため終了します')
        sys.exit(0)

    features = [
        'categoryId',
        # 'categoryId_channelId_comment_count_count',
        # 'categoryId_channelId_comment_count_max',
        # 'categoryId_channelId_comment_count_max_diff',
        # 'categoryId_channelId_comment_count_max_minus_q50',
        # 'categoryId_channelId_comment_count_max_minus_q75',
        # 'categoryId_channelId_comment_count_max_minus_q90',
        # 'categoryId_channelId_comment_count_mean',
        # 'categoryId_channelId_comment_count_mean_diff',
        # 'categoryId_channelId_comment_count_mean_minus_q50',
        # 'categoryId_channelId_comment_count_mean_minus_q75',
        # 'categoryId_channelId_comment_count_mean_minus_q90',
        # 'categoryId_channelId_comment_count_min',
        # 'categoryId_channelId_comment_count_min_diff',
        # 'categoryId_channelId_comment_count_q10',
        # 'categoryId_channelId_comment_count_q25',
        # 'categoryId_channelId_comment_count_q50',
        # 'categoryId_channelId_comment_count_q75',
        # 'categoryId_channelId_comment_count_q90',
        # 'categoryId_channelId_comment_count_std',
        # 'categoryId_channelId_comment_count_sum',
        # 'categoryId_channelId_comments_likes_count',
        # 'categoryId_channelId_comments_likes_max',
        # 'categoryId_channelId_comments_likes_max_diff',
        # 'categoryId_channelId_comments_likes_max_minus_q50',
        # 'categoryId_channelId_comments_likes_max_minus_q75',
        # 'categoryId_channelId_comments_likes_max_minus_q90',
        # 'categoryId_channelId_comments_likes_mean',
        # 'categoryId_channelId_comments_likes_mean_diff',
        # 'categoryId_channelId_comments_likes_mean_minus_q50',
        # 'categoryId_channelId_comments_likes_mean_minus_q75',
        # 'categoryId_channelId_comments_likes_mean_minus_q90',
        # 'categoryId_channelId_comments_likes_min',
        # 'categoryId_channelId_comments_likes_min_diff',
        # 'categoryId_channelId_comments_likes_q10',
        # 'categoryId_channelId_comments_likes_q25',
        # 'categoryId_channelId_comments_likes_q50',
        # 'categoryId_channelId_comments_likes_q75',
        # 'categoryId_channelId_comments_likes_q90',
        # 'categoryId_channelId_comments_likes_std',
        # 'categoryId_channelId_comments_likes_sum',
        # 'categoryId_channelId_delta_count',
        # 'categoryId_channelId_delta_max',
        # 'categoryId_channelId_delta_max_diff',
        # 'categoryId_channelId_delta_max_minus_q50',
        # 'categoryId_channelId_delta_max_minus_q75',
        # 'categoryId_channelId_delta_max_minus_q90',
        # 'categoryId_channelId_delta_mean',
        # 'categoryId_channelId_delta_mean_diff',
        # 'categoryId_channelId_delta_mean_minus_q50',
        # 'categoryId_channelId_delta_mean_minus_q75',
        # 'categoryId_channelId_delta_mean_minus_q90',
        # 'categoryId_channelId_delta_min',
        # 'categoryId_channelId_delta_min_diff',
        # 'categoryId_channelId_delta_q10',
        # 'categoryId_channelId_delta_q25',
        # 'categoryId_channelId_delta_q50',
        # 'categoryId_channelId_delta_q75',
        # 'categoryId_channelId_delta_q90',
        # 'categoryId_channelId_delta_std',
        # 'categoryId_channelId_delta_sum',
        # 'categoryId_channelId_dislikes_comments_count',
        # 'categoryId_channelId_dislikes_comments_max',
        # 'categoryId_channelId_dislikes_comments_max_diff',
        # 'categoryId_channelId_dislikes_comments_max_minus_q50',
        # 'categoryId_channelId_dislikes_comments_max_minus_q75',
        # 'categoryId_channelId_dislikes_comments_max_minus_q90',
        # 'categoryId_channelId_dislikes_comments_mean',
        # 'categoryId_channelId_dislikes_comments_mean_diff',
        # 'categoryId_channelId_dislikes_comments_mean_minus_q50',
        # 'categoryId_channelId_dislikes_comments_mean_minus_q75',
        # 'categoryId_channelId_dislikes_comments_mean_minus_q90',
        # 'categoryId_channelId_dislikes_comments_min',
        # 'categoryId_channelId_dislikes_comments_min_diff',
        # 'categoryId_channelId_dislikes_comments_q10',
        # 'categoryId_channelId_dislikes_comments_q25',
        # 'categoryId_channelId_dislikes_comments_q50',
        # 'categoryId_channelId_dislikes_comments_q75',
        # 'categoryId_channelId_dislikes_comments_q90',
        # 'categoryId_channelId_dislikes_comments_std',
        # 'categoryId_channelId_dislikes_comments_sum',
        # 'categoryId_channelId_dislikes_count',
        # 'categoryId_channelId_dislikes_max',
        # 'categoryId_channelId_dislikes_max_diff',
        # 'categoryId_channelId_dislikes_max_minus_q50',
        # 'categoryId_channelId_dislikes_max_minus_q75',
        # 'categoryId_channelId_dislikes_max_minus_q90',
        # 'categoryId_channelId_dislikes_mean',
        # 'categoryId_channelId_dislikes_mean_diff',
        # 'categoryId_channelId_dislikes_mean_minus_q50',
        # 'categoryId_channelId_dislikes_mean_minus_q75',
        # 'categoryId_channelId_dislikes_mean_minus_q90',
        # 'categoryId_channelId_dislikes_min',
        # 'categoryId_channelId_dislikes_min_diff',
        # 'categoryId_channelId_dislikes_q10',
        # 'categoryId_channelId_dislikes_q25',
        # 'categoryId_channelId_dislikes_q50',
        # 'categoryId_channelId_dislikes_q75',
        # 'categoryId_channelId_dislikes_q90',
        # 'categoryId_channelId_dislikes_std',
        # 'categoryId_channelId_dislikes_sum',
        # 'categoryId_channelId_like_dislike_ratio_count',
        # 'categoryId_channelId_like_dislike_ratio_max',
        # 'categoryId_channelId_like_dislike_ratio_max_diff',
        # 'categoryId_channelId_like_dislike_ratio_max_minus_q50',
        # 'categoryId_channelId_like_dislike_ratio_max_minus_q75',
        # 'categoryId_channelId_like_dislike_ratio_max_minus_q90',
        # 'categoryId_channelId_like_dislike_ratio_mean',
        # 'categoryId_channelId_like_dislike_ratio_mean_diff',
        # 'categoryId_channelId_like_dislike_ratio_mean_minus_q50',
        # 'categoryId_channelId_like_dislike_ratio_mean_minus_q75',
        # 'categoryId_channelId_like_dislike_ratio_mean_minus_q90',
        # 'categoryId_channelId_like_dislike_ratio_min',
        # 'categoryId_channelId_like_dislike_ratio_min_diff',
        # 'categoryId_channelId_like_dislike_ratio_q10',
        # 'categoryId_channelId_like_dislike_ratio_q25',
        # 'categoryId_channelId_like_dislike_ratio_q50',
        # 'categoryId_channelId_like_dislike_ratio_q75',
        # 'categoryId_channelId_like_dislike_ratio_q90',
        # 'categoryId_channelId_like_dislike_ratio_std',
        # 'categoryId_channelId_like_dislike_ratio_sum',
        # 'categoryId_channelId_likes_comments_count',
        # 'categoryId_channelId_likes_comments_max',
        # 'categoryId_channelId_likes_comments_max_diff',
        # 'categoryId_channelId_likes_comments_max_minus_q50',
        # 'categoryId_channelId_likes_comments_max_minus_q75',
        # 'categoryId_channelId_likes_comments_max_minus_q90',
        # 'categoryId_channelId_likes_comments_mean',
        # 'categoryId_channelId_likes_comments_mean_diff',
        # 'categoryId_channelId_likes_comments_mean_minus_q50',
        # 'categoryId_channelId_likes_comments_mean_minus_q75',
        # 'categoryId_channelId_likes_comments_mean_minus_q90',
        # 'categoryId_channelId_likes_comments_min',
        # 'categoryId_channelId_likes_comments_min_diff',
        # 'categoryId_channelId_likes_comments_q10',
        # 'categoryId_channelId_likes_comments_q25',
        # 'categoryId_channelId_likes_comments_q50',
        # 'categoryId_channelId_likes_comments_q75',
        # 'categoryId_channelId_likes_comments_q90',
        # 'categoryId_channelId_likes_comments_std',
        # 'categoryId_channelId_likes_comments_sum',
        # 'categoryId_channelId_likes_count',
        # 'categoryId_channelId_likes_max',
        # 'categoryId_channelId_likes_max_diff',
        # 'categoryId_channelId_likes_max_minus_q50',
        # 'categoryId_channelId_likes_max_minus_q75',
        # 'categoryId_channelId_likes_max_minus_q90',
        # 'categoryId_channelId_likes_mean',
        # 'categoryId_channelId_likes_mean_diff',
        # 'categoryId_channelId_likes_mean_minus_q50',
        # 'categoryId_channelId_likes_mean_minus_q75',
        # 'categoryId_channelId_likes_mean_minus_q90',
        # 'categoryId_channelId_likes_min',
        # 'categoryId_channelId_likes_min_diff',
        # 'categoryId_channelId_likes_q10',
        # 'categoryId_channelId_likes_q25',
        # 'categoryId_channelId_likes_q50',
        # 'categoryId_channelId_likes_q75',
        # 'categoryId_channelId_likes_q90',
        # 'categoryId_channelId_likes_std',
        # 'categoryId_channelId_likes_sum',
        # 'categoryId_channelId_tags_point_count',
        # 'categoryId_channelId_tags_point_max',
        # 'categoryId_channelId_tags_point_max_diff',
        # 'categoryId_channelId_tags_point_max_minus_q50',
        # 'categoryId_channelId_tags_point_max_minus_q75',
        # 'categoryId_channelId_tags_point_max_minus_q90',
        # 'categoryId_channelId_tags_point_mean',
        # 'categoryId_channelId_tags_point_mean_diff',
        # 'categoryId_channelId_tags_point_mean_minus_q50',
        # 'categoryId_channelId_tags_point_mean_minus_q75',
        # 'categoryId_channelId_tags_point_mean_minus_q90',
        # 'categoryId_channelId_tags_point_min',
        # 'categoryId_channelId_tags_point_min_diff',
        # 'categoryId_channelId_tags_point_q10',
        # 'categoryId_channelId_tags_point_q25',
        # 'categoryId_channelId_tags_point_q50',
        # 'categoryId_channelId_tags_point_q75',
        # 'categoryId_channelId_tags_point_q90',
        # 'categoryId_channelId_tags_point_std',
        # 'categoryId_channelId_tags_point_sum',
        'categoryId_comment_count_count',
        'categoryId_comment_count_max',
        'categoryId_comment_count_max_diff',
        'categoryId_comment_count_max_minus_q50',
        'categoryId_comment_count_max_minus_q75',
        'categoryId_comment_count_max_minus_q90',
        'categoryId_comment_count_mean',
        'categoryId_comment_count_mean_diff',
        'categoryId_comment_count_mean_minus_q50',
        'categoryId_comment_count_mean_minus_q75',
        'categoryId_comment_count_mean_minus_q90',
        'categoryId_comment_count_min',
        'categoryId_comment_count_min_diff',
        # 'categoryId_comment_count_per_day_count',
        # 'categoryId_comment_count_per_day_max',
        # 'categoryId_comment_count_per_day_max_diff',
        # 'categoryId_comment_count_per_day_max_minus_q50',
        # 'categoryId_comment_count_per_day_max_minus_q75',
        # 'categoryId_comment_count_per_day_max_minus_q90',
        # 'categoryId_comment_count_per_day_mean',
        # 'categoryId_comment_count_per_day_mean_diff',
        # 'categoryId_comment_count_per_day_mean_minus_q50',
        # 'categoryId_comment_count_per_day_mean_minus_q75',
        # 'categoryId_comment_count_per_day_mean_minus_q90',
        # 'categoryId_comment_count_per_day_min',
        # 'categoryId_comment_count_per_day_min_diff',
        # 'categoryId_comment_count_per_day_q10',
        # 'categoryId_comment_count_per_day_q25',
        # 'categoryId_comment_count_per_day_q50',
        # 'categoryId_comment_count_per_day_q75',
        # 'categoryId_comment_count_per_day_q90',
        # 'categoryId_comment_count_per_day_std',
        # 'categoryId_comment_count_per_day_sum',
        'categoryId_comment_count_q10',
        'categoryId_comment_count_q25',
        'categoryId_comment_count_q50',
        'categoryId_comment_count_q75',
        'categoryId_comment_count_q90',
        'categoryId_comment_count_std',
        'categoryId_comment_count_sum',
        'categoryId_comments_likes_count',
        'categoryId_comments_likes_max',
        'categoryId_comments_likes_max_diff',
        'categoryId_comments_likes_max_minus_q50',
        'categoryId_comments_likes_max_minus_q75',
        'categoryId_comments_likes_max_minus_q90',
        'categoryId_comments_likes_mean',
        'categoryId_comments_likes_mean_diff',
        'categoryId_comments_likes_mean_minus_q50',
        'categoryId_comments_likes_mean_minus_q75',
        'categoryId_comments_likes_mean_minus_q90',
        'categoryId_comments_likes_min',
        'categoryId_comments_likes_min_diff',
        'categoryId_comments_likes_q10',
        'categoryId_comments_likes_q25',
        'categoryId_comments_likes_q50',
        'categoryId_comments_likes_q75',
        'categoryId_comments_likes_q90',
        'categoryId_comments_likes_std',
        'categoryId_comments_likes_sum',
        'categoryId_delta_count',
        'categoryId_delta_max',
        'categoryId_delta_max_diff',
        'categoryId_delta_max_minus_q50',
        'categoryId_delta_max_minus_q75',
        'categoryId_delta_max_minus_q90',
        'categoryId_delta_mean',
        'categoryId_delta_mean_diff',
        'categoryId_delta_mean_minus_q50',
        'categoryId_delta_mean_minus_q75',
        'categoryId_delta_mean_minus_q90',
        'categoryId_delta_min',
        'categoryId_delta_min_diff',
        'categoryId_delta_q10',
        'categoryId_delta_q25',
        'categoryId_delta_q50',
        'categoryId_delta_q75',
        'categoryId_delta_q90',
        'categoryId_delta_std',
        'categoryId_delta_sum',
        # 'categoryId_dislike_per_day_count',
        # 'categoryId_dislike_per_day_max',
        # 'categoryId_dislike_per_day_max_diff',
        # 'categoryId_dislike_per_day_max_minus_q50',
        # 'categoryId_dislike_per_day_max_minus_q75',
        # 'categoryId_dislike_per_day_max_minus_q90',
        # 'categoryId_dislike_per_day_mean',
        # 'categoryId_dislike_per_day_mean_diff',
        # 'categoryId_dislike_per_day_mean_minus_q50',
        # 'categoryId_dislike_per_day_mean_minus_q75',
        # 'categoryId_dislike_per_day_mean_minus_q90',
        # 'categoryId_dislike_per_day_min',
        # 'categoryId_dislike_per_day_min_diff',
        # 'categoryId_dislike_per_day_q10',
        # 'categoryId_dislike_per_day_q25',
        # 'categoryId_dislike_per_day_q50',
        # 'categoryId_dislike_per_day_q75',
        # 'categoryId_dislike_per_day_q90',
        # 'categoryId_dislike_per_day_std',
        # 'categoryId_dislike_per_day_sum',
        'categoryId_dislikes_comments_count',
        'categoryId_dislikes_comments_max',
        'categoryId_dislikes_comments_max_diff',
        'categoryId_dislikes_comments_max_minus_q50',
        'categoryId_dislikes_comments_max_minus_q75',
        'categoryId_dislikes_comments_max_minus_q90',
        'categoryId_dislikes_comments_mean',
        'categoryId_dislikes_comments_mean_diff',
        'categoryId_dislikes_comments_mean_minus_q50',
        'categoryId_dislikes_comments_mean_minus_q75',
        'categoryId_dislikes_comments_mean_minus_q90',
        'categoryId_dislikes_comments_min',
        'categoryId_dislikes_comments_min_diff',
        'categoryId_dislikes_comments_q10',
        'categoryId_dislikes_comments_q25',
        'categoryId_dislikes_comments_q50',
        'categoryId_dislikes_comments_q75',
        'categoryId_dislikes_comments_q90',
        'categoryId_dislikes_comments_std',
        'categoryId_dislikes_comments_sum',
        'categoryId_dislikes_count',
        'categoryId_dislikes_max',
        'categoryId_dislikes_max_diff',
        'categoryId_dislikes_max_minus_q50',
        'categoryId_dislikes_max_minus_q75',
        'categoryId_dislikes_max_minus_q90',
        'categoryId_dislikes_mean',
        'categoryId_dislikes_mean_diff',
        'categoryId_dislikes_mean_minus_q50',
        'categoryId_dislikes_mean_minus_q75',
        'categoryId_dislikes_mean_minus_q90',
        'categoryId_dislikes_min',
        'categoryId_dislikes_min_diff',
        'categoryId_dislikes_q10',
        'categoryId_dislikes_q25',
        'categoryId_dislikes_q50',
        'categoryId_dislikes_q75',
        'categoryId_dislikes_q90',
        'categoryId_dislikes_std',
        'categoryId_dislikes_sum',
        'categoryId_like_dislike_ratio_count',
        'categoryId_like_dislike_ratio_max',
        'categoryId_like_dislike_ratio_max_diff',
        'categoryId_like_dislike_ratio_max_minus_q50',
        'categoryId_like_dislike_ratio_max_minus_q75',
        'categoryId_like_dislike_ratio_max_minus_q90',
        'categoryId_like_dislike_ratio_mean',
        'categoryId_like_dislike_ratio_mean_diff',
        'categoryId_like_dislike_ratio_mean_minus_q50',
        'categoryId_like_dislike_ratio_mean_minus_q75',
        'categoryId_like_dislike_ratio_mean_minus_q90',
        'categoryId_like_dislike_ratio_min',
        'categoryId_like_dislike_ratio_min_diff',
        'categoryId_like_dislike_ratio_q10',
        'categoryId_like_dislike_ratio_q25',
        'categoryId_like_dislike_ratio_q50',
        'categoryId_like_dislike_ratio_q75',
        'categoryId_like_dislike_ratio_q90',
        'categoryId_like_dislike_ratio_std',
        'categoryId_like_dislike_ratio_sum',
        # 'categoryId_like_per_day_count',
        # 'categoryId_like_per_day_max',
        # 'categoryId_like_per_day_max_diff',
        # 'categoryId_like_per_day_max_minus_q50',
        # 'categoryId_like_per_day_max_minus_q75',
        # 'categoryId_like_per_day_max_minus_q90',
        # 'categoryId_like_per_day_mean',
        # 'categoryId_like_per_day_mean_diff',
        # 'categoryId_like_per_day_mean_minus_q50',
        # 'categoryId_like_per_day_mean_minus_q75',
        # 'categoryId_like_per_day_mean_minus_q90',
        # 'categoryId_like_per_day_min',
        # 'categoryId_like_per_day_min_diff',
        # 'categoryId_like_per_day_q10',
        # 'categoryId_like_per_day_q25',
        # 'categoryId_like_per_day_q50',
        # 'categoryId_like_per_day_q75',
        # 'categoryId_like_per_day_q90',
        # 'categoryId_like_per_day_std',
        # 'categoryId_like_per_day_sum',
        'categoryId_likes_comments_count',
        'categoryId_likes_comments_max',
        'categoryId_likes_comments_max_diff',
        'categoryId_likes_comments_max_minus_q50',
        'categoryId_likes_comments_max_minus_q75',
        'categoryId_likes_comments_max_minus_q90',
        'categoryId_likes_comments_mean',
        'categoryId_likes_comments_mean_diff',
        'categoryId_likes_comments_mean_minus_q50',
        'categoryId_likes_comments_mean_minus_q75',
        'categoryId_likes_comments_mean_minus_q90',
        'categoryId_likes_comments_min',
        'categoryId_likes_comments_min_diff',
        'categoryId_likes_comments_q10',
        'categoryId_likes_comments_q25',
        'categoryId_likes_comments_q50',
        'categoryId_likes_comments_q75',
        'categoryId_likes_comments_q90',
        'categoryId_likes_comments_std',
        'categoryId_likes_comments_sum',
        'categoryId_likes_count',
        'categoryId_likes_max',
        'categoryId_likes_max_diff',
        'categoryId_likes_max_minus_q50',
        'categoryId_likes_max_minus_q75',
        'categoryId_likes_max_minus_q90',
        'categoryId_likes_mean',
        'categoryId_likes_mean_diff',
        'categoryId_likes_mean_minus_q50',
        'categoryId_likes_mean_minus_q75',
        'categoryId_likes_mean_minus_q90',
        'categoryId_likes_min',
        'categoryId_likes_min_diff',
        'categoryId_likes_q10',
        'categoryId_likes_q25',
        'categoryId_likes_q50',
        'categoryId_likes_q75',
        'categoryId_likes_q90',
        'categoryId_likes_std',
        'categoryId_likes_sum',
        'categoryId_tags_point_count',
        'categoryId_tags_point_max',
        'categoryId_tags_point_max_diff',
        'categoryId_tags_point_max_minus_q50',
        'categoryId_tags_point_max_minus_q75',
        'categoryId_tags_point_max_minus_q90',
        'categoryId_tags_point_mean',
        'categoryId_tags_point_mean_diff',
        'categoryId_tags_point_mean_minus_q50',
        'categoryId_tags_point_mean_minus_q75',
        'categoryId_tags_point_mean_minus_q90',
        'categoryId_tags_point_min',
        'categoryId_tags_point_min_diff',
        'categoryId_tags_point_q10',
        'categoryId_tags_point_q25',
        'categoryId_tags_point_q50',
        'categoryId_tags_point_q75',
        'categoryId_tags_point_q90',
        'categoryId_tags_point_std',
        'categoryId_tags_point_sum',
        'channelId_comment_count_count',
        'channelId_comment_count_max',
        'channelId_comment_count_max_diff',
        'channelId_comment_count_max_minus_q50',
        'channelId_comment_count_max_minus_q75',
        'channelId_comment_count_max_minus_q90',
        'channelId_comment_count_mean',
        'channelId_comment_count_mean_diff',
        'channelId_comment_count_mean_minus_q50',
        'channelId_comment_count_mean_minus_q75',
        'channelId_comment_count_mean_minus_q90',
        'channelId_comment_count_min',
        'channelId_comment_count_min_diff',
        # 'channelId_comment_count_per_day_count',
        # 'channelId_comment_count_per_day_max',
        # 'channelId_comment_count_per_day_max_diff',
        # 'channelId_comment_count_per_day_max_minus_q50',
        # 'channelId_comment_count_per_day_max_minus_q75',
        # 'channelId_comment_count_per_day_max_minus_q90',
        # 'channelId_comment_count_per_day_mean',
        # 'channelId_comment_count_per_day_mean_diff',
        # 'channelId_comment_count_per_day_mean_minus_q50',
        # 'channelId_comment_count_per_day_mean_minus_q75',
        # 'channelId_comment_count_per_day_mean_minus_q90',
        # 'channelId_comment_count_per_day_min',
        # 'channelId_comment_count_per_day_min_diff',
        # 'channelId_comment_count_per_day_q10',
        # 'channelId_comment_count_per_day_q25',
        # 'channelId_comment_count_per_day_q50',
        # 'channelId_comment_count_per_day_q75',
        # 'channelId_comment_count_per_day_q90',
        # 'channelId_comment_count_per_day_std',
        # 'channelId_comment_count_per_day_sum',
        'channelId_comment_count_q10',
        'channelId_comment_count_q25',
        'channelId_comment_count_q50',
        'channelId_comment_count_q75',
        'channelId_comment_count_q90',
        'channelId_comment_count_std',
        'channelId_comment_count_sum',
        'channelId_comments_likes_count',
        'channelId_comments_likes_max',
        'channelId_comments_likes_max_diff',
        'channelId_comments_likes_max_minus_q50',
        'channelId_comments_likes_max_minus_q75',
        'channelId_comments_likes_max_minus_q90',
        'channelId_comments_likes_mean',
        'channelId_comments_likes_mean_diff',
        'channelId_comments_likes_mean_minus_q50',
        'channelId_comments_likes_mean_minus_q75',
        'channelId_comments_likes_mean_minus_q90',
        'channelId_comments_likes_min',
        'channelId_comments_likes_min_diff',
        'channelId_comments_likes_q10',
        'channelId_comments_likes_q25',
        'channelId_comments_likes_q50',
        'channelId_comments_likes_q75',
        'channelId_comments_likes_q90',
        'channelId_comments_likes_std',
        'channelId_comments_likes_sum',
        'channelId_delta_count',
        'channelId_delta_max',
        'channelId_delta_max_diff',
        'channelId_delta_max_minus_q50',
        'channelId_delta_max_minus_q75',
        'channelId_delta_max_minus_q90',
        'channelId_delta_mean',
        'channelId_delta_mean_diff',
        'channelId_delta_mean_minus_q50',
        'channelId_delta_mean_minus_q75',
        'channelId_delta_mean_minus_q90',
        'channelId_delta_min',
        'channelId_delta_min_diff',
        'channelId_delta_q10',
        'channelId_delta_q25',
        'channelId_delta_q50',
        'channelId_delta_q75',
        'channelId_delta_q90',
        'channelId_delta_std',
        'channelId_delta_sum',
        # 'channelId_dislike_per_day_count',
        # 'channelId_dislike_per_day_max',
        # 'channelId_dislike_per_day_max_diff',
        # 'channelId_dislike_per_day_max_minus_q50',
        # 'channelId_dislike_per_day_max_minus_q75',
        # 'channelId_dislike_per_day_max_minus_q90',
        # 'channelId_dislike_per_day_mean',
        # 'channelId_dislike_per_day_mean_diff',
        # 'channelId_dislike_per_day_mean_minus_q50',
        # 'channelId_dislike_per_day_mean_minus_q75',
        # 'channelId_dislike_per_day_mean_minus_q90',
        # 'channelId_dislike_per_day_min',
        # 'channelId_dislike_per_day_min_diff',
        # 'channelId_dislike_per_day_q10',
        # 'channelId_dislike_per_day_q25',
        # 'channelId_dislike_per_day_q50',
        # 'channelId_dislike_per_day_q75',
        # 'channelId_dislike_per_day_q90',
        # 'channelId_dislike_per_day_std',
        # 'channelId_dislike_per_day_sum',
        'channelId_dislikes_comments_count',
        'channelId_dislikes_comments_max',
        'channelId_dislikes_comments_max_diff',
        'channelId_dislikes_comments_max_minus_q50',
        'channelId_dislikes_comments_max_minus_q75',
        'channelId_dislikes_comments_max_minus_q90',
        'channelId_dislikes_comments_mean',
        'channelId_dislikes_comments_mean_diff',
        'channelId_dislikes_comments_mean_minus_q50',
        'channelId_dislikes_comments_mean_minus_q75',
        'channelId_dislikes_comments_mean_minus_q90',
        'channelId_dislikes_comments_min',
        'channelId_dislikes_comments_min_diff',
        'channelId_dislikes_comments_q10',
        'channelId_dislikes_comments_q25',
        'channelId_dislikes_comments_q50',
        'channelId_dislikes_comments_q75',
        'channelId_dislikes_comments_q90',
        'channelId_dislikes_comments_std',
        'channelId_dislikes_comments_sum',
        'channelId_dislikes_count',
        'channelId_dislikes_max',
        'channelId_dislikes_max_diff',
        'channelId_dislikes_max_minus_q50',
        'channelId_dislikes_max_minus_q75',
        'channelId_dislikes_max_minus_q90',
        'channelId_dislikes_mean',
        'channelId_dislikes_mean_diff',
        'channelId_dislikes_mean_minus_q50',
        'channelId_dislikes_mean_minus_q75',
        'channelId_dislikes_mean_minus_q90',
        'channelId_dislikes_min',
        'channelId_dislikes_min_diff',
        'channelId_dislikes_q10',
        'channelId_dislikes_q25',
        'channelId_dislikes_q50',
        'channelId_dislikes_q75',
        'channelId_dislikes_q90',
        'channelId_dislikes_std',
        'channelId_dislikes_sum',
        'channelId_like_dislike_ratio_count',
        'channelId_like_dislike_ratio_max',
        'channelId_like_dislike_ratio_max_diff',
        'channelId_like_dislike_ratio_max_minus_q50',
        'channelId_like_dislike_ratio_max_minus_q75',
        'channelId_like_dislike_ratio_max_minus_q90',
        'channelId_like_dislike_ratio_mean',
        'channelId_like_dislike_ratio_mean_diff',
        'channelId_like_dislike_ratio_mean_minus_q50',
        'channelId_like_dislike_ratio_mean_minus_q75',
        'channelId_like_dislike_ratio_mean_minus_q90',
        'channelId_like_dislike_ratio_min',
        'channelId_like_dislike_ratio_min_diff',
        'channelId_like_dislike_ratio_q10',
        'channelId_like_dislike_ratio_q25',
        'channelId_like_dislike_ratio_q50',
        'channelId_like_dislike_ratio_q75',
        'channelId_like_dislike_ratio_q90',
        'channelId_like_dislike_ratio_std',
        'channelId_like_dislike_ratio_sum',
        # 'channelId_like_per_day_count',
        # 'channelId_like_per_day_max',
        # 'channelId_like_per_day_max_diff',
        # 'channelId_like_per_day_max_minus_q50',
        # 'channelId_like_per_day_max_minus_q75',
        # 'channelId_like_per_day_max_minus_q90',
        # 'channelId_like_per_day_mean',
        # 'channelId_like_per_day_mean_diff',
        # 'channelId_like_per_day_mean_minus_q50',
        # 'channelId_like_per_day_mean_minus_q75',
        # 'channelId_like_per_day_mean_minus_q90',
        # 'channelId_like_per_day_min',
        # 'channelId_like_per_day_min_diff',
        # 'channelId_like_per_day_q10',
        # 'channelId_like_per_day_q25',
        # 'channelId_like_per_day_q50',
        # 'channelId_like_per_day_q75',
        # 'channelId_like_per_day_q90',
        # 'channelId_like_per_day_std',
        # 'channelId_like_per_day_sum',
        'channelId_likes_comments_count',
        'channelId_likes_comments_max',
        'channelId_likes_comments_max_diff',
        'channelId_likes_comments_max_minus_q50',
        'channelId_likes_comments_max_minus_q75',
        'channelId_likes_comments_max_minus_q90',
        'channelId_likes_comments_mean',
        'channelId_likes_comments_mean_diff',
        'channelId_likes_comments_mean_minus_q50',
        'channelId_likes_comments_mean_minus_q75',
        'channelId_likes_comments_mean_minus_q90',
        'channelId_likes_comments_min',
        'channelId_likes_comments_min_diff',
        'channelId_likes_comments_q10',
        'channelId_likes_comments_q25',
        'channelId_likes_comments_q50',
        'channelId_likes_comments_q75',
        'channelId_likes_comments_q90',
        'channelId_likes_comments_std',
        'channelId_likes_comments_sum',
        'channelId_likes_count',
        'channelId_likes_max',
        'channelId_likes_max_diff',
        'channelId_likes_max_minus_q50',
        'channelId_likes_max_minus_q75',
        'channelId_likes_max_minus_q90',
        'channelId_likes_mean',
        'channelId_likes_mean_diff',
        'channelId_likes_mean_minus_q50',
        'channelId_likes_mean_minus_q75',
        'channelId_likes_mean_minus_q90',
        'channelId_likes_min',
        'channelId_likes_min_diff',
        'channelId_likes_q10',
        'channelId_likes_q25',
        'channelId_likes_q50',
        'channelId_likes_q75',
        'channelId_likes_q90',
        'channelId_likes_std',
        'channelId_likes_sum',
        'channelId_tags_point_count',
        'channelId_tags_point_max',
        'channelId_tags_point_max_diff',
        'channelId_tags_point_max_minus_q50',
        'channelId_tags_point_max_minus_q75',
        'channelId_tags_point_max_minus_q90',
        'channelId_tags_point_mean',
        'channelId_tags_point_mean_diff',
        'channelId_tags_point_mean_minus_q50',
        'channelId_tags_point_mean_minus_q75',
        'channelId_tags_point_mean_minus_q90',
        'channelId_tags_point_min',
        'channelId_tags_point_min_diff',
        'channelId_tags_point_q10',
        'channelId_tags_point_q25',
        'channelId_tags_point_q50',
        'channelId_tags_point_q75',
        'channelId_tags_point_q90',
        'channelId_tags_point_std',
        'channelId_tags_point_sum',
        'collection_date_day',
        'collection_date_day_cos',
        'collection_date_day_sin',
        'collection_date_dayofweek',
        'collection_date_dayofweek_cos',
        'collection_date_dayofweek_sin',
        'collection_date_is_weekend',
        'collection_date_month',
        'collection_date_month_cos',
        'collection_date_month_sin',
        'collection_date_quarter',
        'collection_date_quarter_cos',
        'collection_date_quarter_sin',
        'collection_date_week',
        'collection_date_year',
        'comment_count',
        'comment_count_log',
        'comment_count_per_day',
        'comments_disabled',
        'comments_dislike_ratio',
        'comments_like_ratio',
        'comments_likes',
        'conEn_description',
        'conEn_tags',
        'conEn_title',
        'delta',
        'delta_collection',
        'delta_log',
        'delta_published',
        'delta_sqrt',
        'description_ishttp_in',
        'description_len',
        'description_umap1',
        'description_umap10',
        'description_umap2',
        'description_umap3',
        'description_umap4',
        'description_umap5',
        'description_umap6',
        'description_umap7',
        'description_umap8',
        'description_umap9',
        'dislike_per_day',
        'dislikes',
        'dislikes_comments',
        'dislikes_log',
        'dislikes_log_bin',
        'dislikes_sqrt',
        'freq_categoryId',
        'freq_channelTitle',
        'in_OffChannell',
        'in_OffChannellJa',
        'in_OffJa',
        'in_cm_description',
        'in_cm_tags',
        'in_cm_title',
        'in_ff',
        'in_music_description',
        'in_music_tags',
        'in_music_title',
        'isEn_description',
        'isEn_tags',
        'isEn_title',
        'isJa_description',
        'isJa_tags',
        'isJa_title',
        'like_dislike_ratio',
        'like_per_day',
        'likes',
        'likes_comments',
        'likes_log',
        'likes_log_bin',
        'likes_sqrt',
        'publishedAt_day',
        'publishedAt_day_cos',
        'publishedAt_day_sin',
        'publishedAt_dayofweek',
        'publishedAt_dayofweek_cos',
        'publishedAt_dayofweek_sin',
        'publishedAt_hour',
        'publishedAt_is_weekend',
        'publishedAt_minute',
        'publishedAt_month',
        'publishedAt_month_cos',
        'publishedAt_month_sin',
        'publishedAt_quarter',
        'publishedAt_quarter_cos',
        'publishedAt_quarter_sin',
        'publishedAt_week',
        'publishedAt_year',
        'ratings_disabled',
        'tags_count_en',
        'tags_count_ja',
        'tags_length',
        'tags_num',
        'tags_point',
        'title_len',
        'title_umap1',
        'title_umap10',
        'title_umap2',
        'title_umap3',
        'title_umap4',
        'title_umap5',
        'title_umap6',
        'title_umap7',
        'title_umap8',
        'title_umap9',
        # 'publishedAt_month_publishedAt_dayofweek_dislikes_mean',
        # 'publishedAt_month_publishedAt_dayofweek_dislikes_mean_diff',
        # 'publishedAt_month_publishedAt_dayofweek_likes_mean',
        # 'publishedAt_month_publishedAt_dayofweek_likes_mean_diff',
        # 'publishedAt_year_publishedAt_month_dislikes_mean',
        # 'publishedAt_year_publishedAt_month_dislikes_mean_diff',
        # 'publishedAt_year_publishedAt_month_likes_mean',
        # 'publishedAt_year_publishedAt_month_likes_mean_diff',
        'publishedAt_year_publishedAt_month_likes_mean',
        'publishedAt_year_publishedAt_month_likes_mean_diff',
    ]

    # mlflowトラッキングの設定
    mlflow.set_tracking_uri(TRACKING_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name=run_name)

    try:
        # インスタンス生成
        runner = youtube_Runner(run_name, model_cls, features, setting, model_params, cv)
        use_feature_name = runner.get_feature_name()  # 今回の学習で使用する特徴量名を取得

        # モデルのconfigをjsonで保存
        value_list = [use_feature_name, model_params, cv, setting]
        save_model_config(key_list, value_list, dir_name, run_name)

        # 学習・推論
        runner.run_train_cv()
        runner.run_predict_cv()

        # submit作成
        Submission.create_submission(run_name, dir_name, setting.get('target'))

        if model_type == 'lgb':
            # feature_importanceを計算
            ModelLGB.calc_feature_importance(dir_name, run_name, use_feature_name, cv.get('n_splits'), type='gain')
            mlflow.log_artifact(dir_name + run_name + '_fi_gain.png')

    except Exception as e:
        print(traceback.format_exc())
        print(f'ERROR:{e}')


if __name__ == '__main__':
    fire.Fire(main)
