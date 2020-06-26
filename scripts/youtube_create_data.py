import pandas as pd
import numpy as np
import re
import yaml
import time
import unicodedata
from functools import wraps
import category_encoders as ce
from tqdm import tqdm
from util import Logger
from bert_sentence_vectorizer import BertSequenceVectorizer
tqdm.pandas()

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']


def elapsed_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info_log('Start: {}'.format(f.__name__))
        # slack_notify(notify_type='1', title='Start: {}'.format(f.__name__), value=None, run_name=run_name, ts=ts)
        v = f(*args, **kwargs)
        logger.info_log('End: {} done in {:.2f} s'.format(f.__name__, (time.time() - start)))
        # slack_notify(notify_type='1', title='End: {} done in {:.2f} s'\
        # .format(f.__name__, (time.time() - start)), value=None, run_name=run_name, ts=ts)
        return v
    return wrapper


@elapsed_time
def load_data():
    """データの読み込み"""
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train_data.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test_data.csv')
    return train, test


def is_japanese(string):
    """日本の動画かどうかチェックする"""
    for ch in string:
        try:
            name = unicodedata.name(ch)
            if 'CJK UNIFIED' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                return True
        except Exception:
            continue
    return False


def create_day_feature(df, col, prefix):
    """日時特徴量の生成"""
    attrs = [
        'year',
        'quarter',
        'month',
        'week',
        'day',
        'dayofweek'
    ]
    for attr in attrs:
        dtype = np.int16 if attr == 'year' else np.int8
        df[prefix + '_' + attr] = getattr(df[col].dt, attr).astype(dtype)

    # 土日フラグ
    df[prefix + '_is_weekend'] = df[prefix + '_dayofweek'].isin([5, 6]).astype(np.int8)

    # 日付の周期性を算出
    def sin_cos_encode(df, col):
        df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
        df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
        return df

    for col in [prefix + '_' + 'quarter', prefix + '_' + 'month', prefix + '_' + 'day', prefix + '_' + 'dayofweek']:
        df = sin_cos_encode(df, col)

    return df


@elapsed_time
def get_title(df):
    """titleの特徴量を生成"""
    df['title'].fillna(' ', inplace=True)
    df['title_len'] = df['title'].apply(lambda x: len(x))
    return df


@elapsed_time
def get_published(df):
    """publishedAtの特徴量を生成"""
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], utc=True)
    df['publishedAt_hour'] = df['publishedAt'].apply(lambda x: x.hour)
    df['publishedAt_minute'] = df['publishedAt'].apply(lambda x: x.minute)
    df = create_day_feature(df, 'publishedAt', 'publishedAt')
    return df


@elapsed_time
def get_channel(df):
    """channelIdの特徴量を生成"""
    # エンコード
    cols = ['channelId']
    ce_oe = ce.OrdinalEncoder(cols=cols, handle_unknown='impute')
    df[cols] = ce_oe.fit_transform(df[cols])
    return df


@elapsed_time
def get_collection(df):
    """データレコードの収集日の特徴量を生成"""
    def change_to_Date(df):
        """collection_dateを日時の形式に変換"""
        df['collection_date'] = df['collection_date'].map(lambda x: x.split('.'))
        df['collection_date'] = df['collection_date'].map(lambda x: '20'+x[0]+'-'+x[2]+'-'+x[1])
        return pd.to_datetime(df['collection_date'], format='%Y-%m-%d', utc=True)

    # YYYY-MM-DDの形式に修正
    df['collection_date_after'] = change_to_Date(df)
    # 動画の再生回数収集日の時系列データの抽出
    df = create_day_feature(df, 'collection_date_after', 'collection_date')
    return df


@elapsed_time
def get_tag(df):
    """タグの特徴量を生成"""
    # [none] というタグが一定数あるので、これを欠損値ののマークとする
    df['tags'].fillna('[none]', inplace=True)
    tagdic = dict(pd.Series('|'.join(list(df['tags'])).split('|')).value_counts().sort_values())
    df['tags_num'] = df['tags'].astype(str).apply(lambda x: len(x.split('|')))
    df['tags_length'] = df['tags'].astype(str).apply(lambda x: len(x))
    df['tags_point'] = df['tags'].apply(lambda tags: sum([tagdic[tag] for tag in tags.split('|')]))
    df['tags_count_en'] = df['tags'].apply(lambda x: sum([bool(re.search(r'[a-zA-Z0-9]', x_)) for x_ in x.split('|')]))
    df['tags_count_ja'] = df['tags'].apply(lambda x: sum([is_japanese(x_) for x_ in x.split('|')]))
    return df


@elapsed_time
def get_like_dislike_comment(df):
    """likes/dislikes/commentの特徴量生成"""
    df['likes_log'] = np.log1p(df['likes'])
    df['likes_sqrt'] = np.sqrt(df["likes"])
    df['dislikes_log'] = np.log1p(df['dislikes'])
    df['dislikes_sqrt'] = np.sqrt(df["dislikes"])
    df['like_dislike_ratio'] = df['likes']/(df['dislikes']+1)
    df['comments_like_ratio'] = df['comment_count']/(df['likes']+1)
    df['comments_dislike_ratio'] = df["comment_count"]/(df["dislikes"]+1)
    df['comment_count_log'] = np.log1p(df['comment_count'])
    df['likes_comments'] = df['likes'] * df['comments_disabled']
    df['dislikes_comments'] = df['dislikes'] * df['comments_disabled']
    df['comments_likes'] = df['comment_count'] * df['ratings_disabled']
    return df


@elapsed_time
def get_description(df):
    """descriptionの特徴量を生成"""
    df["description"].fillna(' ', inplace=True)
    df['description_len'] = df['description'].apply(lambda x: len(x))
    df['description_ishttp_in'] = df['description'].apply(lambda x: x.lower().count('http'))
    return df


@elapsed_time
def get_boolean(df):
    """bool型を0/1に変更"""
    # bool型を 0 or 1に変換する
    df['comments_disabled'] = df['comments_disabled'] * 1
    df['ratings_disabled'] = df['ratings_disabled'] * 1
    return df


@elapsed_time
def get_delta(df):
    """日付の差分特徴量を生成"""
    df['delta'] = (df['collection_date_after'] - df['publishedAt']).apply(lambda x: x.days)
    df['delta_log'] = np.log(df['delta'])
    df['delta_sqrt'] = np.sqrt(df['delta'])
    df['delta_published'] = (df['publishedAt'] - df['publishedAt'].min()).apply(lambda x: x.days)
    df['delta_collection'] = (df['collection_date_after'] - df['collection_date_after'].min()).apply(lambda x: x.days)

    # 1日あたりのlike数、dislike数、コメント数
    df['like_per_day'] = df['likes'] / df['delta']
    df['dislike_per_day'] = df['dislikes'] / df['delta']
    df['comment_count_per_day'] = df['comment_count'] / df['delta']
    return df


@elapsed_time
def get_language(df):
    """言語の特徴量を生成"""
    # is japanese
    df['isJa_title'] = df['title'].apply(lambda x: is_japanese(x))
    df['isJa_tags'] = df['tags'].apply(lambda x: is_japanese(x))
    df['isJa_description'] = df['description'].apply(lambda x: is_japanese(x))

    # is english
    df['isEn_title'] = df['title'].apply(lambda x: x.encode('utf-8').isalnum())
    df['isEn_tags'] = df['tags'].apply(lambda x: x.encode('utf-8').isalnum())
    df['isEn_description'] = df['description'].apply(lambda x: x.encode('utf-8').isalnum())

    # cotain english
    df['conEn_title'] = df['title'].apply(lambda x: len(re.findall(r'[a-zA-Z0-9]', x.lower())))
    df['conEn_tags'] = df['tags'].apply(lambda x: len(re.findall(r'[a-zA-Z0-9]', x.lower())))
    df['conEn_description'] = df['description'].apply(lambda x: len(re.findall(r'[a-zA-Z0-9]', x.lower())))
    return df


@elapsed_time
def get_in_word(df):
    """特定の単語が自然言語特徴量に含まれているか否かの特徴量を生成"""
    # Music
    df['in_music_title'] = df['title'].apply(lambda x: 'music' in x.lower())
    df['in_music_tags'] = df['tags'].apply(lambda x: 'music' in x.lower())
    df['in_music_description'] = df['description'].apply(lambda x: 'music' in x.lower())
    # Official
    df['in_ff'] = df['title'].apply(lambda x: 'fficial' in x.lower())
    df['in_OffChannell'] = df['channelTitle'].apply(lambda x: 'fficial' in x.lower())
    df['in_OffJa'] = df['title'].apply(lambda x: '公式' in x.lower())
    df['in_OffChannellJa'] = df['channelTitle'].apply(lambda x: '公式' in x.lower())
    # cm
    df['in_cm_title'] = df['title'].apply(lambda x: 'cm' in x.lower())
    df['in_cm_tags'] = df['tags'].apply(lambda x: 'cm' in x.lower())
    df['in_cm_description'] = df['description'].apply(lambda x: 'cm' in x.lower())
    return df


@elapsed_time
def get_freq(df):
    """出現頻度"""
    for col in ['categoryId', 'channelTitle']:
        freq = df[col].value_counts()
        df['freq_'+col] = df[col].map(freq)
    return df


def get_agg(df, target_col, agg_target_col):
    """引数カラムの集計特徴量を生成する"""

    # カラム名を定義
    target_col_name = ''
    for col in target_col:
        target_col_name += str(col)
        target_col_name += '_'

    gr = df.groupby(target_col)[agg_target_col]
    df[f'{target_col_name}{agg_target_col}_mean'] = gr.transform('mean').astype('float16')
    df[f'{target_col_name}{agg_target_col}_max'] = gr.transform('max').astype('float16')
    df[f'{target_col_name}{agg_target_col}_min'] = gr.transform('min').astype('float16')
    df[f'{target_col_name}{agg_target_col}_std'] = gr.transform('std').astype('float16')
    df[f'{target_col_name}{agg_target_col}_count'] = gr.transform('count').astype('float16')
    df[f'{target_col_name}{agg_target_col}_sum'] = gr.transform('sum').astype('float16')

    # quantile
    # 10%, 25%, 50%, 75%, 90%
    q10 = gr.quantile(0.1).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q10'}, axis=1)
    q25 = gr.quantile(0.25).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q25'}, axis=1)
    q50 = gr.quantile(0.5).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q50'}, axis=1)
    q75 = gr.quantile(0.75).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q75'}, axis=1)
    q90 = gr.quantile(0.9).reset_index().rename({agg_target_col: f'{target_col_name}{agg_target_col}_q90'}, axis=1)
    df = pd.merge(df, q10, how='left', on=target_col)
    df = pd.merge(df, q25, how='left', on=target_col)
    df = pd.merge(df, q50, how='left', on=target_col)
    df = pd.merge(df, q75, how='left', on=target_col)
    df = pd.merge(df, q90, how='left', on=target_col)

    # 差分
    df[f'{target_col_name}{agg_target_col}_max_minus_q90']\
        = df[f'{target_col_name}{agg_target_col}_max'] - df[f'{target_col_name}{agg_target_col}_q90']
    df[f'{target_col_name}{agg_target_col}_max_minus_q75']\
        = df[f'{target_col_name}{agg_target_col}_max'] - df[f'{target_col_name}{agg_target_col}_q75']
    df[f'{target_col_name}{agg_target_col}_max_minus_q50']\
        = df[f'{target_col_name}{agg_target_col}_max'] - df[f'{target_col_name}{agg_target_col}_q50']
    df[f'{target_col_name}{agg_target_col}_mean_minus_q90']\
        = df[f'{target_col_name}{agg_target_col}_mean'] - df[f'{target_col_name}{agg_target_col}_q90']
    df[f'{target_col_name}{agg_target_col}_mean_minus_q75']\
        = df[f'{target_col_name}{agg_target_col}_mean'] - df[f'{target_col_name}{agg_target_col}_q75']
    df[f'{target_col_name}{agg_target_col}_mean_minus_q50']\
        = df[f'{target_col_name}{agg_target_col}_mean'] - df[f'{target_col_name}{agg_target_col}_q50']

    # 自身の値との差分
    # df[f'{target_col}_{agg_target_col}_mean_diff'] = df[target_col] - df[f'{target_col}_{agg_target_col}_mean']
    # df[f'{target_col}_{agg_target_col}_max_diff'] = df[target_col] - df[f'{target_col}_{agg_target_col}_max']
    # df[f'{target_col}_{agg_target_col}_min_diff'] = df[target_col] - df[f'{target_col}_{agg_target_col}_min']

    df[f'{target_col_name}{agg_target_col}_mean_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_mean']
    df[f'{target_col_name}{agg_target_col}_max_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_max']
    df[f'{target_col_name}{agg_target_col}_min_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_min']

    return df


@elapsed_time
def get_binning(df):
    """likesとdislikesをビニング処理する"""
    # likes
    bin_edges = [-float('inf'), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, float('inf')]
    _bin = pd.cut(
                    df['likes_log'],
                    bin_edges,
                    labels=False
    )
    df['likes_log_bin'] = _bin.values

    # dislikes
    bin_edges = [-float('inf'), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
    _bin = pd.cut(
                    df['dislikes_log'],
                    bin_edges,
                    labels=False
    )
    df['dislikes_log_bin'] = _bin.values

    return df


@elapsed_time
def get_agg_features(df):
    """カテゴリごとの集計特徴量生成処理"""
    # categoryIdごとの集計特徴量
    df = get_agg(df, ['categoryId'], 'likes')
    df = get_agg(df, ['categoryId'], 'dislikes')
    df = get_agg(df, ['categoryId'], 'comment_count')
    df = get_agg(df, ['categoryId'], 'delta')
    df = get_agg(df, ['categoryId'], 'tags_point')
    df = get_agg(df, ['categoryId'], 'like_dislike_ratio')
    df = get_agg(df, ['categoryId'], 'likes_comments')
    df = get_agg(df, ['categoryId'], 'dislikes_comments')
    df = get_agg(df, ['categoryId'], 'comments_likes')
    df = get_agg(df, ['categoryId'], 'like_per_day')
    df = get_agg(df, ['categoryId'], 'dislike_per_day')
    df = get_agg(df, ['categoryId'], 'comment_count_per_day')

    # channelIdごとの集計特徴量
    df = get_agg(df, ['channelId'], 'likes')
    df = get_agg(df, ['channelId'], 'dislikes')
    df = get_agg(df, ['channelId'], 'comment_count')
    df = get_agg(df, ['channelId'], 'delta')
    df = get_agg(df, ['channelId'], 'tags_point')
    df = get_agg(df, ['channelId'], 'like_dislike_ratio')
    df = get_agg(df, ['channelId'], 'likes_comments')
    df = get_agg(df, ['channelId'], 'dislikes_comments')
    df = get_agg(df, ['channelId'], 'comments_likes')
    df = get_agg(df, ['channelId'], 'like_per_day')
    df = get_agg(df, ['channelId'], 'dislike_per_day')
    df = get_agg(df, ['channelId'], 'comment_count_per_day')

    """
    # categoryId / channelIdごとの集計特徴量
    df = get_agg(df, ['categoryId', 'channelId'], 'likes')
    df = get_agg(df, ['categoryId', 'channelId'], 'dislikes')
    df = get_agg(df, ['categoryId', 'channelId'], 'comment_count')
    df = get_agg(df, ['categoryId', 'channelId'], 'delta')
    df = get_agg(df, ['categoryId', 'channelId'], 'tags_point')
    df = get_agg(df, ['categoryId', 'channelId'], 'like_dislike_ratio')
    df = get_agg(df, ['categoryId', 'channelId'], 'likes_comments')
    df = get_agg(df, ['categoryId', 'channelId'], 'dislikes_comments')
    df = get_agg(df, ['categoryId', 'channelId'], 'comments_likes')

    # likes_log_binごとの集計特徴量
    df = get_agg(df, ['likes_log_bin'], 'likes')
    df = get_agg(df, ['likes_log_bin'], 'dislikes')
    df = get_agg(df, ['likes_log_bin'], 'comment_count')
    df = get_agg(df, ['likes_log_bin'], 'delta')
    df = get_agg(df, ['likes_log_bin'], 'tags_point')
    df = get_agg(df, ['likes_log_bin'], 'like_dislike_ratio')
    df = get_agg(df, ['likes_log_bin'], 'likes_comments')
    df = get_agg(df, ['likes_log_bin'], 'dislikes_comments')
    df = get_agg(df, ['likes_log_bin'], 'comments_likes')

    # dislikes_log_binごとの集計特徴量
    df = get_agg(df, ['dislikes_log_bin'], 'likes')
    df = get_agg(df, ['dislikes_log_bin'], 'dislikes')
    df = get_agg(df, ['dislikes_log_bin'], 'comment_count')
    df = get_agg(df, ['dislikes_log_bin'], 'delta')
    df = get_agg(df, ['dislikes_log_bin'], 'tags_point')
    df = get_agg(df, ['dislikes_log_bin'], 'like_dislike_ratio')
    df = get_agg(df, ['dislikes_log_bin'], 'likes_comments')
    df = get_agg(df, ['dislikes_log_bin'], 'dislikes_comments')
    df = get_agg(df, ['dislikes_log_bin'], 'comments_likes')
    """

    # 年/月ごと
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'likes')
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'dislikes')
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'comment_count')
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'delta')
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'tags_point')
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'like_dislike_ratio')
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'likes_comments')
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'dislikes_comments')
    df = get_agg(df, ['publishedAt_year', 'publishedAt_month'], 'comments_likes')

    # その月/曜日ごと
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'likes')
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'dislikes')
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'comment_count')
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'delta')
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'tags_point')
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'like_dislike_ratio')
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'likes_comments')
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'dislikes_comments')
    df = get_agg(df, ['publishedAt_month', 'publishedAt_dayofweek'], 'comments_likes')

    return df


@elapsed_time
def get_title_vectorizer(df):
    """BERTからタイトルの文書ベクトルを取得する"""
    BSV = BertSequenceVectorizer()
    title_feature = df['title'].progress_apply(lambda x: BSV.vectorize(x))
    title_feature = pd.DataFrame(title_feature.values.tolist())
    title_feature = title_feature.add_prefix('title_vec')
    title_feature.to_pickle(FEATURE_DIR_NAME + 'title_vec.pkl')
    # 元のDFと結合
    df = pd.concat([df, title_feature], axis=1)
    return df


@elapsed_time
def get_desc_vectorizer(df):
    """BERTから説明文の文書ベクトルを取得する"""
    BSV = BertSequenceVectorizer()
    description_feature = df['description'].progress_apply(lambda x: BSV.vectorize(x))
    description_feature = pd.DataFrame(description_feature.values.tolist())
    description_feature = description_feature.add_prefix('description_vec')
    description_feature.to_pickle(FEATURE_DIR_NAME + 'description_vec.pkl')
    # 元のDFと結合
    df = pd.concat([df, description_feature], axis=1)
    return df


@elapsed_time
def concat_title_vectorizer(df):
    title_feature = pd.read_pickle(FEATURE_DIR_NAME + 'title_vec.pkl')
    df = pd.concat([df, title_feature], axis=1)
    return df


@elapsed_time
def concat_desc_vectorizer(df):
    description_feature = pd.read_pickle(FEATURE_DIR_NAME + 'description_vec.pkl')
    df = pd.concat([df, description_feature], axis=1)
    return df


@elapsed_time
def concat_title_umap_vectorizer(df):
    umap_df = pd.read_pickle(FEATURE_DIR_NAME + 'title_umap_10d.pkl')
    df = pd.concat([df, umap_df], axis=1)
    return df


@elapsed_time
def concat_description_umap_vectorizer(df):
    umap_df = pd.read_pickle(FEATURE_DIR_NAME + 'description_umap_10d.pkl')
    df = pd.concat([df, umap_df], axis=1)
    return df


@elapsed_time
def main(create_title_vec, create_desc_vec, create_title_umap):

    logger.info_log('★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆')

    # laod data
    train, test = load_data()
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

    # title
    df = get_title(df)
    # publishedAt
    df = get_published(df)
    # channel
    df = get_channel(df)
    # collection_date
    df = get_collection(df)
    # tag
    df = get_tag(df)
    # likes/dislikes/comment
    df = get_like_dislike_comment(df)
    # description
    df = get_description(df)
    # boolean特徴量
    df = get_boolean(df)
    # 日付のdelta特徴量
    df = get_delta(df)
    # language
    df = get_language(df)
    # 特定の単語が含まれるかどうか
    df = get_in_word(df)
    # 出現頻度
    df = get_freq(df)

    # ビニング処理
    df = get_binning(df)

    # カテゴリ特徴量ごとの集計特徴量生成
    df = get_agg_features(df)

    # titleの文章ベクトルをBERTから取得
    if create_title_vec:
        df = get_title_vectorizer(df)
    else:
        df = concat_title_vectorizer(df)

    # descriptionの文章ベクトルをBERTから取得
    if create_desc_vec:
        df = get_desc_vectorizer(df)
    else:
        df = concat_desc_vectorizer(df)

    # titleの特徴量をUMAPで圧縮したデータを取得
    if create_title_umap:
        pass
    else:
        df = concat_title_umap_vectorizer(df)

    # descriptionの特徴量をUMAPで圧縮したデータを取得
    if create_title_umap:
        pass
    else:
        df = concat_description_umap_vectorizer(df)

    # trainとtestに分割
    train = df.iloc[:len(train), :]
    test = df.iloc[len(train):, :]

    logger.info_log('Save train test')
    train.to_pickle(FEATURE_DIR_NAME + 'train.pkl')
    test.to_pickle(FEATURE_DIR_NAME + 'test.pkl')
    logger.info_log(f'train shape: {train.shape}, test shape, {test.shape}')

    # 生成した特徴量のリスト
    features_list = list(set(df.columns) - {'y', 'id', 'title', 'video_id',
                                            'publishedAt', 'channelId', 'channelTitle',
                                            'collection_date', 'tags', 'thumbnail_link',
                                            'description', 'collection_date_after'})  # 学習に不要なカラムは除外
    # 特徴量リストの保存
    features_list = sorted(features_list)
    with open(FEATURE_DIR_NAME + 'features_list.txt', 'wt') as f:
        for i in range(len(features_list)):
            f.write('\'' + str(features_list[i]) + '\',\n')

    return 'main() Done!'


if __name__ == "__main__":
    global logger
    logger = Logger()

    # 全部の特徴量を作る場合はすべてTrueで実行する
    main(create_title_vec=False, create_desc_vec=False, create_title_umap=False)
