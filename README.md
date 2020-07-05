# Probspace YouTube動画視聴回数予測コンペ

## コンペ概要
YouTube APIとして公開されているメタデータを用いて、動画の視聴回数を予測する

### URL
https://prob.space/competitions/youtube-view-count


### Result
public 13th -> private 10th

## Get Started

下記リポジトリをクローンし環境変数にパスを設定したのち、下記コマンド実行で再現可能です。

```
git clone https://github.com/takapy0210/compe_base_scripts
export PYTHONPATH=上記をクローンしたpath
```

```sh
cd scripts
python youtube_create_data.py
python run.py
```

