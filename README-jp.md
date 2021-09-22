# japanese-dialog-transformers

このリポジトリでは，NTTが提供する，日本語Transformer Encoder-decoder対話モデルを，[fairseq](https://github.com/pytorch/fairseq)上で評価するために必要な情報を公開しています．

---

| 本ページの項目一覧 |
|-|
| [更新履歴](#更新履歴) |
| [ご利用の前に](#ご利用の前に) |
| [モデルのダウンロード](#モデルのダウンロード) |
| [提供モデルのご利用方法](#提供モデルのご利用方法) |
| [利用規約](LICENSE.md) |

---

## 更新履歴

* 2021/09/17 対話モデル（fairseq版 `japanese-dialog-transformer-1.6B`），データセット(`JEmpatheticDialogues` and `JPersonaChat`)，および評価用コードを公開しました．

---

## ご利用の前に
提供する対話モデルは，モデル性能の評価・検証用です。
これらのモデルをダウンロードする前に，[利用規約](LICENSE.md)と[注意文書](Notice-jp.md)をご確認ください。以下の３点にご同意いただける場合に限り，本モデルをダウンロード・ご利用いただけます。
1. [利用規約](LICENSE.md)
2. 本モデルの評価・検証目的にのみに利用し，対話サービスの提供自体を目的とする用途へ利用しないこと
3. 生成された文によって被害が生じないよう万全の配慮と対策をおこない，適切・不適切を問わず生成した文に対する責任を負うこと

### BibTeX
本モデルを利用した結果を公開する場合には、以下の論文を引用ください。
<!-- You can use the following BibTeX entry for citation if you find our method useful. -->
```BibTeX
@misc{sugiyama2021empirical,
      title={Empirical Analysis of Training Strategies of Transformer-based Japanese Chit-chat Systems}, 
      author={Hiroaki Sugiyama and Masahiro Mizukami and Tsunehiro Arimoto and Hiromi Narimatsu and Yuya Chiba and Hideharu Nakajima and Toyomi Meguro},
      year={2021},
      eprint={2109.05217},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---
## モデルのダウンロード
- 事前学習モデルのダウンロードは[こちら](https://www.dropbox.com/s/k3ugxmr7nw6t86l/japanese-dialog-transformer-1.6B.pt?dl=0)
- PersonaChatでファインチューンしたモデルのダウンロードは[こちら](https://www.dropbox.com/s/e5ib6rhsbldup3v/japanese-dialog-transformer-1.6B-persona50k.pt?dl=0)
- EmpatheticDialoguesでファインチューンしたモデルのダウンロードは[こちら](https://www.dropbox.com/s/laqz0jcgxvpxiy0/japanese-dialog-transformer-1.6B-empdial50k.pt?dl=0)

---

## 提供モデルのご利用方法

本ページで公開するモデルは、fairseqに含まれるスクリプトを用いて、発話文生成や追加のfine-tuningを行うことができます。

### 依存ライブラリのインストール
検証環境は以下の通りです．[miniconda](https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh)上で検証しています．
- Python 3.8.10
- CUDA 11.1/10.2
- Pytorch 1.8.2 （インストールコマンドは必ず[公式ページ](https://pytorch.org/get-started/locally/)をご確認ください．pipを推奨します．）
- fairseq 1.0.0a0（正式リリースではなく，`github clone`でのみダウンロード可能です．検証時のcommit IDは8adff65ab30dd5f3a3589315bbc1fafad52943e7です．）
- sentencepiece 0.1.96

fairseqのインストールに当たっては，公式ページの[Requirements and Installation](https://github.com/pytorch/fairseq#requirements-and-installation)をご確認いただき，`github clone`を利用して最新版もしくは検証済み版（8adff65）をインストールください．通常のpip installでは旧バージョンの0.10.2までしか入りません．

また，独自のデータでfinetuneを行う場合は，sentencepieceのスタンドアローン版のインストールが必要です．

### 一問一答の対話（fairseq-interactive）
一問一答形式の対話を行います．fairseq-interactiveは文脈を保持する手段がなく，入力文されたのみを見て応答を生成します．Finetune・論文実験時の文脈を利用する設定と異なっておりますので，やや不適切な発話が生成されやすくなっております．ご注意ください．

以下のコマンドでは，beam・nbest（出力候補数）について，結果を見やすくするため，小さめの値（10）を利用しております．実際に利用する場合は，20以上に設定するほうがよい結果を得られると思います．

~~~
fairseq-interactive data/sample/bin/ \
 --path checkpoints/persona50k-flat_1.6B_33avog1i_4.16.pt\
 --beam 10 \
 --seed 0 \
 --min-len 10 \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model data/dicts/sp_oall_32k.model \
--no-repeat-ngram-size 3 \
--nbest 10 \
--sampling \
--sampling-topp 0.9 \
--temperature 1.0 
~~~

### 文脈を保持する対話（dialog.py）
4発話程度の文脈を保持して対話を行います．Finetune・論文実験時相当の設定です．

~~~
python scripts/dialog.py data/sample/bin/ \
 --path checkpoints/dials5_1e-4_1li20zh5_tw5.143_step85.pt \
 --beam 80 \
 --min-len 10 \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model data/dicts/sp_oall_32k.model \
 --no-repeat-ngram-size 3 \
 --nbest 80 \
 --sampling \
 --sampling-topp 0.9 \
 --temperature 1.0 \
 --show-nbest 5
~~~

### 特定データセット上でのパープレキシティ計算
特定データセット上でのパープレキシティ（ppl）の計算を行います．
pplが低いほど，そのデータセットでの対話をモデルが表現できていると評価することができます．

~~~
fairseq-validate $DATA_PATH \
 --path $MODEL_PATH \
 --task translation \
 --source-lang src \
 --target-lang dst \
 --batch-size 2 \ 
 --ddp-backend no_c10d \
 --valid-subset test \ 
 --skip-invalid-size-inputs-valid-test 
~~~

### Persona-chat, EmpatheticDialoguesを利用したFinetune
PretrainモデルをPersona-chatやEmpatheticDialoguesでFinetuneすることで，提供するFinetune済みモデルとほぼ同じモデルを作成することができます．

また，独自の対話データをお持ちの場合は，data/*/raw内に同じ形式でデータを配置することで，そのデータでFinetuneを行うことも可能です．ただし，Finetuneを施したモデルを公開・配布することは利用規約上許可しておりませんのでご注意ください（独自データを公開し，第三者に本モデルからFinetuneさせることは可能です）．
#### データセットのダウンロードと変換

* [JEmpatheticDialogues](https://www.dropbox.com/s/rkzyeu58p48ndz3/japanese_empathetic_dialogues.xlsx?dl=0)
* [JPersonaChat](https://www.dropbox.com/s/sda9wzexh7ntlij/japanese_persona_chat.xlsx?dl=0
)

データをExcelからシンプルな入力文（src）・出力文（dst）の形式に変換します．srcとdstの同じ行が対応する入出力のペアとなります．50000行をtrainとして分割出力します．
~~~
python scripts/extract_ed.py japanese_empathetic_dialogues.xlsx data/empdial/raw/
~~~

---

## License

[利用規約](LICENSE.md)
