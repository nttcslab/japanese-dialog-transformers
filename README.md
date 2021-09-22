# japanese-dialog-transformers

**[日本語の説明文はこちら](README-jp.md)**

This repository provides the information necessary to evaluate the Japanese Transformer Encoder-decoder dialogue model provided by NTT on [fairseq](https://github.com/pytorch/fairseq).


---

| Table of contents. |
|-|
| [Update log](#update-log) |
| [Notice for using the codes](#before-using) |
| [Model download](#model-download) |
| [Quick start](#quick-start) |
| [LICENSE](LICENSE.md) |

---

## Update log

* Sept. 17, 2021: Published dialogue models (fairseq version `japanese-dialog-transformer-1.6B`), datasets(`JEmpatheticDialogues` and `JPersonaChat`) and evaluation codes.

---

## Notice for using the codes
The dialogue models provided are for evaluation and verification of model performance.
Before downloading these models, please read the [LICENSE](LICENSE.md) and [CAUTION](Notice-jp.md) documents. You can download and use these models only if you agree to the following three points.

1. [LICENSE](LICENSE.md)
2. To be used only for the purpose of evaluation and verification of this model, and not for the purpose of providing dialogue service itself.
3. Take all possible care and measures to prevent damage caused by the generated text, and take responsibility for the text you generate, whether appropriate or inappropriate.

### BibTeX
When publishing results using this model, please cite the following paper.
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
## Model download
- [Pre-trained model](https://www.dropbox.com/s/k3ugxmr7nw6t86l/japanese-dialog-transformer-1.6B.pt?dl=0)
- [Finetuned model with JPersonaChat](https://www.dropbox.com/s/e5ib6rhsbldup3v/japanese-dialog-transformer-1.6B-persona50k.pt?dl=0)
- [Finetuned model with JEmpatheticDialogues](https://www.dropbox.com/s/laqz0jcgxvpxiy0/japanese-dialog-transformer-1.6B-empdial50k.pt?dl=0)

---

## Quick start

The models published on this page can be used for utterance generation and additional fine-tuning using the scripts included in fairseq.

### Install dependent libraries
The verification environment is as follows.

- Python 3.8.10 on [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh)
- CUDA 11.1/10.2
- Pytorch 1.8.2 （For the installation commands, be sure to check the [official page](https://pytorch.org/get-started/locally/). We recommend using pip.)
- fairseq 1.0.0a0（Available only from github: validated commit ID was 8adff65ab30dd5f3a3589315bbc1fafad52943e7）
- sentencepiece 0.1.96

When installing fairseq, please check [Requirements and Installation](https://github.com/pytorch/fairseq#requirements-and-installation) of the official page and install the latest or verified version (8adff65) using `github clone`. Normal pip install will only install the older version 0.10.2.
If you want to run finetune with your own data, you need to install the standalone version of sentencepiece.



### fairseq-interactive
Since fairseq-interactive does not have any way to keep the context, it generates responses based on the input sentences only, which is different from the setting that uses the context in Finetune and the paper experiment, so it is easy to generate inappropriate utterances.

In the following command, a small value (10) is used for beam and nbest (number of output candidates) to make the results easier to read. In actual use, it would be better to set the number to 20 or more for better results.

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

### dialog.py
The system utilizes a context of about four utterances, which is equivalent to the settings used in the Finetune and paper experiments.

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

### Perplexity calculation on a specific data set
Computes the perplexity (ppl) on a particular dataset.
The lower the ppl, the better the model can represent the interaction on that dataset.

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

### Finetuning with Persona-chat and EmpatheticDialogues
By finetuning the Pretrained model with PersonaChat or EmpatheticDialogues, you can create a model that is almost identical to the finetuned model provided.

If you have your own dialogue data, you can place the data in the same format in data/*/raw and perform Finetune on that data. Please note, however, that we do not allow the release or distribution of Finetune models under the LICENSE. You can release your own data and let a third party run Finetune from this model.

#### Downloading and converting datasets

* [JEmpatheticDialogues](https://www.dropbox.com/s/rkzyeu58p48ndz3/japanese_empathetic_dialogues.xlsx?dl=0)
* [JPersonaChat](https://www.dropbox.com/s/sda9wzexh7ntlij/japanese_persona_chat.xlsx?dl=0)

Convert data from Excel to a simple input statement (src) and output statement (dst) format, where the same row in src and dst is the corresponding input/output pair. 50000 rows are split and output as a train.

~~~
python scripts/extract_ed.py japanese_empathetic_dialogues.xlsx data/empdial/raw/
~~~

---

## License

[LICENSE](LICENSE.md)
