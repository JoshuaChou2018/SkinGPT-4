# Pre-trained Multimodal Large Language Model Enhances Dermatological Diagnosis using SkinGPT-4

[Juexiao Zhou](https://www.joshuachou.ink/), Xiaonan He, Liyuan Sun, Jiannan Xu, Xiuying Chen, Yuetan Chu, Longxi Zhou, Xingyu Liao, Bin Zhang, Shawn Afvari, Xin Gao

King Abdullah University of Science and Technology, KAUST

<a href='s41467-024-50043-3.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>

## Installation

```
conda env create -f environment.yml
conda activate skingpt4_llama2
conda install -c conda-forge mamba=1.4.7
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Download our trained weights

**Our previous trained weights for skin disease diagnosis with only base dataset and Llama2 could be downloaded at [skingpt4_llama2_13bchat_base_pretrain_stage2.pth](https://drive.google.com/file/d/1tcwEKSBl8J7wUKBJDwptcH7AwB5Ge7iW/view).** Then modify line 10 at SkinGPT-4-llama2/eval_configs/skingpt4_eval_llama2_13bchat.yaml to be the path of SkinGPT-4 weight.

**Our previous trained weights for skin disease diagnosis with only step-1 dataset and Vicuna could be downloaded at [skingpt4_vicuna_v1.pth](https://drive.google.com/file/d/1PGBMBioipGxN5yfX6Okx4BGyPBm1prAF/view?usp=sharing).** Then modify line 11 at SkinGPT-4-llama2/eval_configs/skingpt4_eval_vicuna.yaml to be the path of SkinGPT-4 weight.

**Please note:**

- The released trained model above **cannot be used for skin disease diagnosis**, they can **only be used for testing code**.

- The latest model trained with both **public skin disease datasets** and the **proprietary skin disease dataset** based on **falcon-40b-instruct** (deprecated) and **llama-2-13b-chat-hf** (code published only) are **not publicly available** currently due to privacy issues.

- Please feel free to keep in touch with **xin.gao@kaust.edu.sa** and **juexiao.zhou@kaust.edu.sa** for potential collaboration.

## Prepare weight for LLMs

### Llama2 Version

```shell
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
```

Then modify line 16 at SkinGPT-4-llama2/skingpt4/configs/models/skingpt4_llama2_13bchat.yaml to be the path of Llama-2-13b-chat-hf.

### Vicuna Version

```shell
# download Vicuna’s **delta** weight
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0

# get llama-13b model
git clone https://huggingface.co/huggyllama/llama-13b

pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
pip install transformers==4.28.0

python -m fastchat.model.apply_delta --base ./llama-13b --target ./vicuna --delta ./vicuna-13b-delta-v0
```

Then modify line 16 at SkinGPT-4-llama2/skingpt4/configs/models/skingpt4_vicuna.yaml to be the path of vicuna.

## Launching Demo Locally

### Llama2 Version

```
python demo.py --cfg-path eval_configs/skingpt4_eval_llama2_13bchat.yaml  --gpu-id 0
```

### Vicuna Version

```
python demo.py --cfg-path eval_configs/skingpt4_eval_vicuna.yaml  --gpu-id 0
```

## Illustraion of SkinGPT-4

![Figure_1](https://cdn.jsdelivr.net/gh/JoshuaChou2018/oss@main/uPic/EQFQXz.Figure_1.png)

## Examples of Skin disease diagnosis

![Figure_3](https://cdn.jsdelivr.net/gh/JoshuaChou2018/oss@main/uPic/vXavYQ.Figure_3.png)



## Clinical Evaluation

![Figure_4](https://cdn.jsdelivr.net/gh/JoshuaChou2018/oss@main/uPic/vrHwZ4.Figure_4.png)



## Acknowledgement

- [MiniGPT-4](https://minigpt-4.github.io/) This repo is developped on MiniGPT-4, an awesome repo for vision-language chatbot!
- Lavis
- Vicuna
- Falcon
- Llama 2

## Citation

Our paper has been accepted by **Nature Communications**.

![image-20240708131953458](https://cdn.jsdelivr.net/gh/JoshuaChou2018/oss@main/uPic/ce5kAk.image-20240708131953458.png)

If you find SkinGPT-4 to be helpful in your research or applications, please cite SkinGPT-4 using this BibTeX:

```
@article{zhou2024pre,
  title={Pre-trained multimodal large language model enhances dermatological diagnosis using SkinGPT-4},
  author={Zhou, Juexiao and He, Xiaonan and Sun, Liyuan and Xu, Jiannan and Chen, Xiuying and Chu, Yuetan and Zhou, Longxi and Liao, Xingyu and Zhang, Bin and Afvari, Shawn and others},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={5649},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
