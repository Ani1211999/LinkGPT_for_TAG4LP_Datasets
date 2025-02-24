# LinkGPT for TAG4LP Datasets


<p align="center">
    <a href="https://arxiv.org/abs/2406.04640" alt="arXiv">
        <img src="https://img.shields.io/badge/arXiv-2406.04640-b31b1b.svg?logo=arxiv&logoColor=fff" /></a>
    <a href="https://drive.google.com/file/d/1-_57MT-Mtp_oYnqSc0Kos7BpDBAyPuy5/view?usp=drive_link" alt="GoogleDrive Datasets">
        <img src="https://img.shields.io/badge/GoogleDrive-Datasets-4285F4?logo=googledrive&logoColor=fff" /></a>
    <a href="https://drive.google.com/file/d/17h3ToYyZFp9dcQ9FJjLL6KT-KvrN1BpH/view?usp=sharing" alt="GoogleDrive Models">
        <img src="https://img.shields.io/badge/GoogleDrive-Models-4285F4?logo=googledrive&logoColor=fff" /></a>
</p>

## Introduction

This repository uses LinkGPT framework as published in the paper [LinkGPT: Teaching Large Language Models To Predict Missing Links]( https://arxiv.org/abs/2406.04640 ) and the repository [LinkGPT](https://github.com/twelfth-star/LinkGPT) for performing prediction on the datasets mentioned in the [TAG4LP](https://github.com/ChenS676/TAG4LP) repository.


![Architecture of LinkGPT](assets/arch.png)

## Environment Preparation

```bash
# clone this repo
git clone https://github.com/Ani1211999/LinkGPT_for_TAG4LP_Datasets.git
cd LinkGPT

# create the conda environment
conda create -n linkgpt python=3.9
conda activate linkgpt

# install pytorch (refer to https://pytorch.org/get-started/previous-versions/ for other cuda versions)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install other dependencies
pip install -r requirements.txt
```

## Data Preparation

Download the data from [here](https://drive.google.com/file/d/1-_57MT-Mtp_oYnqSc0Kos7BpDBAyPuy5/view?usp=drive_link) and extract it into `LinkGPT/data`. You may also save the data in any other location, but you will need to modify the `LINKGPT_DATA_PATH` variable in all scripts accordingly. The structure of the data should look like the following tree diagram.

```bash
.
└── datasets
    ├── amazon_clothing_20k
    │   ├── dataset_for_lm.pkl
    │   ├── eval_yn_dataset_0_examples.pkl
    │   ├── eval_yn_dataset_2_examples.pkl
    │   ├── eval_yn_dataset_4_examples.pkl
    │   ├── eval_yn_dataset_large_candidate_set.pkl
    │   ├── ft_np_dataset.pkl
    │   ├── ft_yn_dataset.pkl
    │   ├── ppr_data.pt
    │   └── text_emb_cgtp.pt
    ├── amazon_sports_20k
    │   └── ... (same as above)
    ├── mag_geology_20k
    │   └── ...
    └── mag_math_20k
        └── ...

```
For the TAG4LP Dataset move the dataset in a folder rag_data/ 

![image](https://github.com/user-attachments/assets/6a372dde-0a82-4311-9b03-7978210509b6)

## Language Model Dataset Generator
Run the file _generate_lm_dataset.py_ present in the _linkgpt/dataset/_ folder by executing the command in the terminal as shown below.
Before running the file, edit the dataset parameters like name, text attributes, description attributes etc. in the file(default is for arxiv_2023 dataset)-

![image](https://github.com/user-attachments/assets/43cc433d-8f1c-4081-8578-c55aab417299)
![image](https://github.com/user-attachments/assets/cdd4ded3-8b0f-4401-9d55-5a25b7f0c90f)
![image](https://github.com/user-attachments/assets/52516819-da14-4554-9be2-ac66c4e03d00)

```bash
python linkgpt/dataset/generate_lm_dataset.py
```
## Text Embeddings and PPR scores for Pairwise Encoders Generation
Generate the `ppr_data.pt` and `text_emb_cgtp.pt` files based on `dataset_for_lm.pkl` by running the following command. Refer to the script for more details.

```bash
bash scripts/{dataset_name}/preparation.sh
```

## Link Prediction and Neighbor Prediction Dataset Generation
Generate the Link Prediction Dataset - `ft_yn_dataset.pkl` and  Neighbor Prediction Dataset - `ft_np_dataset.pkl` based on the `dataset_for_lm.pkl`.

Before running the following command edit the prompts for both neighbor and link prediction as per your dataset in the _linkgpt/dataset/yn_dataset.py_(for link prediction) and _linkgpt/dataset/np_dataset.py_ for neighbor prediction, as shown below.
![image](https://github.com/user-attachments/assets/49e8fd48-6094-4db5-a84c-867119907db5)
![image](https://github.com/user-attachments/assets/3eed3246-6658-4067-8a20-9ddc572771ea)

Execute the following command in the terminal-

```bash
bash scripts/{dataset_name}/run_np_lp_dataset_generation.sh
```

## Evaluation Dataset Generation
Generate the Evaluation Dataset - `eval_yn_dataset_4_examples.pkl`  based on the `dataset_for_lm.pkl`. Edit the prompts for the same in the  _linkgpt/dataset/yn_dataset.py_ as shown below.
![image](https://github.com/user-attachments/assets/aaf748f0-a7db-4bb9-9c61-361562ef5a87)

Execute the following command in the terminal-

```bash
bash scripts/{dataset_name}/run_eval_dataset_generation.sh
```

## Training

You may use the following command to train the model by yourself. The model checkpoints will be saved in `LinkGPT/data/models`.

```bash
bash scripts/{dataset_name}/train_linkgpt.sh
```

#Not Applicable for TAG4LP datasets
You may also download the fine-tuned models from [here](https://drive.google.com/file/d/17h3ToYyZFp9dcQ9FJjLL6KT-KvrN1BpH/view?usp=sharing
) and extract them into `LinkGPT/data`. T

The structure of the models should look like the following tree diagram.

```bash
└── models
    ├── amazon_clothing_20k
    │   └── linkgpt-llama2-7b-cgtp
    │       ├── stage1
    │       │   ├── linkgpt_special_token_emb.pt
    │       │   ├── lora_model
    │       │   │   ├── adapter_config.json
    │       │   │   ├── adapter_model.safetensors
    │       │   │   └── README.md
    │       │   ├── node_alignment_proj.pt
    │       │   ├── pairwise_alignment_proj.pt
    │       │   └── pairwise_encoder.pt
    │       └── stage2
    │           └── ... (same as stage1)
    ├── amazon_sports_20k
    │   └── ... (same as above)
    ├── mag_geology_20k
    │   └── ...
    └── mag_math_20k
        └── ...
```

## Evaluation

![Table 1](assets/table1.png)

To reproduce the results in Table 1 (LinkGPT w/o retrieval), you may use the following command to evaluate the model. The evaluation results will be saved in `LinkGPT/data/eval_output`.
 
```bash
bash scripts/{dataset_name}/eval_rank.sh
```

![Figure 4](assets/figure4.png)

To reproduce the results in Figure 4 (LinkGPT w/ retrieval), you may use the following commands to evaluate the model. The evaluation results will also be saved in `LinkGPT/data/eval_output`.

```bash
bash scripts/{dataset_name}/eval_retrieval_rerank.sh
```

## Citation

```bibtex
@article{he2024linkgpt,
  title={LinkGPT: Teaching Large Language Models To Predict Missing Links},
  author={He, Zhongmou and Zhu, Jing and Qian, Shengyi and Chai, Joyce and Koutra, Danai},
  journal={arXiv preprint arXiv:2406.04640},
  year={2024}
}
```
