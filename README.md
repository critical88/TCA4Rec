# 🧩 Token-level Collaborative Alignment for LLM-based Generative Recommendation

Official **PyTorch** implementation of our work **TCA4Rec** — *Token-level Collaborative Alignment for LLM-based Generative Recommendation*.

---

## 📦 Environment

The experiments are conducted under the following environment:

```
python==3.9.18
torch==2.5.0+cu118
transformers==4.51.3
pytorch-lightning==2.5.0.post0
peft==0.15.2
torch_scatter==2.1.2+pt25cu118
pandas==2.3.3
scikit-learn==1.6.1
```
## 📂 Data Preparation

We use the [Amazon 2018 dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
.
Please download the raw data first, and then preprocess it with the following script:

```bash
cd utils
python -u preprocess.py --dataset Toys_and_Games
```

For quick reproduction, we also provide three preprocessed datasets under the ./data/ directory.

## ⚙️ Running CF Models

For ease of integration, we provide a reference implementation of SASRec.
```bash
cd sasrec
python -u main.py --dataset Toys_and_Games --model SASRec --cuda 0
```

After training, the best checkpoint will be automatically saved at:

> sasrec/log/SASRec/Toys_and_Games/{time}/best_model.pth


Please move this checkpoint to the following path before running TCA4Rec:

> cf_model/sasrec/Toys_and_Games.pt

## 🚀 Running TCA4Rec

Before launching TCA4Rec, ensure that both the **CF model** and **LLM model** are correctly prepared and placed in the designated directories.

You can start training with:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
  --dataset Toys_and_Games \
  --model llm4rec \
  --use_msl --alpha 0.1 > game_tca4rec_msl.out 2>&1 &
```
## ⏱️ Training & Evaluation Time

Using `Llama-3.2-3B` on an `NVIDIA RTX A6000`, the runtime is approximately:

- **Training**: ~20 minutes per epoch

- **Evaluation**: ~30 minutes per round

<!-- ## 📘 Citation

If you find this work useful, please consider citing our paper:

@article{tca4rec2025,
  title={Token-level Collaborative Alignment for LLM-based Generative Recommendation},
  author={...},
  year={2025},
  journal={...}
} -->