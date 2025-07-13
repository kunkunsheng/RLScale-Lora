## Granular Music Attribute Transformation with Proximal Policy Optimization Adapters for Diffusion Model（ Accepted by ACM MM 2025）

## 🗃️ Abstract
The rapid development of music diffusion models has provided diverse paths for music creation transformations. However, existing methods still lack continuous strength regulation over stylistic attributes—specifically, they cannot achieve scalable adjustment of intensity (e.g., smooth transitions between "gentle" and "intense" jazz) while preserving spectral-temporal coherence. To address this, we propose RLScale-LoRA, a two-stage finetuning framework built on a structurally modified low-rank adaptation (LoRA) architecture with scale layers. In Stage 1, we finetune the modified LoRA to specialize in capturing attribute-aware latent spaces on unseen/seen music data. Stage 2 trains lightweight scale layers via proximal policy optimization (PPO), where reward functions enforce intermediate spectral-temporal state stability. Therefore, our RLScale-LoRA achieves precise, continuous music attribute transformations.

## 🗃️ Project Structure
```plaintext
RLScale-Lora/
│
├── audio-sample/          # audio samples
├── audioldm_eval/         # dateset, data preprocessing and loading
├── src/modules/lora_model # lora model code
├── src/inference.py           # inference script
├── src/train.py               # training script
├── src/train_ppo.py           # training script
└── README.md              # this documentation
```
## Setup
To set up your python environment:
```
conda create -n RLScale-Lora python=3.9
conda activate RLScale-Lora

cd RLScale-Lora
pip install -r requirements.txt

```
## train music Attribute

Before training music attributes, you need to create a dataset of contrasting musical attributes—that is, one set of examples with the attribute and another without it to represent the musical concept. For ease of training, you can convert these audio samples into Mel‐spectrogram features and save them separately.
Then, you can run `src/train.py` and `src/train_PPO.py` to train the control of the music attribute intensity.
## Resources

  - Audio samples in audio-sample/ and audioldm
## Acknowledgements

We sincerely thank the following repositories and their authors for providing valuable references and inspiration for **RLScale-Lora**:
- [Concept Sliders] (https://github.com/rohitgandikota/sliders):
- [Terra] (https://github.com/zwebzone/terra):
- [AudioLDM] (https://github.com/haoheliu/AudioLDM):
Their work has greatly contributed to our research and implementation.


## 📖 Citation
