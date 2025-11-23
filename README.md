# ICLR2026-AERL (Anonymous)
## Abstract
Although considerable progress has been made toward enhancing the robustness of deep neural networks (DNNs), they continue to exhibit significant vulnerability to gradient-based adversarial attacks in supervised learning (SL) settings. We investigate adversarial robustness under reinforcement learning (RL), training image classifiers with policy-gradient objectives and 
ϵ-greedy exploration. When training models with several architectures on CIFAR-10, CIFAR-100, and ImageNet-100 datasets, RL consistently improves adversarial accuracy under white-box gradient-based attacks. Our results show that on a representative 6-layer CNN, adversarial accuracy increases from approximately 5% to 55% on CIFAR-10, 2% to 25% on CIFAR-100, and 5% to 18% on ImageNet-100, while clean accuracy decreases only 3–5% relative to SL. However, transfer analysis reveals that adversarial examples crafted on RL models transfer poorly: both SL and RL retain approximately 43% accuracy against these attacks. In contrast, adversarial examples crafted on SL models transfer effectively, reducing both SL and plain RL to around 8% accuracy. This indicates that while plain RL can prevent the generation of strong adversarial examples, it remains vulnerable to transferred attacks from other models, thus requiring adversarial training (RL-adv, ~30% adversarial accuracy) for comprehensive defense against cross-model attacks. Analysis of loss geometry and gradient dynamics shows that RL induces smaller gradient norms and rapidly changing input-gradient directions, reducing exploitable information for gradient-based attackers. Despite higher computational overhead, these findings suggest RL-based training can complement existing defenses by naturally smoothing loss landscapes, motivating hybrid approaches that combine SL efficiency with RL-induced gradient regularization. Anonymous code and configuration files are available at https://github.com/iclr2026aerl/ICLR2026-AERL.

## Install
### Environment
- Python 3.12
- CUDA 12.1 

### Cleverhans
For cleverhans, please use the cleverhans in our repository for better experience (the original one has memory leakage), see [here](https://github.com/cleverhans-lab/cleverhans/issues/1230). Unzip `cleverhans/cleverhans-fix.zip`, then:
```bash
pip install -e .
```

### Other dependencies
```bash
# from repo root
# torch: 2.3.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# scikit-learn: 1.5.1
pip install scikit-learn==1.5.1
# autoattack (a392200)
pip install git+https://github.com/fra31/auto-attack
# pandas: 2.2.2
pip install pandas==2.2.2
# tqdm: 4.66.4
pip install tqdm==4.66.4
# ImageNet-100 helper:
pip install datasets==4.0.0
```
if numpy problem, try:
```bash
pip install "numpy==1.26.4" --force-reinstall --no-cache-dir
```

## Running
### Datasets
- CIFAR-10/100: auto-downloaded by torchvision or run:
```bash
python dataset/download_dataset.py --dataset cifar10   # or cifar100
```
- ImageNet-100 (HF dataset):
```bash
python dataset/download_dataset.py --dataset imagenet100 --size 32

```
Data are stored under `dataset/` as compressed `.npz` files.

### Training
- Supervised (SL):
```bash
python sl_train/train.py \
  -- model cnn --dataset cifar10 --batch_size 256 \
  -- num_epochs 100 -- learning_rate 0.0002 \
  -- start_adv_epoch 80 --epsilon 0.01
```
- RL (policy-gradient + epsilon-greedy), e.g., CIFAR-10:
```bash
python rl_train/rl_train_cifar10.py \
  --batch_size 32 --num_epochs 100 --learning_rate 1e-3 \
  --model cnn --exploration_epsilon 0.1 --agent PolicyAdvEpsilonAgent
```
(Use `rl_train_cifar100.py` or `rl_train_im100.py` for other datasets.)

### Evaluation / Attacks
- PGD:
```bash
python attack/attack_standard_pgd.py \
  --model_path ../training_runs/cnn/cnn_best_model.pth \
  --dataset cifar10 --model cnn \
  --eps 1.0 --eps_iter 0.1 --nb_iter 100 --norm 2
```
- AutoAttack:
```bash
python attack/attack_autoattack.py \
  --model_path ../training_runs/cnn/cnn_best_model.pth \
  --dataset cifar10 --model cnn \
  --eps 1.0 --norm 2 --version standard
```







