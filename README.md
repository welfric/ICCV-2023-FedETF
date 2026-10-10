# ICCV-2023-FedETF
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


>**This is the official implementation of ICCV 2023 paper "[No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic and Fixed Classifier](https://openaccess.thecvf.com/content/ICCV2023/html/Li_No_Fear_of_Classifier_Biases_Neural_Collapse_Inspired_Federated_Learning_ICCV_2023_paper.html) ([arixv](https://arxiv.org/abs/2303.10058))".**

## Paper Overview
**TLDR**: We devise FedETF which is inspired by the neural collapse phenomenon, showing both strong generalization and personalization performances.

**Abstract**: Data heterogeneity is an inherent challenge that hinders the performance of federated learning (FL). Recent studies have identified _the biased classifiers_ of local models as the key bottleneck. Previous attempts have used classifier calibration after FL training, but this approach falls short in improving the poor feature representations caused by training-time classifier biases. Resolving the classifier bias dilemma in FL requires a full understanding of the mechanisms behind the classifier. Recent advances in neural collapse have shown that the classifiers and feature prototypes under perfect training scenarios collapse into an optimal structure called simplex equiangular tight frame (ETF). Building on this neural collapse insight, we propose a solution to the FL's classifier bias problem by _utilizing a synthetic and fixed ETF classifier during training_. The optimal classifier structure enables all clients to learn unified and optimal feature representations even under extremely heterogeneous data. We devise several effective modules to better adapt the ETF structure in FL, achieving both high generalization and personalization. Extensive experiments demonstrate that our method achieves state-of-the-art performances on CIFAR-10, CIFAR-100, and Tiny-ImageNet.

![image](https://github.com/ZexiLee/ICCV-2023-FedETF/blob/main/figs/motiv_fig.png)

![image](https://github.com/ZexiLee/ICCV-2023-FedETF/blob/main/figs/framework_overview.png)

---
## Codebase Overview
- We implement our proposed FedETF and the FL baselines (FedAvg, FedDF, FedBE, FedDyn, FedProx, FedRoD, FedNH, Ditto, FedRep, FedProto, CCVR, etc.) into the same framework. We note that:
  - Running ''_main.py_'' for the methods.
  - For CCVR, it needs additional running ''_main_runccvr.py_'' for the final result since CCVR is a post-hoc method after federated training.
- We decouple the algorithms into the ''_server_method_'' and the ''_client_method_''. Therefore, the server and client methods can be flexibly combined.
- For convenience, we add the ''_set_server_method_'' function so that you can only set the client_method and the server_method can be assigned accordingly.
- The original algorithms can be reproduced by setting:
  ```
  FedETF: {'server_method': 'fedavg', 'client_method': 'fedetf'}.
  FedAvg: {'server_method': 'fedavg', 'client_method': 'local_train'}.
  FedDF: {'server_method': 'feddf', 'client_method': 'local_train'}.
  FedBE: {'server_method': 'fedbe', 'client_method': 'local_train'}.
  FedDyn: {'server_method': 'feddyn', 'client_method': 'feddyn'}.
  FedProx: {'server_method': 'fedavg', 'client_method': 'fedprox'}.
  FedRoD: {'server_method': 'fedavg', 'client_method': 'fedrod'}.
  FedProto: {'server_method': 'fedproto', 'client_method': 'fedproto'}.
  FedNH: {'server_method': 'fednh', 'client_method': 'fednh'}.
  CCVR: {'server_method': 'ccvr', 'client_method': 'ccvr'}.
  FedRep: {'server_method': 'fedrep', 'client_method': 'fedrep'}.
  Ditto: {'server_method': 'fedavg', 'client_method': 'ditto'}.
  FedSem: {'server_method': 'fedavg', 'client_method': 'fedsem'}.
  ```
- The values in the ''_Exp settings_'' block of ''_main.py_'' are used unless the same flag is given on the command line.

---
## FedSem: fixed semantic classifier from CLIP text embeddings
FedSem keeps FedETF's idea of a fixed classifier but chooses its geometry with class semantics. The server builds the anchors once (``fedsem.py``):
1. Encode every class name with a frozen CLIP text encoder under CLIP's 18 CIFAR prompt templates and average the normalized embeddings, giving $u_k$.
2. Centre and renormalize, $v_k = (u_k - \bar u)/\lVert u_k - \bar u\rVert$, to undo CLIP's narrow cone, and form $G_{sem} = VV^\top$.
3. Blend with the simplex ETF Gram matrix: $G_\lambda = (1-\lambda) G_{etf} + \lambda G_{sem}$ (PSD with unit diagonal).
4. Factorize $G_\lambda = Q\Lambda Q^\top$ and set $T_\lambda = [Q\Lambda^{1/2}\ 0] R$ with a fixed random rotation $R$, so every anchor has unit norm and $T_\lambda T_\lambda^\top = G_\lambda$.

Clients receive $T_\lambda$ once and never run CLIP. They train only the backbone and the projector with cross-entropy on $z = s\,T_\lambda f(x)$, where $f(x)$ is the unit-norm projected feature, and the server runs FedAvg. Prediction is $\arg\max_k t_k^\top f(x)$. The anchors are frozen exactly: they are a non-persistent buffer, so neither the optimizer nor the aggregation touches them.

| Flag | Default | Meaning |
|---|---|---|
| ``--fedsem_lambda`` | 0.5 | 0 gives a fixed simplex ETF, 1 gives centred CLIP anchors |
| ``--fedsem_scale`` | 10.0 | fixed logit scale $s$ |
| ``--fedsem_shuffle`` | 0 | 1 uses $\Pi T_\lambda$ for a random permutation $\Pi$ (same geometry, wrong classes) |
| ``--fedsem_learnable`` | 0 | 1 unfreezes $T_\lambda$, which is then trained and averaged like the other weights |
| ``--fedsem_loss`` | ce | ``balanced`` uses FedETF's balanced softmax loss instead, for like-for-like comparisons |
| ``--clip_backend`` | openai | ``openai`` (``pip install git+https://github.com/openai/CLIP.git``) or ``hf`` (``pip install transformers``) |
| ``--clip_model`` | ViT-B/32 | CLIP model name for the chosen backend (``openai/clip-vit-base-patch32`` for ``hf``) |
| ``--fedsem_text_emb`` | (none) | cache file for the class-name embeddings, loaded if it exists and written otherwise |

CLIP is only needed when ``--fedsem_lambda`` > 0 and the cache file does not exist yet. The anchor dimension is the model's projector width and must be at least the number of classes. ``E`` counts local epochs, as for every other method in this codebase.

Controls that isolate the value of the semantics (same seed, data partition and participation schedule):
```
python main.py --client_method fedsem --fedsem_lambda 0.5                       # semantic anchors
python main.py --client_method fedsem --fedsem_lambda 0.5 --fedsem_shuffle 1    # shuffled anchors at the same lambda
python main.py --client_method fedsem --fedsem_lambda 0                         # fixed ETF (FedETF's core)
python main.py --client_method fedsem --fedsem_lambda 0.5 --fedsem_learnable 1  # learnable head initialized at T_lambda
python main.py --client_method local_train                                      # FedAvg with its usual linear head
```
To compare anchors and nothing else, FedETF can run with the semantic anchors in place of its random ETF: ``--client_method fedetf --fedetf_anchors semantic`` (with ``--fedsem_lambda``, ``--fedsem_shuffle``, ``--clip_*`` and ``--fedsem_text_emb`` as above). Its balanced loss, learnable temperature, aggregation and personalization are unchanged, so it differs from ``--client_method fedetf`` only in the fixed classifier.
With ``--fedsem_lambda 0`` FedSem is FedETF's fixed ETF classifier without its extras (learnable temperature, balanced loss, alternating personalization of the proto and projection layer), so it is the like-for-like geometry arm; ``--client_method fedetf`` remains the full FedETF baseline. The shuffled control draws a permutation with no fixed points, so every class gets another class's anchor.
The rotation $R$ and the permutation $\Pi$ come from a generator seeded with ``--random_seed``, and encoding with CLIP leaves the global random state untouched, so these runs differ only in the classifier.
  
---
## Citing This Repository

Please cite our paper if you find this repo useful in your work:

```
@InProceedings{Li_2023_ICCV,
    author    = {Li, Zexi and Shang, Xinyi and He, Rui and Lin, Tao and Wu, Chao},
    title     = {No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic and Fixed Classifier},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {5319-5329}
}
```

---

## Contact

Please feel free to contact via email (<zexi.li@zju.edu.cn>) if you have further questions.
