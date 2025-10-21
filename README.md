#  Multi-Head Latent Attention (MHLA) — PyTorch Implementation

[![PyTorch](https://img.shields.io/badge/Built_with-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DeepSeek](https://img.shields.io/badge/Inspired_by-DeepSeekV3-black?logo=openai)](https://github.com/deepseek-ai)

###  Author
**Implementation by:** [Keerat Singh (Keeratking22)](https://github.com/Keeratking22)  
**Concept originally from:** [DeepSeek-V3 Research Paper (2025)](https://github.com/deepseek-ai)  
**Framework:** PyTorch  
**Language:** Python  

---

##  Disclaimer

This project is an **independent educational implementation** of the *Multi-Head Latent Attention (MHLA)* mechanism described in the **DeepSeek-V3** paper.  
I am **not affiliated with DeepSeek, MIT, or any related institution**.  
This work is shared **for research, learning, and open-source experimentation** only.

---

##  Overview

This repository provides a clean and reproducible **PyTorch implementation** of the *Multi-Head Latent Attention (MHLA)* block from **DeepSeek-V3**.  
MHLA extends traditional attention by introducing a **latent intermediate space** and **rotary positional embeddings (RoPE)** to better capture long-range dependencies and sequence order.

It aims to demonstrate how latent projections can improve efficiency and contextual richness in LLM-style architectures.

---

##  Key Features

- **Latent Projection (`dl`):**  
  Compresses input embeddings into a latent space before attention, reducing computational load.

- **Rotary Positional Embeddings (RoPE):**  
  Injects positional information using rotational transformations, eliminating explicit positional encodings.

- **Dual-Branch Query/Key Encoding:**  
  Combines static (`u`) and position-aware (`r`) representations for deeper contextual modeling.

- **Causal Masking:**  
  Uses a lower-triangular attention mask for autoregressive token prediction.

- **Multi-Head Parallelization:**  
  Splits attention across multiple heads to capture diverse latent subspaces.

---

## Code Structure

### 1 Rotary Positional Encoding (`Rope`)
Encodes token positions as rotations:
### 2 Multi-Head Latent Attention (multiheadlatentattention)
Implements the full latent attention mechanism:
⚙️ Forward Pass Summary
Input projection:
Inputs are mapped to latent query/key/value vectors.

Rotary encoding:
Applies RoPE to position-aware branches.

Concatenation:
Combines static (u) and rotary (r) representations.

Attention computation:
Scaled dot-product attention with dropout and causal masking.

Output projection:
Concatenated head outputs → final linear projection.

Example Usage
python
Copy code
import torch
from multihead_latent_attention import multiheadlatentattention

x = torch.randn(2, 128, 512)  # (batch, seq_len, dim)

attn = multiheadlatentattention(
    d_in=512,
    d_out=512,
    dl=256,
    num_head=8,
    dropout_rate=0.1,
    contextlenght=128
)

out = attn(x)
print(out.shape)  # (2, 128, 512)
#Research Reference
This implementation is based on:

DeepSeek-V3: Towards Efficient Latent-Attention Large Language Models
DeepSeek AI Research, 2025
Official Repository →

## Possible Applications
Efficient Transformer or LLM backbones

Positional encoding research

Educational and benchmarking experiments

Lightweight sequence modeling

## Citation
If you use or reference this work, please cite both the DeepSeek-V3 paper and this independent re-implementation:

java
Copy code
@software{keeratsingh2025mhla,
  author = {Keeratpreet Singh},
  title = {Multi-Head Latent Attention (PyTorch Implementation of DeepSeek-V3 Mechanism)},
  year = {2025},
  url = {https://github.com/Keeratking22}
}
💡 Contact
📧 keeratpreetsingh2@gmail.com
🌐 GitHub: Keeratpreetsingh

## License
This repository is released under the MIT License.
Original architecture © DeepSeek AI Research.
Implementation © 2025 Keeratpreet Singh.
