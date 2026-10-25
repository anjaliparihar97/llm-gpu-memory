# How much GPU memory does an LLM really need?

A hands-on study of GPU memory when **running** and **training** large language models, measured on a free NVIDIA T4 GPU in Google Colab.

Most guides say quantization or LoRA "saves memory." I wanted to measure it: predict memory from first principles, run controlled experiments on the same model and data, and compare the prediction with reality, including the trade-offs in speed and quality. Some results surprised me.

## Key findings

- **Quantization saved memory but cost speed.** 4-bit cut weight memory by 60%, but generation was about 5× slower than 16-bit on the T4, and 8-bit was about 15× slower.
- **Long prompts were the biggest memory risk.** A 7,500-token prompt needed 6.4 GB on top of 1.1 GB of model weights.
- **Batching was the best optimization.** 16 prompts at once gave 7.6× more throughput for only 8% more memory.
- **In training, the weights are only a small part of the memory.** Weights were 1.84 GB, but the training step peaked at 9.27 GB. Gradients and Adam's optimizer states added exactly the amounts theory predicts.

## What's inside

| Notebook | Question | What it measures |
|---|---|---|
| [`01_inference_memory_benchmark.ipynb`](notebooks/01_inference_memory_benchmark.ipynb) | How much memory does it take to *run* a model? | 16-bit vs. 8-bit vs. 4-bit quantization, memory growth with prompt length, batching throughput |
| [`02_training_memory_benchmark.ipynb`](notebooks/02_training_memory_benchmark.ipynb) | Where does memory go when you *train* a model? | How gradients work, the anatomy of one training step, and seven training setups |

## Setup

- **Hardware:** NVIDIA Tesla T4 (14.6 GB usable), free tier of Google Colab
- **Models:** Qwen2.5 1.5B Instruct for inference (about 1.54 billion parameters), Qwen2.5 0.5B for training (494 million parameters)
- **Libraries:** PyTorch, Hugging Face Transformers, PEFT, bitsandbytes 0.50.2
- **Measurement:** peak GPU memory with `torch.cuda.max_memory_allocated`, speed in tokens per second or seconds per step, answers on a fixed test set
- **Method:** predict memory from the parameter count first, then measure

---

## Part 1: Running a model (inference)

### Quantization: memory, speed and quality

| Precision | Weights (GB) | Peak (GB) | Tokens/sec | First token (s) | Quality |
|---|---|---|---|---|---|
| 16-bit | 2.88 | 2.89 | 19.0 | 0.050 | 1.0 |
| 8-bit | 1.67 | 1.68 | 1.3 | 0.556 | 1.0 |
| 4-bit | 1.14 | 1.15 | 3.5 | 0.228 | 1.0 |

![Quantization memory](results/quantization_memory.png)

**Prediction vs. reality.** With about 1.54 billion parameters, I predicted 2.87 GB in 16-bit, 1.43 GB in 8-bit and 0.72 GB in 4-bit. 16-bit matched almost exactly. The quantized models used more than predicted because **not every layer is quantized**: the embedding layer stays in 16-bit, and in this model it's large (about 233 million parameters, because the vocabulary has about 152,000 tokens). Quantized layers also store extra scale values. Recalculating with the embeddings in 16-bit gives about 1.65 GB for 8-bit, within a few percent of the measured 1.67 GB.

**Quality.** All three versions answered the factual questions correctly (capital of Germany, 17 × 3, a German translation), and the explanations were very similar in content. My test set has only five questions, so this shows no visible difference here, not that quantization never affects quality.

**Speed.** This was the surprise. 4-bit was about 5× slower than 16-bit and 8-bit about 15× slower. The compressed weights have to be converted back to 16-bit during every calculation. For a small model that already fits comfortably in memory, there's nothing to gain from that trade. The 8-bit method in bitsandbytes is particularly slow because it handles unusual "outlier" values separately. **Quantization saves memory; it doesn't automatically make a model faster.** Speed gains depend on the hardware and on optimized inference engines.

### Prompt length

| Prompt tokens | Peak (GB) | Extra over weights (GB) | Seconds |
|---|---|---|---|
| 242 | 1.19 | 0.05 | 1.37 |
| 946 | 1.30 | 0.16 | 1.40 |
| 1,881 | 1.64 | 0.50 | 1.53 |
| 3,762 | 2.88 | 1.74 | 2.89 |
| 7,513 | 7.55 | 6.41 | 8.78 |

![Memory and prompt length](results/kv_cache_growth.png)

The longest prompt needed 6.41 GB on top of the 4-bit model's weights, more than five times the size of the model itself. Memory also grew **faster than the prompt**: doubling the prompt from about 3,800 to 7,500 tokens increased the extra memory by about 3.7 times.

The KV cache (the model's memory of tokens it has processed) grows in a straight line with prompt length, and it's small for this model, which uses grouped-query attention. Growth faster than linear points to **temporary working memory while processing the prompt**, especially the attention calculation, which grows roughly with the square of the prompt length. That's why techniques like FlashAttention, which avoid storing the full attention matrix, matter so much for long contexts.

### Batching

| Batch size | Total seconds | Tokens/sec | Peak (GB) |
|---|---|---|---|
| 1 | 34.90 | 9.3 | 1.15 |
| 4 | 15.27 | 21.3 | 1.18 |
| 8 | 9.46 | 34.4 | 1.19 |
| 16 | 4.59 | 70.8 | 1.24 |

![Batching throughput](results/batching_throughput.png)

Processing 16 prompts together raised throughput from 9.3 to 70.8 tokens per second, about 7.6 times more, while memory rose only from 1.15 to 1.24 GB. A GPU is built to do many calculations in parallel, so one prompt at a time leaves most of it unused.

### What I'd do in practice

To serve this model to many users on one T4, I'd use **16-bit, not quantization**: it fits easily at 2.9 GB and was 5 to 15 times faster. Then I'd **batch requests**, which gave the biggest gain for almost no memory, ideally with an inference engine like vLLM that batches continuously and manages memory efficiently. I'd also **limit the maximum prompt length**, since long prompts were the main memory risk. Quantization makes sense when a model wouldn't fit otherwise, for example a much larger model on the same GPU.

---

## Part 2: Training a model

### How gradients work, in ten lines

The notebook starts with one weight `w` learning that `w × 2` should equal 6. Each step computes the loss, calculates the gradient (which direction reduces the loss, and how strongly), and moves `w` a little against it.

| Step | w | Loss | Gradient |
|---|---|---|---|
| 0 | 0.500 | 25.000 | −20.000 |
| 1 | 1.500 | 9.000 | −12.000 |
| 2 | 2.100 | 3.240 | −7.200 |
| 4 | 2.676 | 0.420 | −2.592 |
| 7 | 2.930 | 0.020 | −0.560 |

`w` moves toward the correct value of 3, and the gradient shrinks as it gets closer, so the steps automatically get smaller. A large language model does the same for hundreds of millions of weights at once, which is why training needs so much more memory than inference: **every weight needs its own gradient, and the Adam optimizer keeps two more values per weight.**

### Anatomy of one training step

One full FP32 training step on Qwen2.5 0.5B (494 million parameters), batch of 4 sequences × 256 tokens:

| Phase | Memory allocated (GB) | What was added |
|---|---|---|
| 1. Weights loaded | 1.84 | The model's weights |
| 2. After forward pass | 6.07 | +4.23 GB of activations |
| 3. After backward pass | 3.71 | Activations freed, +1.87 GB of gradients |
| 4. After optimizer step | 7.42 | +3.71 GB of Adam states |
| **Peak during the step** | **9.27** | |

![Anatomy of a training step](results/anatomy_training_step.png)

**Prediction vs. reality.** In FP32 with Adam, each parameter costs about 16 bytes: 4 for the weight, 4 for its gradient and 8 for Adam's two running averages.

| Part | Predicted | Measured |
|---|---|---|
| Weights (494M × 4 bytes) | 1.84 GB | 1.84 GB |
| Gradients (same size as weights) | 1.84 GB | 1.87 GB |
| Adam states (2 × weights) | 3.68 GB | 3.71 GB |
| Total before activations (16 bytes per parameter) | 7.36 GB | 7.42 GB |

The theory matched almost exactly. What the simple formula leaves out is **activations**: the forward pass added 4.23 GB, more than twice the weights, for only 1,024 tokens. The peak of 9.27 GB happens during the backward pass, when gradients are being created while activations are still in memory. **The weights were only about 20% of the peak.**

<!--
### Seven training setups

Paste the results table from Step 5 here, then remove these comment lines.

![Peak memory by technique](results/peak_memory_by_technique.png)
![Memory vs. speed](results/memory_vs_speed.png)
-->

### What I learned from training

Training memory is mostly not the model. Gradients double the memory, Adam doubles it again, and activations can be larger than all of the weights, which explains why fine-tuning even a small model can run out of memory on a GPU where running it is easy. It also explains why techniques like 8-bit optimizers (which shrink Adam's states), gradient checkpointing (which shrinks activations) and LoRA (which avoids most gradients and optimizer states by training only small adapters) are so effective.

---

## How to run it

1. Open a notebook from the `notebooks` folder in [Google Colab](https://colab.research.google.com) (File → Upload notebook)
2. Runtime → Change runtime type → **T4 GPU** → Save
3. Run the install cell, then **Runtime → Restart session**, and continue from the next cell
4. Run the remaining cells from top to bottom (about 15 to 20 minutes per notebook)
5. Download the charts and CSV files from the Colab file browser

To run locally on an NVIDIA GPU: `pip install -r requirements.txt`, then open the notebooks in Jupyter.

## Limitations

- Results come from one GPU type (T4) and small models. Absolute numbers will differ on other hardware and model sizes; the relative patterns are what matter.
- The quality check uses only five prompts, enough to spot obvious damage, not to measure subtle quality differences.
- In my training run, the WikiText download failed, so the notebook used random tokens. Memory measurements are unaffected, since memory depends on tensor sizes rather than content, but the training loss values aren't meaningful.
- `max_memory_allocated` measures memory used by PyTorch tensors. Total GPU usage shown by `nvidia-smi` is higher because of the CUDA context and caching.

## Tools

Python · PyTorch · Hugging Face Transformers · PEFT · bitsandbytes · pandas · matplotlib · Google Colab

## Author

**Anjali Parihar**, data and AI engineer
[LinkedIn](https://www.linkedin.com/in/anjali-parihar-) · [Portfolio](https://anjaliparihar97.github.io)

## License

MIT
