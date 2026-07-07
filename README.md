# 🫁 X-Ray VLM — Vision Language Model for Medical Imaging

Built by **Jitin Varma** | [HuggingFace Space](https://huggingface.co/spaces/aspect29/xray-vlm)

---

## What is this?

A LLaVA-style Vision Language Model that answers natural language questions about chest X-rays. Upload an X-ray, ask a question, get a text answer.

```
"Is pneumonia present in this X-ray?" → "Yes, pneumonia is detected."
```

Built from scratch in 3 weeks on Google Colab free tier.

---

## Architecture

```
X-Ray Image
     ↓
CLIP ViT-L/14 (frozen)        — encodes image into 768-dim vector
     ↓
Projection Layer (trainable)  — maps 768 → 2048 dims
     ↓
[IMAGE TOKEN, question tokens] — prepended to text sequence
     ↓
TinyLlama 1.1B + LoRA         — generates text answer
     ↓
"Yes, pneumonia is detected."
```

This is the exact LLaVA architecture — bridging a vision encoder and a language model with a learned projection layer.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Image Encoder | CLIP ViT-L/14 (OpenAI) |
| Language Model | TinyLlama 1.1B |
| Fine-tuning | LoRA (PEFT) — only 0.2% of params trained |
| Quantization | 4-bit (bitsandbytes) |
| Dataset | HF Vision Chest X-Ray Pneumonia |
| UI | Gradio |
| Deployment | HuggingFace Spaces |

---

## Key Concepts Implemented

- **Multimodal bridging** — connecting vision and language model spaces via a trainable projection layer
- **LoRA fine-tuning** — parameter efficient fine-tuning, only 2.2M of 1.1B parameters trained
- **4-bit quantization** — fitting a 1.1B model on Colab free T4 GPU
- **Image-conditioned generation** — image embedding prepended to text token sequence with -100 label masking on image position
- **Streaming datasets** — loading 42GB dataset without downloading it

---

## How to Run

1. Open `xray_vlm.ipynb` in Google Colab
2. Set runtime to T4 GPU (Runtime → Change runtime type)
3. Run All
4. Upload a chest X-ray to the Gradio UI
5. Ask a question

---

## Results

Proof of concept trained for 100 steps on Colab free T4 GPU.

| Step | Loss |
|------|------|
| 0 | 3.06 |
| 10 | 0.99 |
| 50 | 0.00025 |
| 100 | 0.00007 |

Full training would require thousands of steps on sustained GPU compute for clinically meaningful results.

---

## Project Structure

```
xray-vlm/
├── xray_vlm.ipynb      # Full training notebook
├── README.md           # This file
```

---

## Disclaimer

This is a research/learning project. Not intended for clinical use.
