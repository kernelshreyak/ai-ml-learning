# AI ML Learning

Repository of AI, ML, CV, NLP, and LLM experiments, notebooks, and scripts.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kernelshreyak/ai-ml-learning/HEAD?urlpath=%2Fdoc%2Ftree%2Fsample.ipynb)

## Quick Start

Create a virtual environment and install the dependency set that matches the kind of work you want to do.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Use the ML stack when you work on notebooks or scripts that need PyTorch, TensorFlow, computer vision, or reinforcement learning packages:

```bash
pip install -r requirements-ml.txt
```

For notebooks, launch Jupyter after installation:

```bash
jupyter lab
```

## Dependency Sets

- `requirements.txt`: core data analysis, notebook, web, and LLM tooling.
- `requirements-ml.txt`: extends the core set with CPU PyTorch wheels and the heavier ML stack.

## Repository Areas

- `datascience/`: stats, exploratory analysis, and classical ML experiments.
- `deep-learning/`: neural networks, TensorFlow exercises, and coursework notebooks.
- `language-models/`: small language modeling experiments and inference demos.
- `llama3.2-fine-tuning/`: LLaMA fine-tuning and merge/testing workflows.
- `peft-qlora/`: QLoRA reasoning fine-tuning pipeline with SFT and GRPO stages.
- `multi-agent-llm/`: multi-agent agentic workflows built with LangChain and LangGraph.
- `rag-pipelines/`: retrieval-augmented generation experiments and CLIP tests.
- `reinforcement-learning/`: bandits, CartPole, and Panda robotics RL experiments.
- `object-detection/`: YOLO and related detection scripts.
- `computer-vision-opencv/`: OpenCV-based experiments and image processing resources.
- `realtime-cv-web/`: real-time CV demo with Python and browser components.
- `text-to-speech/`: OpenAI audio and TTS examples.
- `azure-ai-learning/`: Azure AI and Azure ML learning notes and scripts.
- `sample_datasets/`: datasets used across notebooks and scripts.

## Notable Top-Level Notebooks And Scripts

- `computer-vision-learning.ipynb`
- `cv-test.ipynb`
- `detect-sleep-states.ipynb`
- `Learning_stable_diffusion.ipynb`
- `mnist-classifier.ipynb`
- `nlp_learning.ipynb`
- `pytorch_learning.ipynb`
- `random-forest.ipynb`
- `regularized-regression-linear.py`
- `semantic-segmentation.py`
- `svm-test.ipynb`

## Notes

- Some folders have their own README files with folder-specific setup or run commands.
- Several notebooks depend on external datasets, model weights, or API access.
- The ML dependency file includes CPU-only PyTorch wheels so it works without a CUDA setup by default.
