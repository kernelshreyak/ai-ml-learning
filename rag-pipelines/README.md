## RAG Pipelines

This folder contains notebook experiments for retrieval-augmented generation and related model testing workflows.

### What This Folder Contains

- `agentic_hybrid_rag_with_bm25_rrf.ipynb` - notebook for adding an agent loop on top of a hybrid RAG pipeline that uses BM25 retrieval and RRF ranking.
- `CLIP_model_test.ipynb` - notebook for testing a CLIP model setup, based on the OpenAI CLIP reference implementation.

### Focus Areas

- Hybrid retrieval pipelines that combine lexical and ranking-based retrieval strategies.
- Agentic control loops layered on top of retrieval workflows.
- Model validation and experimentation with CLIP-based vision-language components.

### Running The Notebooks

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then open the notebooks from this folder in Jupyter or VS Code and run them as usual.

### Notes

- These notebooks are experimental and may rely on external datasets, model weights, or API access depending on the cells you run.
- If you add new RAG experiments here, keep the folder README updated so the notebook map stays accurate.
