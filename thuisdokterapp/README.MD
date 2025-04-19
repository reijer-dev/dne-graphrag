Thank you for the clarification! Let's update the `README.md` to correctly reflect the location of the files. Here's the modified version:

---

## 🧠 Thuisdokter GraphRAG

A **Retrieval-Augmented Generation (RAG)** system for Dutch medical documents, combining **OpenAI GPT** with a structured **RDF knowledge graph** for improved answer quality and traceability.

---

### 🚀 Features

- ✅ **Triple extraction** from raw `.txt` documents using OpenAI
- ✅ **RDF (.ttl) graph generation** using `rdflib`
- ✅ **Streamlit app** for comparing GPT-only vs GraphRAG answers
- ✅ **Token usage, cost**, and **quality metrics** (BLEU, ROUGE-L, Cosine)
- ✅ **Batch evaluation** from CSV
- ✅ **Modular Python class** `MichaelRag` for integration

---

### 📦 Requirements

Install required libraries:

```bash
pip install -r requirements.txt
```

Example dependencies:
```txt
openai
rdflib
streamlit
scikit-learn
matplotlib
nltk
sentence-transformers
networkx
```

---

### 📁 Project Structure

```bash
thuisdokterapp/
├── __init__.py                   # defines MichaelRag class
├── graph_rag_comparison_app.py    # RAG comparison + metrics
├── my_config.py                  # API keys & config
├── thuisdokter_graph.ttl         # output RDF graph
├── documents/                    # input .txt files
├── thuisdokter_graph_builder_modular.py  # triple extractor + RDF builder
├── main.py                       # CLI runner for single queries
└── batch_evaluate.py             # Batch evaluation from CSV
```

---

### ✅ Usage

#### 🧪 1. Run Interactive CLI

You can run the CLI for single queries with `main.py`:

```bash
python main.py
```

It will prompt you to build the knowledge graph and ask medical questions.

---

#### 📊 2. Batch Evaluation

To evaluate multiple questions, you can use a CSV file with a `question` column:

```bash
python batch_evaluate.py medical_questions_50.csv
```

Results will be saved to `graphrag_results.csv`.

---

#### 🧱 3. From Python (Jupyter / Scripts)

```python
from thuisdokterapp import MichaelRag

rag = MichaelRag()
rag.process_document()         # Build RDF graph (run only when new docs are added)
rag.graph_stats()              # Print graph stats (triples, subjects, predicates)
answer = rag.query("Wat is astma?")
print(answer)
```

---

### 📈 Metrics Tracked

| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram overlap |
| **ROUGE-L** | Longest Common Subsequence |
| **Cosine Similarity** | SBERT embeddings |
| **Token Usage** | Prompt + completion tokens |
| **Cost ($)** | Based on OpenAI pricing |
| **Efficiency** | Score per dollar |

---

### ✨ Sample Evaluation Output

| Question                  | GPT Answer | GraphRAG Answer | BLEU  | ROUGE-L | Cost ($) |
|---------------------------|------------|-----------------|-------|---------|----------|
| Wat is astma?              | ...        | ...             | 0.76  | 0.84    | 0.0031   |
| Hoe behandel je hoge bloeddruk? | ...    | ...             | 0.80  | 0.85    | 0.0028   |

---

### 📬 Credits

Developed with ❤️ for medical language understanding and semantic retrieval.

Need help? Want to contribute? Open an issue or pull request!

---

## 🧳 Next Steps

- **Batch evaluation**: Test the system on a larger set of questions (use the `batch_evaluate.py` script).
- **RDF graph**: Visualize the knowledge graph and explore how the context is used in queries.



