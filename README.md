## Setup

0. use python version >=12 && <13
1. check if requirements.txt needs platform specific changes for your machine
2. Use the requirements.txt file for the pip install
> pip install -r requirements.txt
3. Create an [API key](https://aistudio.google.com/apikey)
4. create /evaluation/apikey.txt with google api key as contents
5. enable Gemini AI for the current API key 
>https://console.cloud.google.com/marketplace/product/google/generativelanguage.googleapis.com?invt=AbuUwg&project=gen-lang-client-0620995921
6. download & install ollama https://ollama.com/download

## Results
In the code, the result of the training and inference evaluation is already included.

### Training
For VanillaRag, the file vanillarag_gemini_embeddings.json.zip contains the compressed embeddings of the documents. It is zipped to be able to upload it to Github.
For MichaelRag, the produced thuisdokter_graph.ttl, embeddings-cache.npz and processed_triples.jsonl are produced during training.
For LightRag, all files in light_rag/work_dir make up the total knowledge graph produced.

### Inference
evaluation-inference-1744869498.327011.txtx contains the raw questions, answers and scores.
evaluation-inference.csvy contains the data that is visualized in the Results section.
