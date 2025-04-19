import os
from pathlib import Path
import openai
import weaviate
from weaviate.util import generate_uuid5
from my_config import (
    OPENAI_API_KEY,
    WCD_API_KEY,
    WCD_URL,
    DATABRICKS_ACCESS_TOKEN,
    DATABRICKS_SERVER_HOSTNAME,
)

# Set your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")  # or your Weaviate cluster URL


# Step 1: Load .txt files from 'documents' folder
def load_documents(folder_path="./documents"):
    data = []
    for file_path in Path(folder_path).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            title = lines[0].replace("Document title: ", "").strip()
            url = lines[1].replace("URL: ", "").strip()
            paragraph_title = lines[2].replace("Paragraph title: ", "").strip()
            text = "\n".join(lines[3:]).replace("Text: ", "").strip()
            data.append(
                {
                    "filename": file_path.name,
                    "title": title,
                    "url": url,
                    "paragraph_title": paragraph_title,
                    "text": text,
                }
            )
    return data


# Step 2: Extract triples with OpenAI
def extract_triples(text):
    prompt = f"""
Je bent een medisch kennis-extractie assistent. Haal uit onderstaande tekst duidelijke kennisdriehoeken (subject, relatie, object).

Tekst:
{text}

Formaat: 
[("subject1", "relatie1", "object1"), ("subject2", "relatie2", "object2"), ...]

Antwoord:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return eval(response["choices"][0]["message"]["content"])


# Step 3: Store extracted triples in Weaviate
def store_triples(triples, context_text):
    for subj, pred, obj in triples:
        data = {
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "context": context_text,
        }
        client.data_object.create(
            data_object=data,
            class_name="Triple",
            uuid=generate_uuid5(data),
        )


# Step 4: Ask a question using GraphRAG
def graphrag_query(question):
    response = (
        client.query.get("Triple", ["subject", "predicate", "object", "context"])
        .with_near_text({"concepts": [question]})
        .with_limit(5)
        .do()
    )

    hits = response["data"]["Get"]["Triple"]
    if not hits:
        return "Geen relevantie gevonden."

    context = "\n".join(
        [f"{hit['subject']} — {hit['predicate']} — {hit['object']}" for hit in hits]
    )

    final_prompt = f"""
Gebruik de volgende kennisdriehoeken om de vraag te beantwoorden:

{context}

Vraag: {question}
Antwoord:
"""
    answer = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0,
    )
    return answer["choices"][0]["message"]["content"]


# Optional: Create schema once
def create_schema():
    if not client.schema.contains({"class": "Triple"}):
        client.schema.create_class(
            {
                "class": "Triple",
                "vectorizer": "text2vec-openai",
                "properties": [
                    {"name": "subject", "dataType": ["text"]},
                    {"name": "predicate", "dataType": ["text"]},
                    {"name": "object", "dataType": ["text"]},
                    {"name": "context", "dataType": ["text"]},
                ],
            }
        )


# MAIN RUN
if __name__ == "__main__":
    create_schema()
    docs = load_documents("./documents")
    for doc in docs:
        print(f"⏳ Extracting from: {doc['filename']}")
        triples = extract_triples(doc["text"])
        print(f"📦 Found {len(triples)} triples")
        store_triples(triples, doc["text"])
    print("✅ All triples stored in Weaviate.")
