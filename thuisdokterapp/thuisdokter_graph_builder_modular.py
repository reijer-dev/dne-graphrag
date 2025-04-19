
"""
thuisdokter_graph_builder.py

Modularized tool to extract knowledge triples from documents using OpenAI,
save them as JSONL, convert them to RDF (.ttl), and visualize the resulting graph.
"""

import os
import json
import time
from pathlib import Path
from urllib.parse import quote
from rdflib import Graph, URIRef, Namespace
from openai import OpenAI, RateLimitError
import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai


class ThuisdokterGraphBuilder:
    def __init__(self, documents_path="./documents", output_jsonl="thuisdokterapp/processed_triples.jsonl", output_ttl="thuisdokter_graph.ttl"):
        self.documents_path = documents_path
        self.output_jsonl = output_jsonl
        self.output_ttl = output_ttl
        self.EX = Namespace("http://example.org/")
        self.processed_files = None

        # Configure Gemini
        apikey = open("evaluation/apikey.txt", "r").read()
        genai.configure(api_key=apikey) # Replace with your actual API key
        self.gemini = genai.GenerativeModel('gemini-2.0-flash') # Or 'gemini-pro-vision' for multimodal models

    def load_documents(self):
        data = []
        for file_path in Path(self.documents_path).glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
                title = lines[0].replace("Document title: ", "").strip()
                text = "\n".join(lines[3:]).replace("Text: ", "").strip()
                data.append({
                    "filename": file_path.name,
                    "title": title,
                    "text": text
                })
        return data
    
    def load_document(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            title = lines[0].replace("Document title: ", "").strip()
            text = "\n".join(lines[3:]).replace("Text: ", "").strip()
            return {
                "filename": f.name,
                "title": title,
                "text": text
            }

    def already_processed(self):
        if not os.path.exists(self.output_jsonl):
            return set()
        with open(self.output_jsonl, "r") as f:
            return {json.loads(line)["filename"] for line in f}

    def safe_extract_triples(self, text, model="gpt-3.5-turbo", max_retries=3, delay=5):
        prompt = f"""
Je bent een medische kennis-extractie assistent. Haal uit onderstaande tekst in JSON duidelijke kennisdriehoeken (subject, relatie, object).

Tekst:
{text}

Formaat:
[("subject1", "relatie1", "object1"), ("subject2", "relatie2", "object2"), ...]

Antwoord:
"""
        for attempt in range(max_retries):
            try:
                response = self.gemini.generate_content(prompt)
                content = response.text.replace('```', '')[4:]
                return eval(content)
            except RateLimitError:
                print(f"⚠️ Rate limit hit. Waiting {delay} seconds...")
                time.sleep(delay)
            except Exception as e:
                print(f"❌ Error: {e}")
                return []
        return []

    def extract_and_save_triple(self, path):
        if self.processed_files is None:
            self.processed_files = self.already_processed()

        with open(self.output_jsonl, "a", encoding="utf-8") as out_f:
            doc = self.load_document(path)
            if doc["filename"] in self.processed_files:
                return
            
            triples = self.safe_extract_triples(doc["text"])

            out_f.write(json.dumps({
                "filename": doc["filename"],
                "title": doc["title"],
                "triples": triples
            }) + "\n")
            self.processed_files.add(doc["filename"])
            time.sleep(0.1)

    def sanitize_uri_part(self, text):
        if not text:
            return None
        return quote(text.strip().lower().replace(" ", "_").replace("/", "_"))

    def jsonl_to_rdf(self):
        g = Graph()
        skipped = 0

        with open(self.output_jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    entry = json.loads(line)
                    for triple in entry.get("triples", []):
                        if isinstance(triple, (list, tuple)) and len(triple) == 3:
                            subj, pred, obj = triple
                            s = URIRef(self.EX[self.sanitize_uri_part(subj)])
                            p = URIRef(self.EX[self.sanitize_uri_part(pred)])
                            o = URIRef(self.EX[self.sanitize_uri_part(obj)])
                            g.add((s, p, o))
                        else:
                            raise ValueError("Triple is not a list of 3 items")
                except Exception as e:
                    print(f"⚠️ Line {line_num} skipped: {e}")
                    skipped += 1

        g.serialize(destination=self.output_ttl, format="turtle")
        print(f"✅ RDF exported to: {self.output_ttl}")
        print(f"🚫 Skipped {skipped} malformed triples")

    def visualize_graph(self, max_nodes=50):
        g = Graph()
        g.parse(self.output_ttl, format="turtle")

        nxg = nx.DiGraph()

        for subj, pred, obj in g:
            nxg.add_edge(subj.split("/")[-1], obj.split("/")[-1], label=pred.split("/")[-1])
            if nxg.number_of_nodes() >= max_nodes:
                break

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(nxg, k=0.5)

        nx.draw(nxg, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
        edge_labels = nx.get_edge_attributes(nxg, 'label')
        nx.draw_networkx_edge_labels(nxg, pos, edge_labels=edge_labels, font_color='gray')

        plt.title("🔍 Visualized Knowledge Graph from Thuisdokter")
        plt.axis("off")
        plt.show()
