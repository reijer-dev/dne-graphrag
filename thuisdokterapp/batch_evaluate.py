
import pandas as pd
import argparse
from thuisdokterapp import MichaelRag

def main(input_csv, output_csv):
    rag = MichaelRag()

    print(f"📥 Loading questions from: {input_csv}")
    df = pd.read_csv(input_csv)

    results = []

    for idx, row in df.iterrows():
        question = row["question"]
        print(f"🔍 [{idx+1}/{len(df)}] Evaluating: {question}")
        answer = rag.query(question)

        # Get the most recent comparison log entry
        entry = rag.session_state["comparison_log"][-1] if "comparison_log" in rag.session_state else {}

        results.append({
            "question": question,
            "gpt_answer": entry.get("gpt_answer", ""),
            "graphrag_answer": entry.get("graphrag_answer", ""),
            "bleu": entry.get("bleu", 0),
            "rouge_l": entry.get("rouge_l", 0),
            "cosine_sim": entry.get("cosine_sim", 0),
            "gpt_tokens": entry.get("gpt_tokens", 0),
            "graphrag_tokens": entry.get("graphrag_tokens", 0),
            "gpt_cost": entry.get("gpt_cost", 0),
            "graphrag_cost": entry.get("graphrag_cost", 0),
            "total_tokens": entry.get("total_tokens", 0),
            "total_cost": entry.get("total_cost", 0),
            "efficiency_gpt": entry.get("efficiency_gpt", 0),
            "efficiency_graphrag": entry.get("efficiency_graphrag", 0),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"✅ Evaluation complete. Results saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate GraphRAG vs GPT")
    parser.add_argument("input_csv", help="Path to CSV file with a 'question' column")
    parser.add_argument("--output_csv", default="graphrag_results.csv", help="Path to save the results CSV")
    args = parser.parse_args()

    main(args.input_csv, args.output_csv)
