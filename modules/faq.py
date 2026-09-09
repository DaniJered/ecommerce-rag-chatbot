import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util


class FAQModule:
    def __init__(self, faq_path="data/faqs.csv"):
        self.faq_path = faq_path

        # Load FAQ dataset
        self.df = pd.read_csv(self.faq_path)

        # Load embedding model
        print("Loading FAQ embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create embeddings for FAQ questions
        self.faq_embeddings = self.model.encode(
            self.df["Question"].tolist(),
            convert_to_tensor=True
        )

    def search(self, query, top_k=3):
        """Find the most relevant FAQs for the user query."""

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        scores = util.cos_sim(
            query_embedding,
            self.faq_embeddings
        )[0]

        top_results = scores.topk(
            k=min(top_k, len(self.df))
        )

        results = []

        for score, index in zip(
            top_results.values,
            top_results.indices
        ):
            row = self.df.iloc[int(index)]

            results.append({
                "FAQID": int(row["FAQID"]),
                "Question": row["Question"],
                "Answer": row["Answer"],
                "Score": float(score)
            })

        return results

    def answer(self, query):
        """Retrieve the best FAQ and generate an answer using Ollama."""

        results = self.search(query, top_k=3)

        best = results[0]

        # Only use the retrieved FAQ as context
        context = "\n".join(
            [
                f"Question: {item['Question']}\n"
                f"Answer: {item['Answer']}"
                for item in results
            ]
        )

        prompt = f"""
You are an e-commerce customer support assistant.

Answer the user's question using ONLY the FAQ information provided below.

FAQ information:
{context}

User question:
{query}

Give a short, clear and helpful answer.
Do not invent information that is not present in the FAQ.
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3.5:4b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0
                    }
                },
                timeout=30
            )

            response.raise_for_status()

            answer = response.json()["response"]

        except requests.exceptions.RequestException as e:
            # Fallback if Ollama is unavailable
            answer = best["Answer"]

        return {
            "answer": answer.strip(),
            "matched_faq": best,
            "results": results
        }


if __name__ == "__main__":
    faq = FAQModule()

    question = input("\nAsk an FAQ question: ")

    result = faq.answer(question)

    print("\n--- FAQ RESULT ---")
    print("User:", question)
    print("Matched FAQ:", result["matched_faq"]["Question"])
    print("Similarity:", round(result["matched_faq"]["Score"], 3))
    print("Answer:", result["answer"])