import pandas as pd
import requests


class ProductModule:
    def __init__(self, product_path="data/products.csv"):
        self.product_path = product_path

        # Load product dataset
        self.df = pd.read_csv(self.product_path)

    def find_product(self, query):
        """Find a product based on its name or product ID."""

        query_lower = query.lower()

        # Search by product name
        matches = self.df[
            self.df["ProductName"]
            .str.lower()
            .apply(lambda name: name in query_lower or query_lower in name)
        ]

        # Search by ProductID if no name match
        if matches.empty:
            for _, row in self.df.iterrows():
                if str(row["ProductID"]) in query_lower:
                    matches = pd.DataFrame([row])
                    break

        return matches

    def answer(self, query):
        """Answer a product-specific question using Ollama."""

        matches = self.find_product(query)

        if matches.empty:
            return {
                "answer": "I could not find the requested product.",
                "product": None
            }

        product = matches.iloc[0]

        context = f"""
Product ID: {product['ProductID']}
Product Name: {product['ProductName']}
Category: {product['Category']}
Brand: {product['Brand']}
Price: ₹{product['Price']}
Description: {product['Description']}
Rating: {product['Rating']}
"""

        prompt = f"""
You are an e-commerce product assistant.

Answer the user's question using ONLY the product information below.

Product information:
{context}

User question:
{query}

Give a short and clear answer.
Do not invent product information.
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3.5:4b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )

            response.raise_for_status()

            answer = response.json()["response"].strip()

        except requests.exceptions.RequestException:
            answer = (
                f"{product['ProductName']} is a "
                f"{product['Category']} from {product['Brand']}. "
                f"It costs ₹{product['Price']} and has a "
                f"rating of {product['Rating']}."
            )

        return {
            "answer": answer,
            "product": product.to_dict()
        }


if __name__ == "__main__":
    product_module = ProductModule()

    query = input("\nAsk a product question: ")

    result = product_module.answer(query)

    print("\n--- PRODUCT RESULT ---")
    print("User:", query)

    if result["product"]:
        print("Product:", result["product"]["ProductName"])
        print("Answer:", result["answer"])
    else:
        print("Answer:", result["answer"])