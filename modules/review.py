import pandas as pd
import requests


class ReviewModule:

    def __init__(
        self,
        review_path="data/reviews.csv",
        model="qwen3.5:4b"
    ):
        self.review_path = review_path
        self.model = model
        self.url = "http://localhost:11434/api/generate"

        # Load review dataset
        self.df = pd.read_csv(self.review_path)

    def find_reviews(self, product_name):
        """
        Find reviews for a specific product.
        """

        product_name = product_name.lower().strip()

        matches = self.df[
            self.df["ProductName"]
            .str.lower()
            .str.contains(product_name, na=False)
        ]

        return matches

    def find_product_from_query(self, query):
        """
        Find the product mentioned in the user's query.
        """

        query_lower = query.lower()

        # Check complete product names first
        for product_name in self.df["ProductName"].unique():

            if product_name.lower() in query_lower:
                return product_name

        # Check partial product name
        words = query_lower.split()

        for product_name in self.df["ProductName"].unique():

            product_words = product_name.lower().split()

            if any(
                word in words
                for word in product_words
                if len(word) > 3
            ):
                return product_name

        return None

    def summarize_reviews(self, query):

        product_name = self.find_product_from_query(query)

        if product_name is None:
            return {
                "answer": "I could not identify the product for the review analysis.",
                "product": None
            }

        reviews = self.find_reviews(product_name)

        if reviews.empty:
            return {
                "answer": "No reviews were found for this product.",
                "product": product_name
            }

        # Limit the number of reviews sent to the LLM
        review_text = "\n".join(
            [
                f"- Rating: {row['Rating']}/5 | Review: {row['Review']}"
                for _, row in reviews.head(30).iterrows()
            ]
        )

        average_rating = reviews["Rating"].mean()

        prompt = f"""
You are an e-commerce review analysis assistant.

Analyze the customer reviews below.

Product:
{product_name}

Average rating:
{average_rating:.2f}/5

Customer reviews:
{review_text}

Provide a concise review summary containing:

1. Overall customer opinion
2. Positive aspects
3. Negative aspects
4. Overall rating

Use ONLY the information provided.
Do not invent information.
"""

        try:

            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )

            response.raise_for_status()

            answer = response.json()["response"].strip()

        except requests.exceptions.RequestException:

            answer = (
                f"{product_name} has an average customer rating "
                f"of {average_rating:.2f}/5 based on "
                f"{len(reviews)} reviews."
            )

        return {
            "answer": answer,
            "product": product_name,
            "average_rating": average_rating,
            "review_count": len(reviews)
        }


if __name__ == "__main__":

    review_module = ReviewModule()

    query = input(
        "\nAsk about product reviews: "
    )

    result = review_module.summarize_reviews(query)

    print("\n--- REVIEW SUMMARY ---")
    print("User:", query)
    print("Product:", result["product"])
    print("Answer:", result["answer"])

    if "average_rating" in result:
        print(
            "Average Rating:",
            round(result["average_rating"], 2)
        )

        print(
            "Number of Reviews:",
            result["review_count"]
        )