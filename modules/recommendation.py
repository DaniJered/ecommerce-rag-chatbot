import pandas as pd
from sentence_transformers import SentenceTransformer, util


class RecommendationModule:

    def __init__(self, product_path="data/products.csv"):

        self.product_path = product_path

        # Load product dataset
        self.df = pd.read_csv(self.product_path)

        print("Loading recommendation model...")

        # Load sentence transformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Combine useful product information
        self.product_text = (
            self.df["ProductName"].fillna("")
            + " "
            + self.df["Category"].fillna("")
            + " "
            + self.df["Brand"].fillna("")
            + " "
            + self.df["Description"].fillna("")
        ).tolist()

        # Generate embeddings for products
        self.product_embeddings = self.model.encode(
            self.product_text,
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def recommend(self, query, top_k=5):

        # Convert user query into embedding
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        # Calculate similarity
        scores = util.cos_sim(
            query_embedding,
            self.product_embeddings
        )[0]

        # Get top products
        top_results = scores.topk(
            k=min(top_k, len(self.df))
        )

        recommendations = []

        for score, index in zip(
            top_results.values,
            top_results.indices
        ):

            product = self.df.iloc[int(index)]

            recommendations.append({
                "ProductID": int(product["ProductID"]),
                "ProductName": product["ProductName"],
                "Category": product["Category"],
                "Brand": product["Brand"],
                "Price": product["Price"],
                "Rating": product["Rating"],
                "Description": product["Description"],
                "Similarity": float(score)
            })

        return recommendations


if __name__ == "__main__":

    recommender = RecommendationModule()

    query = input(
        "\nWhat type of product are you looking for? "
    )

    results = recommender.recommend(query)

    print("\n--- PRODUCT RECOMMENDATIONS ---")
    print("User:", query)

    for i, product in enumerate(results, start=1):

        print(f"\n{i}. {product['ProductName']}")
        print(f"   Category: {product['Category']}")
        print(f"   Brand: {product['Brand']}")
        print(f"   Price: ₹{product['Price']}")
        print(f"   Rating: {product['Rating']}")
        print(
            f"   Similarity: "
            f"{product['Similarity']:.3f}"
        )