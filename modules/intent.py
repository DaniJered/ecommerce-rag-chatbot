import requests
import json


class IntentModule:
    def __init__(self, model="qwen3.5:4b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def detect_intent(self, query):
        """
        Detect the user's intent using Ollama.
        """

        prompt = f"""
You are an intent classifier for an e-commerce chatbot.

Classify the user's query into exactly ONE of these intents:

FAQ
PRODUCT
RECOMMENDATION
REVIEW
UNKNOWN

Definitions:

FAQ:
Questions about orders, delivery, tracking, cancellation,
returns, payment, shipping, or general store information.

PRODUCT:
Questions about a specific product, such as price, brand,
category, description, or rating.

RECOMMENDATION:
Requests asking the system to suggest, recommend, or find
products based on preferences or requirements.

REVIEW:
Questions asking about customer reviews, opinions, feedback,
sentiment, or review summaries for a product.

UNKNOWN:
Anything that does not fit the above categories.

User query:
{query}

Return ONLY valid JSON in this format:

{{
    "intent": "FAQ"
}}

Do not add explanations.
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

            text = response.json()["response"].strip()

            # Remove markdown code fences if the model adds them
            text = text.replace("```json", "").replace("```", "").strip()

            result = json.loads(text)

            intent = result.get("intent", "UNKNOWN").upper()

            valid_intents = {
                "FAQ",
                "PRODUCT",
                "RECOMMENDATION",
                "REVIEW",
                "UNKNOWN"
            }

            if intent not in valid_intents:
                intent = "UNKNOWN"

            return intent

        except Exception:
            # Simple fallback classification
            query_lower = query.lower()

            if any(
                word in query_lower
                for word in [
                    "track",
                    "return",
                    "cancel",
                    "delivery",
                    "parcel",
                    "order",
                    "shipping"
                ]
            ):
                return "FAQ"

            if any(
                word in query_lower
                for word in [
                    "recommend",
                    "suggest",
                    "looking for",
                    "best",
                    "want a",
                    "need a"
                ]
            ):
                return "RECOMMENDATION"

            if any(
                word in query_lower
                for word in [
                    "review",
                    "reviews",
                    "feedback",
                    "opinion",
                    "sentiment"
                ]
            ):
                return "REVIEW"

            return "PRODUCT"


if __name__ == "__main__":

    intent_module = IntentModule()

    query = input("\nEnter an e-commerce query: ")

    intent = intent_module.detect_intent(query)

    print("\n--- INTENT DETECTION ---")
    print("User:", query)
    print("Detected Intent:", intent)