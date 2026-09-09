from modules.faq import FAQModule
from modules.product import ProductModule
from modules.recommendation import RecommendationModule
from modules.review import ReviewModule
from modules.intent import IntentModule


class EcommerceChatbot:

    def __init__(self):

        print("\nInitializing E-Commerce Chatbot...\n")

        self.intent_module = IntentModule()

        self.faq_module = FAQModule()

        self.product_module = ProductModule()

        self.recommendation_module = RecommendationModule()

        self.review_module = ReviewModule()

        print("\nChatbot initialization complete.\n")

    def process_query(self, query):

        # Step 1: Detect intent
        intent = self.intent_module.detect_intent(query)

        print(f"\n[Intent Detected: {intent}]")

        # Step 2: Route query to appropriate module

        if intent == "FAQ":

            result = self.faq_module.answer(query)

            return {
                "intent": intent,
                "answer": result["answer"]
            }

        elif intent == "PRODUCT":

            result = self.product_module.answer(query)

            return {
                "intent": intent,
                "answer": result["answer"]
            }

        elif intent == "RECOMMENDATION":

            results = self.recommendation_module.recommend(
                query,
                top_k=5
            )

            answer = "Here are some products you may like:\n\n"

            for i, product in enumerate(
                results,
                start=1
            ):

                answer += (
                    f"{i}. {product['ProductName']}\n"
                    f"   Brand: {product['Brand']}\n"
                    f"   Category: {product['Category']}\n"
                    f"   Price: ₹{product['Price']}\n"
                    f"   Rating: {product['Rating']}\n"
                    f"   Similarity: "
                    f"{product['Similarity']:.3f}\n\n"
                )

            return {
                "intent": intent,
                "answer": answer
            }

        elif intent == "REVIEW":

            result = self.review_module.summarize_reviews(
                query
            )

            return {
                "intent": intent,
                "answer": result["answer"]
            }

        else:

            return {
                "intent": intent,
                "answer": (
                    "Sorry, I could not understand "
                    "your request. Please ask an "
                    "e-commerce related question."
                )
            }


if __name__ == "__main__":

    chatbot = EcommerceChatbot()

    print("=" * 60)
    print("        E-COMMERCE AI CHATBOT")
    print("=" * 60)

    while True:

        query = input("\nYou: ")

        if query.lower() in {
            "exit",
            "quit",
            "bye"
        }:
            print("\nChatbot: Goodbye!")
            break

        result = chatbot.process_query(query)

        print("\nChatbot:")
        print(result["answer"])