from chatbot import EcommerceChatbot


def main():

    chatbot = EcommerceChatbot()

    print("\n" + "=" * 60)
    print("        LOCAL E-COMMERCE RAG CHATBOT")
    print("=" * 60)

    print("\nThe chatbot can handle:")
    print("1. FAQs")
    print("2. Product Questions")
    print("3. Product Recommendations")
    print("4. Review Summaries")
    print("5. Intent Detection")

    print("\nType 'exit' to stop.")

    while True:

        query = input("\nYou: ").strip()

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "bye"
        }:
            print("\nBot: Goodbye!")
            break

        result = chatbot.process_query(query)

        print("\nBot:")
        print(result["answer"])


if __name__ == "__main__":
    main()