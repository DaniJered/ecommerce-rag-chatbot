# E-Commerce RAG Chatbot

An AI-powered local e-commerce chatbot built using Python, Ollama,
Sentence Transformers, and a Retrieval-Augmented Generation (RAG)
approach.

## Project Overview

This project implements a modular e-commerce chatbot capable of:

- Answering frequently asked questions
- Answering product-specific questions
- Recommending products using semantic similarity
- Summarizing customer reviews
- Detecting user intent using an LLM
- Routing queries to the appropriate module

The chatbot runs locally using Ollama.

## Architecture

User Query
    |
    v
Intent Detection
    |
    v
Query Routing
    |
    +----> FAQ Module
    |
    +----> Product Module
    |
    +----> Recommendation Module
    |
    +----> Review Module
    |
    v
Ollama LLM
    |
    v
Final Response

## Technologies Used

- Python 3
- Ollama
- Qwen 3.5 4B
- Sentence Transformers
- all-MiniLM-L6-v2
- Pandas
- Requests
- Scikit-learn

## Project Structure

ecommerce-chatbot/
│
├── README.md
├── requirements.txt
├── main.py
├── chatbot.py
│
├── data/
│   ├── products.csv
│   ├── faqs.csv
│   └── reviews.csv
│
├── modules/
│   ├── faq.py
│   ├── intent.py
│   ├── product.py
│   ├── recommendation.py
│   └── review.py
│
└── screenshots/
    ├── faq.png
    ├── product.png
    ├── recommendation.png
    ├── review.png
    └── intent.png

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt

Install Ollama and make sure the required model is available:

ollama pull qwen3.5:4b
Running the Project

Navigate to the project directory:

cd ecommerce-chatbot

Run:

python main.py
Example Queries
FAQ
How can I track my order?

Example response:

Use the Track Order page with your order ID.
Product Question
What is the price of Lenovo Laptop 1?

Example response:

The price of Lenovo Laptop 1 is ₹30316.
Recommendation
I need a good laptop for everyday use

The system retrieves semantically similar products from the
product dataset.

Review Summary
What do customers think about Lenovo Laptop 1?

Example result:

Lenovo Laptop 1 has an average customer rating of 4.10/5
based on 10 reviews.
Dataset

The project uses three CSV datasets:

products.csv — product information
faqs.csv — frequently asked questions and answers
reviews.csv — customer reviews and ratings
Modules
FAQ Module

Uses Sentence Transformers to retrieve semantically similar FAQ
questions and Ollama to generate the final response.

Product Module

Retrieves information about a specific product and uses Ollama
to answer product-related questions.

Recommendation Module

Uses Sentence Transformers and cosine similarity to retrieve
products relevant to the user's requirements.

Review Module

Retrieves customer reviews for a product and generates a review
summary.

Intent Module

Uses Ollama to classify queries into:

FAQ
PRODUCT
RECOMMENDATION
REVIEW
UNKNOWN
Chatbot Module

The main chatbot coordinates intent detection, query routing,
retrieval, and response generation.

Result

The implemented chatbot successfully handles:

FAQ queries
Product questions
Product recommendations
Customer review analysis
Intent detection

The system operates locally using Ollama and the provided
e-commerce datasets.
