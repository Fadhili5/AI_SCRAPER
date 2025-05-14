# AI_SCRAPER
**AI Crawler** is a Streamlit-based application that allows you to crawl and analyze web pages for question-answering tasks. By entering a URL, the tool retrieves and processes the content, then answers user questions based on the retrieved context using language models and vector stores.

## Some of its features include:
- Load content from a webpage using Selenium.
- Split loaded content into smaller chunks for efficient processing.
- Create a FAISS vector store from the processed content.
- Retrieve similar document chunks based on a user query.
- Answer user questions using the loaded context and a language model (LLM).

## Set Up the Environment:
How to install the required libraries
```
pip install -r requirements.txt
```

## To run the application:
```
streamlit run ai_crawler.py
```
