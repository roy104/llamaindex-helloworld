import os
from dotenv import load_dotenv
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex


def main(url: str) -> None:
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is LlamaIndex?")
    print(response)


if __name__ == '__main__':
    load_dotenv()
    print("Hello World")
    print(f"OPENAI_API_KEY: {os.environ['OPENAI_API_KEY']}")
    print("****")
    main(url="https://docs.llamaindex.ai/en/stable/community/integrations/vector_stores/")
