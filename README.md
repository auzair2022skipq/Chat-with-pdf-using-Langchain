# PDF Reader using LLM and Langchain

This is a simple Python script that uses the Langchain framework and the OpenAI Language Model (LLM) to read and extract text from a PDF file. It allows users to upload a PDF file, ask questions about the content of the PDF, and obtain answers using the Langchain framework's question-answering capabilities.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Python 3.x
- `dotenv` package
- `streamlit` package
- `PyPDF2` package
- Langchain framework (install using `pip install langchain`)

## Setup

1. Clone the repository or create a new Python file.
2. Install the required dependencies using `pip`:
```shell
pip install dotenv streamlit PyPDF2 langchain
```
3. Import the necessary modules:
```python
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
```
4. Load the environment variables (if any) using `load_dotenv()`.

## Usage

1. Set the Streamlit page configuration and display the header:
```python
st.set_page_config(page_title='Ask anything about your pdf!')
st.header("Ask your pdf!")
```
2. Allow the user to upload a PDF file:
```python
pdf = st.file_uploader("Upload your pdf!", type="pdf")
```
3. If a PDF is uploaded, read and extract the text from it:
```python
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
```
4. Split the text into chunks using the `CharacterTextSplitter` from Langchain:
```python
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
```
5. Create embeddings and build a knowledgebase using the Langchain framework:
```python
    embeddings = OpenAIEmbeddings()
    knowledgebase = FAISS.from_texts(chunks, embeddings)
```
6. Ask the user to input a question:
```python
    userquestion = st.text_input("Ask a question!")
```
7. If a question is provided, perform a similarity search on the knowledgebase and run the question-answering chain:
```python
    if userquestion:
        docs = knowledgebase.similarity_search(userquestion)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=userquestion)
            print(cb)
        st.write(response)
```
8. Run the Streamlit application:
```python
st.run()
```

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.

## Contributing

Contributions are welcome! If you find any issues or want to enhance the functionality of this code, please submit a pull request or open an issue.

## Disclaimer

This code uses the Langchain framework and the OpenAI Language Model. Make sure to adhere to the terms and
