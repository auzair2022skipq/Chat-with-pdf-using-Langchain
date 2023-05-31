from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
load_dotenv()
st.set_page_config(page_title='Ask anything about your pdf!')
st.header("Ask your pdf!")
pdf=st.file_uploader("Upload your pdf!",type="pdf")

if pdf is not None:
    pdf_reader= PdfReader(pdf)
    text=""
    for page in pdf_reader.pages:
        text+= page.extract_text()
    
    #split in chunks
    text_splitter= CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks= text_splitter.split_text(text)
    #st.write(chunks)

    embeddings= OpenAIEmbeddings()
    knowledgebase= FAISS.from_texts(chunks,embeddings)

    userquestion= st.text_input("Ask a question!")

    if userquestion:
        docs= knowledgebase.similarity_search(userquestion)
        #st.write(docs)
        llm=OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response=chain.run(input_documents=docs, question=userquestion)
            print(cb)

        st.write(response)
