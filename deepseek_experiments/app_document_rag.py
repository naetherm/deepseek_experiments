import os
import tempfile

import streamlit as st
# There are many more PDF loaders available in LangChain
# Take a look at https://python.langchain.com/api_reference/community/document_loaders.html
# And don't forget to install the required dependencies for each loader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
    

PROMPT = """
1. Only use the following context below to answer the question.
2. If unsure, say "I don't know", never make up answers to questions.
3. Keep answers short, precise, and under up to 4 sentences.

Context: {context}

Question: {question}

Answer:
"""

PROMPT_TEMPLATE = PromptTemplate.from_template(PROMPT)


def main():
  st.title("Document QA App")
  temp_dir = tempfile.TemporaryDirectory()

  uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

  if uploaded_file:
    # Save PDF
    with open(os.path.join(temp_dir.name, "temp.pdf"), "wb") as f_out:
      f_out.write(uploaded_file.getvalue())
    
    # Load PDF
    loader = PDFPlumberLoader(os.path.join(temp_dir.name, "temp.pdf"))
    docs = loader.load()


    # Split the provided document into chunks
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)


    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Connect retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Define the LLM we want to use
    llm = Ollama(model="deepseek-r1:1.5b")
    
    llm_chain = LLMChain(
      llm=llm,
      prompt=PROMPT_TEMPLATE,
      callbacks=None,
      verbose=True,
    )
    
    doc_prompt = PromptTemplate(
      template="Context:\ncontent:{page_content}\nsource:{source}",
      input_variables=["page_content", "source"],
    )
    
    qa = RetrievalQA(
      combine_documents_chain=StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=doc_prompt,
        callbacks=None,
      ),
      retriever=retriever,
    )
    
    user_input = st.text_input("What do you want to know:")
    
    if user_input:
      with st.spinner("Thinking..."):
        response = qa(user_input)["result"]
        st.write(response)
  else:
    st.write("Before we continue, please upload your PDF file.")


if __name__ == "__main__":
  main()
