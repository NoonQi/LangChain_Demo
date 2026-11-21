import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def main():
    # 1. Load Environment Variables (API Key)
    print("Loading environment variables...")
    load_dotenv()
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY not found in .env file.")
        return

    # 2. Load the PDF Document
    pdf_path = "data/sample.pdf"
    print(f"Loading document from {pdf_path}...")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} pages.")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        print("Please ensure 'data/sample.pdf' exists.")
        return

    # 3. Split Text into Chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 4. Generate Embeddings and Store in Vector Database (FAISS)
    print("Generating embeddings and building FAISS Vector Database...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings_model)

    # 5. Create the Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 6. Initialize the Gemini LLM
    print("Initializing Google Gemini model...")
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.3)

    # 7. Create the Prompt Template
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved "
        "context to answer the user's question. If you don't know the answer, "
        "just say that you don't know. Do not make up information."
        "\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 8. Build the RAG Chain
    print("Building RAG Chain...")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 9. Ask Questions in a Loop
    print("\n--- System Ready! ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_question = input("Ask a question about the document: ")
        
        if user_question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_question.strip():
            continue
            
        print("Thinking...")
        
        try:
            # Invoke the chain
            response = rag_chain.invoke({"input": user_question})
            
            # Print the Answer
            print("\n--- Answer ---")
            print(response["answer"])
            print("--------------\n")
            
        except Exception as e:
            print(f"\nAn error occurred while generating the answer: {e}\n")

if __name__ == "__main__":
    main()