import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Automatically parse explicit local credentials
load_dotenv()

# ==========================================
# 1. LOAD THE PDF
# ==========================================
pdf_path = "data/mg_service_manual.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Missing core reference documentation file at '{pdf_path}'. Please add it to resume.")

loader = PyPDFLoader(pdf_path)
car_docs = loader.load()
print(f"Pages successfully loaded from manual: {len(car_docs)}")

# ==========================================
# 2. LOAD THE LANGUAGE MODEL
# ==========================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==========================================
# 3. LOAD THE EMBEDDING MODEL
# ==========================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# ==========================================
# 4. SPLIT THE PDF INTO CHUNKS
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(car_docs)
print(f"Document text split down into: {len(splits)} isolated chunks.")

# ==========================================
# 5. CREATE CHROMA VECTOR DATABASE
# ==========================================
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

# ==========================================
# 6. CREATE THE RETRIEVER
# ==========================================
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ==========================================
# 7. CREATE THE RAG PROMPT
# ==========================================
prompt = ChatPromptTemplate.from_template("""
You are a helpful car assistant.

Use the following context from the car manual to answer
the user's question.

If you don't know the answer based on the context,
say that you don't know.

Keep your answer concise and use a maximum of
three sentences.

Context:
{context}

Question:
{question}

Answer:
""")

# ==========================================
# 8. CREATE THE RAG CHAIN
# ==========================================
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# ==========================================
# 9. CONSTRUCT TARGET DRIVER QUERY
# ==========================================
query = "The Gasoline Particulate Filter Full warning has appeared. What does this mean and what should I do?"

# ==========================================
# 10. GENERATE AND PRINT THE ASSIGNMENT VARIABLE
# ==========================================
print("\nInvoking RAG execution chain pipeline queries...")
answer = rag_chain.invoke(query).content

print("\n--- Resulting Variable 'answer' ---")
print(answer)
