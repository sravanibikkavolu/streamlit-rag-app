import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

st.set_page_config(page_title="PDF RAG Chatbot")

st.title("📚 PDF RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    # Read PDF
    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    # Create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    # Create ChromaDB
    client = chromadb.Client()

    try:
        client.delete_collection("pdf_collection")
    except:
        pass

    collection = client.get_or_create_collection(
        name="pdf_collection"
    )

    # Store embeddings
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i].tolist()]
        )

    st.success(f"Created {len(chunks)} chunks")
    st.success(f"Created {len(embeddings)} embeddings")
    st.success("Stored embeddings in ChromaDB")

    with st.expander("View First 3 Chunks"):
        for i, chunk in enumerate(chunks[:3]):
            st.subheader(f"Chunk {i+1}")
            st.write(chunk)

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        question_embedding = model.encode([question])

        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=3
        )

        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.
If the answer is not available in the context, say:
"I could not find that information in the PDF."

Context:
{context}

Question:
{question}
"""

        with st.spinner("Generating answer..."):

            groq_client = Groq(
                api_key=st.secrets["GROQ_API_KEY"]
            )

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Chunks"):
            for chunk in retrieved_chunks:
                st.write(chunk)