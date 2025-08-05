import os
import time
import streamlit as st
import faiss
import numpy as np
import networkx as nx
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def load_documents(path: str) -> List[str]:
    loader = TextLoader(path)
    docs = loader.load()
    return [doc.page_content for doc in docs]


def chunk_documents(texts: List[str], chunk_size=500, overlap=50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.create_documents(texts)
    return [chunk.page_content for chunk in chunks]


class FaissVectorIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build_index(self, chunks: List[str]):
        self.texts = chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, k=5) -> List[str]:
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, k)
        return [self.texts[i] for i in I[0]]


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def build_from_chunks(self, chunks: List[str]):
        for i, chunk in enumerate(chunks):
            entities = self.extract_entities(chunk)
            for e1 in entities:
                for e2 in entities:
                    if e1 != e2:
                        self.graph.add_edge(e1, e2, chunk=chunk)

    def extract_entities(self, text: str) -> List[str]:
        return list(set([w for w in text.split() if w.istitle()]))

    def traverse(self, query: str) -> List[str]:
        entities = self.extract_entities(query)
        results = set()
        for entity in entities:
            if entity in self.graph:
                neighbors = list(nx.single_source_shortest_path_length(self.graph, entity, cutoff=2).keys())
                for n in neighbors:
                    edge_data = self.graph.get_edge_data(entity, n)
                    if edge_data:
                        results.add(edge_data.get('chunk', ''))
        return list(results)


class HybridRetriever:
    def __init__(self, vector_index: FaissVectorIndex, graph: KnowledgeGraph):
        self.vector_index = vector_index
        self.graph = graph

    def retrieve(self, query: str, k=5) -> List[str]:
        vector_results = self.vector_index.search(query, k)
        graph_results = self.graph.traverse(query)
        combined = list(set(vector_results + graph_results))
        return combined[:k]


def generate_answer(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


def main():
    st.set_page_config(page_title="Hybrid RAG System")
    st.title("🔍 Hybrid RAG with Graph Knowledge Integration")
    query = st.text_input("Ask your question:")

    if 'retriever' not in st.session_state:
        with st.spinner("Loading documents and initializing retriever..."):
            docs = load_documents("sample.txt")
            chunks = chunk_documents(docs)

            v_index = FaissVectorIndex()
            v_index.build_index(chunks)

            k_graph = KnowledgeGraph()
            k_graph.build_from_chunks(chunks)

            retriever = HybridRetriever(v_index, k_graph)
            st.session_state.retriever = retriever

    if query and 'retriever' in st.session_state:
        start = time.time()
        results = st.session_state.retriever.retrieve(query)
        answer = generate_answer(query, results)
        end = time.time()

        st.markdown("## 🤖 Generated Answer")
        st.success(answer)

        st.markdown("### 🧠 Retrieved Chunks")
        for i, res in enumerate(results):
            st.markdown(f"**Chunk {i+1}:**\n{res}")

        st.markdown(f"⏱️ **Latency:** {end - start:.2f} seconds")


if __name__ == '__main__':
    main()


