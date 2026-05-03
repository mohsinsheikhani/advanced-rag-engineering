import streamlit as st
from dotenv import load_dotenv
import sys
sys.path.insert(0, '.')

load_dotenv('.env')

from app.document_store import get_document_store
from retrieval.rag_pipeline import create_rag_pipeline

st.set_page_config(page_title="MIA RAG Assistant", page_icon="🤖")
st.title("MIA Knowledge Base Assistant")

# Initialize pipeline
def init_pipeline():
    ds = get_document_store()
    return create_rag_pipeline(ds)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = init_pipeline()

pipeline = st.session_state.pipeline

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about MIA..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.run({
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query}
            }, include_outputs_from={"retriever", "llm"})
            
            answer = result["llm"]["replies"][0]
            st.markdown(answer)
            
            # Show retrieved chunks in expander
            if "retriever" in result:
                with st.expander("📚 Retrieved Context"):
                    for i, doc in enumerate(result["retriever"]["documents"], 1):
                        st.markdown(f"**[{i}]** `{doc.meta.get('file_path', 'unknown')}` (score: {doc.score:.3f})")
                        st.markdown(doc.content[:200] + "...")
                        st.divider()
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
