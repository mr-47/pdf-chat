import hashlib

try:
    import streamlit as st
except ModuleNotFoundError:
    raise SystemExit(
        "streamlit is not installed. The web UI needs the optional extra:\n"
        "  pip install -r requirements-ui.txt\n"
        "or use ./run.sh -m streamlit run app.py\n"
        "The terminal interface does not need it: ./run.sh cli.py <file.pdf>"
    ) from None

from rag import build_index_from_bytes, index_pages, retrieve_matches

TOP_K = 4


def file_key(name: str, data: bytes) -> str:
    return f"{name}:{hashlib.sha256(data).hexdigest()}"


@st.cache_resource(show_spinner="Indexing PDF...")
def index_upload(key: str, data: bytes):
    vectorstore, chunks = build_index_from_bytes(data)
    return vectorstore, len(chunks), index_pages(chunks)


def collect_matches(indexes, question: str, k: int):
    scored = []
    for name, vectorstore in indexes:
        for doc, distance in retrieve_matches(vectorstore, question, k=k):
            scored.append(
                {
                    "name": name,
                    "page": doc.metadata.get("page", "?"),
                    "distance": distance,
                    "text": doc.page_content.strip(),
                }
            )
    scored.sort(key=lambda row: row["distance"])
    return scored[:k]


def render_matches(matches) -> None:
    for rank, match in enumerate(matches, start=1):
        st.markdown(
            f"**{match['name']} - page {match['page']}** "
            f"(distance {match['distance']:.3f})"
        )
        st.markdown(match["text"])
        if rank < len(matches):
            st.divider()


st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("Chat with your PDF")
st.caption(
    "Retrieves matching passages from your PDFs. "
    "No LLM is called, so you get source text rather than a written answer."
)

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )
    top_k = st.slider("Passages per question", 1, 10, TOP_K)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

indexes = []
for upload in uploads or []:
    data = upload.getvalue()
    try:
        vectorstore, chunk_count, pages = index_upload(
            file_key(upload.name, data), data
        )
    except Exception as exc:
        st.error(f"Could not index {upload.name}: {exc}")
        continue

    indexes.append((upload.name, vectorstore))
    with st.sidebar:
        st.success(
            f"{upload.name}: {chunk_count} chunks, "
            f"{len(pages)} pages ({pages[0]}-{pages[-1]})"
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_matches(message["matches"])

if not indexes:
    st.info("Upload one or more PDFs in the sidebar to begin.")
else:
    if not st.session_state.messages:
        st.info(
            "Ask a question about the documents. Try something like "
            "\"what is the main conclusion?\""
        )

    question = st.chat_input("Ask a question about your PDFs")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                matches = collect_matches(indexes, question, top_k)
            if matches:
                render_matches(matches)
            else:
                st.warning("Nothing found. Try different wording.")

        st.session_state.messages.append(
            {"role": "assistant", "matches": matches}
        )
