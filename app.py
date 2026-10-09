import streamlit as st

from src.ingest import extract_video_id, get_transcript_chunks
from src.vector_store import create_vector_db
from src.rag import get_rag_chain


# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="TubeTalk (Groq Edition)",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RAG-AGENT: Chat with YouTube")


# ---------------- Sidebar ----------------

with st.sidebar:
    st.header("🎥 YouTube Video")

    video_url = st.text_input(
        "Enter YouTube Video URL"
    )

    if st.button("Process Video"):

        if not video_url:
            st.warning("Please enter a YouTube URL.")

        else:
            with st.spinner("Downloading transcript and creating embeddings..."):

                try:
                    # Extract video ID
                    vid_id = extract_video_id(video_url)

                    if not vid_id:
                        st.error("Invalid YouTube URL.")
                    else:
                        # Get transcript
                        chunks = get_transcript_chunks(vid_id)

                        if chunks:

                            # Create FAISS vector database
                            vector_store = create_vector_db(chunks)

                            # Store vector database
                            st.session_state.vector_store = vector_store

                            # Create RAG chain
                            st.session_state.chain = get_rag_chain(
                                vector_store
                            )

                            st.success(
                                "✅ Video processed successfully! "
                                "You can now ask questions."
                            )

                        else:
                            st.error(
                                "❌ Could not retrieve transcript."
                            )

                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")


# ---------------- Main Chat Interface ----------------

if "chain" in st.session_state:

    user_query = st.chat_input(
        "Ask something about the video..."
    )

    if user_query:

        # User message
        with st.chat_message("user"):
            st.write(user_query)

        # AI response
        with st.chat_message("assistant"):

            try:

                response = st.session_state.chain.invoke(
                    {"query": user_query}
                )

                # Display answer
                st.write(response["result"])

                # Display sources
                if "source_documents" in response:

                    with st.expander("📌 View Source Timestamps"):

                        for doc in response["source_documents"]:

                            source_link = doc.metadata.get(
                                "source",
                                "#"
                            )

                            st.markdown(
                                f"- [Jump to segment]({source_link})"
                            )

                            st.caption(
                                doc.page_content[:150] + "..."
                            )

            except Exception as e:

                err_text = str(e)

                if (
                    "decommissioned" in err_text.lower()
                    or "model" in err_text.lower()
                ):

                    st.error(
                        "⚠️ Groq model error. "
                        "Please check your GROQ_MODEL "
                        "in the .env file."
                    )

                else:

                    st.error(
                        f"❌ Error generating response: {err_text}"
                    )