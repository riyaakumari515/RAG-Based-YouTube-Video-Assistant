import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_video_id(url):
    """Extracts video ID from URL."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript_chunks(video_id):
    """
    Fetches transcript and splits it into chunks.
    Returns a list of Document objects.
    """
    try:
        # Different versions of youtube_transcript_api expose different APIs
        # Try the most common names in order and fall back gracefully.
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            # Newer versions expose instance methods like `fetch` and `list`.
            api = YouTubeTranscriptApi()
            if hasattr(api, "fetch"):
                transcript = api.fetch(video_id)
            elif hasattr(api, "list"):
                listing = api.list(video_id)
                # try to extract the actual transcript (first entry) if possible
                if isinstance(listing, list) and listing:
                    transcript = listing[0]
                else:
                    transcript = listing
            else:
                raise RuntimeError("No compatible transcript API found")
        
        # Normalize transcript into list of items with 'start' and 'text'
        items = None
        if transcript is None:
            return None
        # If transcript is a dict, try to extract the list inside common keys
        if isinstance(transcript, dict):
            for k in ("transcript", "transcripts", "data", "sections", "items", "parts"):
                if k in transcript and isinstance(transcript[k], list):
                    items = transcript[k]
                    break
            # fallback: if dict looks like a single item
            if items is None and all(k in transcript for k in ("text", "start")):
                items = [transcript]
        elif isinstance(transcript, list):
            items = transcript
        elif hasattr(transcript, '__iter__') and not isinstance(transcript, (str, bytes)):
            # FetchedTranscript and similar are iterable
            try:
                items = list(transcript)
            except Exception:
                items = None
        else:
            # unknown format
            items = None

        if not items:
            return None

        full_text = ""
        for item in items:
            # tolerate different key names and structures
            start = 0
            text = None
            if isinstance(item, dict):
                try:
                    start = int(item.get("start", 0) or 0)
                except Exception:
                    start = 0
                text = item.get("text") or item.get("caption") or item.get("content")
            elif hasattr(item, "start") or hasattr(item, "text"):
                try:
                    start = int(getattr(item, "start", 0) or 0)
                except Exception:
                    # some transcripts use floats
                    try:
                        start = int(float(getattr(item, "start", 0) or 0))
                    except Exception:
                        start = 0
                text = getattr(item, "text", None) or getattr(item, "content", None)
            else:
                text = str(item)

            if not text:
                continue

            # ensure text is safe string (avoid print-time encoding issues later)
            full_text += f" [START:{start}] {text}"

        # intelligent splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n", " ", ""]
        )
        
        # We process the raw transcript list to keep better timestamp alignment 
        # But for a simple RAG, splitting the big text block is often more robust for context.
        # A more advanced method would attach metadata to every chunk specifically.
        
        # simplified approach for this demo:
        chunks = text_splitter.create_documents([full_text])
        
        # Attempt to extract the first timestamp from the chunk text for metadata
        for chunk in chunks:
            # find first occurrence of [START:123]
            match = re.search(r"\[START:(\d+)\]", chunk.page_content)
            if match:
                timestamp = match.group(1)
                chunk.metadata["source"] = f"https://youtu.be/{video_id}?t={timestamp}"
                chunk.metadata["timestamp"] = timestamp
            else:
                chunk.metadata["source"] = f"https://youtu.be/{video_id}"

        return chunks

    except Exception as e:
        print(f"Error: {e}")
        return None