import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import json
from langchain_community.document_loaders import YoutubeLoader        

def get_youtube_transcript(url: str):
    """
    Fetch the transcript for a YouTube video.

    Args:
        url: Full YouTube video URL.

    Returns:
        transcript (str)
    """
    # Always load transcript only; metadata will be fetched separately to avoid API errors.
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()
    transcript = docs[0].page_content
    return transcript

def search_youtube(query: str) -> str:
    """Search YouTube and return video URLs based on the query."""
    from youtube_search import YoutubeSearch
    results = YoutubeSearch(query, 10).to_json()
    data = json.loads(results)
    return str([
        "https://www.youtube.com" + video["url_suffix"]
        for video in data["videos"]
    ])
    
if __name__ == "__main__":
    video_url = "https://youtu.be/x5lhdef9kUM"  # replace with your video

    transcript_text = get_youtube_transcript(video_url)
    print(transcript_text)
