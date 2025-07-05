import os
import sys
import json
import uuid
from os.path import dirname as up, join, exists
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from tools.prompt_tools import (
    generate_knowledge_graph_prompt as get_knowledge_graph_prompt,
    generate_notes_prompt as get_notes_prompt,
    generate_sentiment_analysis_prompt as get_sentiment_analysis_prompt,
    generate_topic_modeling_prompt as get_topic_modeling_prompt
)
from tools.youtube_tools import (
    get_youtube_transcript as fetch_youtube_transcript,
    search_youtube as fetch_youtube_search
)

from fastmcp import FastMCP

# Ensure data directory exists
os.makedirs("/app/data/transcripts", exist_ok=True)
os.makedirs("/app/data/notes_md", exist_ok=True)
os.makedirs("/app/data/knowledge_graphs", exist_ok=True)
os.makedirs("/app/data/sentiment_analysis", exist_ok=True)
os.makedirs("/app/data/topic_modeling", exist_ok=True)

@dataclass
class Transcript:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source: str = ""  # URL or file path
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict = field(default_factory=dict)

@dataclass
class Note:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    transcript_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


my_mcp = FastMCP("Video Analysis Kit")

# In-memory storage (consider using a database in production)
_transcripts: Dict[str, Transcript] = {}
_notes: Dict[str, Note] = {}

@my_mcp.tool
def search_youtube(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search YouTube and return video information based on the query.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return (1-50)
    
    Returns:
        List[Dict[str, str]]: List of video information including id, title, and url
    """
    max_results = max(1, min(50, int(max_results)))  # Ensure between 1-50
    return fetch_youtube_search(query, max_results)

@my_mcp.tool
def save_transcript(content: str, source: str = "", metadata: Optional[Dict] = None) -> Dict:
    """
    Save a transcript with metadata.
    
    Args:
        content (str): Transcript content (required)
        source (str): Source URL or file path (default: "")
        metadata (Dict, optional): Additional metadata (default: None)
        
    Returns:
        Dict: {
            "id": str,  # Unique ID of the saved transcript
            "filepath": str,  # Path to the saved transcript file
            "status": str,  # "success" or "error"
            "message": str  # Additional status message
        }
        
    Raises:
        ValueError: If content is empty or invalid
        IOError: If there's an error writing the file
    """
    try:
        # Input validation
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
            
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
            
        # Create transcript object
        transcript = Transcript(
            content=str(content),
            source=str(source) if source else "",
            metadata=metadata
        )
        
        # Ensure the transcripts directory exists
        os.makedirs("/app/data/transcripts", exist_ok=True)
        
        # Save to file
        filename = f"transcript_{transcript.id}.json"
        filepath = os.path.abspath(join("/app/data/transcripts", filename))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(transcript), f, ensure_ascii=False, indent=2)
        
        # Update in-memory storage
        _transcripts[transcript.id] = transcript
        
        return {
            "id": transcript.id,
            "filepath": filepath,
            "status": "success",
            "message": "Transcript saved successfully"
        }
        
    except Exception as e:
        error_msg = f"Error saving transcript: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {
            "id": "",
            "filepath": "",
            "status": "error",
            "message": error_msg
        }

def _save_transcript_impl(content: str, source: str = "", metadata: Optional[Dict] = None) -> Dict:
    """
    Internal implementation of save_transcript without the @tool decorator.
    This is used by other functions that need to save transcripts.
    """
    # Input validation
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string")
        
    # Ensure metadata is a dictionary
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")
        
    # Create transcript object
    transcript = Transcript(
        content=str(content),
        source=str(source) if source else "",
        metadata=metadata
    )
    
    # Ensure the transcripts directory exists
    os.makedirs("/app/data/transcripts", exist_ok=True)
    
    # Save to file
    filename = f"transcript_{transcript.id}.json"
    filepath = os.path.abspath(join("/app/data/transcripts", filename))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(asdict(transcript), f, ensure_ascii=False, indent=2)
    
    # Update in-memory storage
    _transcripts[transcript.id] = transcript
    
    return {
        "id": transcript.id,
        "filepath": filepath,
        "status": "success",
        "message": "Transcript saved successfully"
    }

@my_mcp.tool
def save_transcript(content: str, source: str = "", metadata: Optional[Dict] = None) -> Dict:
    """
    Save a transcript with metadata.
    
    Args:
        content (str): Transcript content (required)
        source (str): Source URL or file path (default: "")
        metadata (Dict, optional): Additional metadata (default: None)
        
    Returns:
        Dict: {
            "id": str,  # Unique ID of the saved transcript
            "filepath": str,  # Path to the saved transcript file
            "status": str,  # "success" or "error"
            "message": str  # Additional status message
        }
    """
    try:
        return _save_transcript_impl(content, source, metadata)
    except Exception as e:
        error_msg = f"Error saving transcript: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {
            "id": "",
            "filepath": "",
            "status": "error",
            "message": error_msg
        }

@my_mcp.tool
def get_youtube_transcript(url: str, save: bool = True) -> Dict:
    """
    Get and optionally save the transcript of a YouTube video.
    
    Args:
        url: URL of the YouTube video
        save: Whether to save the transcript
        
    Returns:
        Dict: Transcript information including content and ID if saved
    """
    try:
        content = fetch_youtube_transcript(url)
        
        if save:
            result = _save_transcript_impl(content, source=url, metadata={"type": "youtube"})
            result["content"] = content
            return result
        
        return {"content": content, "status": "success"}
        
    except Exception as e:
        error_msg = f"Error getting YouTube transcript: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {
            "content": "",
            "status": "error",
            "message": error_msg
        }




@my_mcp.tool
def generate_knowledge_graph_prompt() -> str:
    """
    Fetch the knowledge graph generation prompt that instructs an LLM to extract entities 
    and relationships from text and format them as a Neo4j-compatible JSON structure.
    
    This prompt guides the LLM to:
    - Extract entities as nodes with unique IDs, labels (types), and properties
    - Identify relationships between entities with types and directional connections
    - Output results in a structured JSON format ready for Neo4j import
    
    Usage:
        1. Call this function to get the prompt template
        2. Send the modified prompt to an LLM
        3. The LLM will return a JSON with "nodes" and "relationships" arrays
    
    Returns:
        str: A detailed prompt template for knowledge graph generation
    """
    return get_knowledge_graph_prompt()

@my_mcp.tool
def generate_notes_prompt(knowledge_graph_json: str = None) -> str:
    """
    Returns the reusable prompt template that an LLM agent can use to:
      1. Transcribe audio (with speaker labels and timestamps) or accept a transcript.
      2. Organize content into structured, Markdown-formatted notes.
      3. Summarize key points, definitions, examples, data, and dates.
      4. Produce 'Key Takeaways' and 'Action Items' sections.
      5. Flag unclear segments and verify terminology.

    Returns:
        str: A detailed prompt template for high-quality notes generation
    """
    return get_notes_prompt(knowledge_graph_json)

@my_mcp.tool
def generate_sentiment_analysis_prompt(text: str) -> str:
    """
    Returns the sentiment analysis prompt template that an LLM agent can use to:
      1. Analyze overall sentiment classification (Positive/Negative/Neutral/Mixed)
      2. Assess sentiment intensity and emotional dimensions
      3. Identify key sentiment indicators and phrases
      4. Track sentiment trends throughout the content
      5. Provide contextual insights and actionable recommendations
      6. Return structured JSON format with comprehensive analysis

    Returns:
        str: A detailed prompt template for comprehensive sentiment analysis
    """
    return get_sentiment_analysis_prompt(text)

@my_mcp.tool
def generate_topic_modeling_prompt(text: str) -> str:
    """
    Returns the topic modeling prompt template that an LLM agent can use to:
      1. Analyze educational content to determine if it's tutorial or lecture
      2. Extract domain-specific information and technical elements
      3. Identify learning outcomes, prerequisites, and skill development
      4. Classify content difficulty and target audience
      5. Extract practical applications and real-world relevance
      6. Return structured JSON format with comprehensive educational analysis

    Returns:
        str: A detailed prompt template for comprehensive topic modeling
    """
    return get_topic_modeling_prompt(text)


@my_mcp.tool
def save_analysis(analysis_json: str, analysis_type: str) -> str:
    """
    Save the analysis JSON to a file in the appropriate subfolder.
    
    Args:
        analysis_json: JSON string containing analysis data
        analysis_type: Type of analysis (knowledge_graph, sentiment_analysis, or topic_modeling)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if analysis_type == "knowledge_graph":
        os.makedirs("/app/data/knowledge_graphs", exist_ok=True)
        path = f"/app/data/knowledge_graphs/knowledge_graph_{timestamp}.json"
    elif analysis_type == "sentiment_analysis":
        os.makedirs("/app/data/sentiment_analysis", exist_ok=True)
        path = f"/app/data/sentiment_analysis/sentiment_analysis_{timestamp}.json"
    elif analysis_type == "topic_modeling":
        os.makedirs("/app/data/topic_modeling", exist_ok=True)
        path = f"/app/data/topic_modeling/topic_modeling_{timestamp}.json"
    else:
        raise ValueError(f"Invalid analysis type: {analysis_type}")
    with open(path, "w") as f:
        f.write(analysis_json)
    return f"Analysis saved to {path}"

def _create_note_impl(title: str, content: str, transcript_id: Optional[str] = None, 
                     tags: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> Dict:
    """
    Internal implementation of note creation without the @tool decorator.
    Saves notes in Markdown format with YAML front matter.
    """
    # Create note object
    note = Note(
        title=title,
        content=content,
        transcript_id=transcript_id,
        tags=tags or [],
        metadata=metadata or {}
    )
    _notes[note.id] = note
    
    # Ensure notes directory exists
    os.makedirs("/app/data/notes_md", exist_ok=True)
    
    # Create safe filename
    safe_title = "".join(c if c.isalnum() else "_" for c in title[:50])
    filename = f"{safe_title}_{note.id}.md"
    filepath = os.path.abspath(join("/app/data/notes_md", filename))
    
    # Prepare metadata for YAML front matter
    metadata = {
        "id": note.id,
        "title": title,
        "created_at": note.created_at,
        "updated_at": note.updated_at,
        "transcript_id": transcript_id,
        "tags": tags or [],
        **(metadata or {})
    }
    
    # Save as Markdown with YAML front matter
    with open(filepath, 'w', encoding='utf-8') as f:
        # Write YAML front matter
        f.write("---\n")
        for key, value in metadata.items():
            if value is not None:  # Skip None values
                if isinstance(value, str):
                    f.write(f"{key}: {value}\n")
                elif isinstance(value, (list, dict)):
                    f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
                else:
                    f.write(f"{key}: {value}\n")
        f.write("---\n\n")
        
        # Write content
        f.write(content)
        
        # Add footer with metadata
        f.write("\n\n---\n")
        f.write(f"*Note ID: {note.id}*  ")
        f.write(f"*Created: {datetime.fromisoformat(note.created_at).strftime('%Y-%m-%d %H:%M:%S')}*  ")
        if tags:
            f.write(f"\n*Tags: {', '.join(tags)}*")
    
    return {
        "id": note.id, 
        "title": note.title, 
        "filepath": filepath,
        "status": "success",
        "message": f"Note saved as Markdown to {filepath}"
    }

def _get_note_impl(note_id: str) -> Optional[Dict]:
    """
    Internal implementation of get_note without the @tool decorator.
    Reads notes from Markdown files with YAML front matter.
    """
    # Check in-memory cache first
    if note_id in _notes:
        return asdict(_notes[note_id])
    
    # Try to find the markdown file
    notes_dir = "/app/data/notes_md"
    if not os.path.exists(notes_dir):
        return None
        
    # Look for the note file with the given ID
    for filename in os.listdir(notes_dir):
        if f"_{note_id}.md" in filename:
            filepath = os.path.join(notes_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse YAML front matter
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        yaml_content = parts[1].strip()
                        note_content = parts[2].strip()
                        
                        try:
                            import yaml
                            metadata = yaml.safe_load(yaml_content) or {}
                            
                            # Create note object
                            note = Note(
                                id=metadata.get('id', note_id),
                                title=metadata.get('title', ''),
                                content=note_content,
                                transcript_id=metadata.get('transcript_id'),
                                tags=metadata.get('tags', []),
                                metadata={k: v for k, v in metadata.items() 
                                         if k not in ['id', 'title', 'transcript_id', 'tags', 
                                                     'created_at', 'updated_at']}
                            )
                            
                            # Update cache
                            _notes[note.id] = note
                            return asdict(note)
                            
                        except Exception as e:
                            print(f"Error parsing YAML in note {note_id}: {str(e)}", file=sys.stderr)
                            
            except Exception as e:
                print(f"Error loading note {note_id}: {str(e)}", file=sys.stderr)
    
    # Fallback to old JSON format if exists
    old_filepath = join("/app/data/notes", f"note_{note_id}.json")
    if exists(old_filepath):
        try:
            with open(old_filepath, 'r', encoding='utf-8') as f:
                note_data = json.load(f)
                note = Note(**note_data)
                _notes[note.id] = note  # Cache in memory
                return asdict(note)
        except Exception as e:
            print(f"Error loading old format note {note_id}: {str(e)}", file=sys.stderr)
    
    return None

@my_mcp.tool
def create_notes(title: str, content: str, transcript_id: Optional[str] = None, 
               tags: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> Dict:
    """
    Create a new note with metadata.
    
    Args:
        title: Note title
        content: Note content (markdown supported)
        transcript_id: Optional ID of related transcript
        tags: List of tags for categorization
        metadata: Additional metadata
        
    Returns:
        Dict: {
            "id": str,  # Note ID
            "title": str,  # Note title
            "filepath": str,  # Path to saved note file
            "status": str  # "success" or "error"
        }
    """
    try:
        return _create_note_impl(title, content, transcript_id, tags or [], metadata or {})
    except Exception as e:
        error_msg = f"Error creating note: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {
            "id": "",
            "title": "",
            "filepath": "",
            "status": "error",
            "message": error_msg
        }

@my_mcp.tool
def save_notes(content: str, title: str = "Untitled", path: str = None, 
              tags: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> Dict:
    """
    Save notes with metadata to a Markdown (.md) file.
    
    Args:
        content: Notes content (markdown formatted)
        title: Title for the notes (will be used as H1 heading)
        path: Optional custom path (if not provided, will use default location)
        tags: Optional list of tags to include in metadata
        metadata: Additional metadata to include in the note
        
    Returns:
        Dict: {
            "id": str,  # Note ID
            "title": str,  # Note title
            "filepath": str,  # Path to saved markdown file
            "status": str,  # "success" or "error"
            "message": str  # Status message
        }
    """
    try:
        # Ensure content is properly formatted as markdown
        if not content.strip().startswith('#'):
            content = f"# {title}\n\n{content}"
            
        # Create note with metadata
        note_metadata = metadata or {}
        if tags:
            note_metadata['tags'] = tags
            
        note_info = _create_note_impl(
            title=title,
            content=content,
            tags=tags or [],
            metadata=note_metadata
        )
        
        # Determine output path
        if not path:
            # Use default path if none provided
            os.makedirs("/app/data/notes_md", exist_ok=True)
            safe_title = "".join(c if c.isalnum() else "_" for c in title)
            path = os.path.abspath(f"/app/data/notes_md/{safe_title}_{note_info['id']}.md")
        elif not path.lower().endswith('.md'):
            # Ensure .md extension
            path = f"{path}.md"
        
        # Save as markdown file
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                # Add YAML front matter for metadata
                f.write(f"---\n")
                f.write(f"title: {title}\n")
                if tags:
                    f.write(f"tags: {', '.join(tags)}\n")
                if metadata:
                    for key, value in metadata.items():
                        if key not in ['title', 'tags']:  # Already handled
                            f.write(f"{key}: {value}\n")
                f.write(f"date: {datetime.utcnow().isoformat()}\n")
                f.write(f"id: {note_info['id']}\n")
                f.write("---\n\n")
                
                # Add content
                f.write(content)
                
                # Add footer with metadata
                f.write("\n\n---\n")
                f.write(f"*Note ID: {note_info['id']}*  ")
                f.write(f"*Created: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
                
        except Exception as e:
            error_msg = f"Could not save markdown file: {str(e)}"
            print(f"Warning: {error_msg}", file=sys.stderr)
            return {
                **note_info,
                "status": "error",
                "message": error_msg
            }
        
        return {
            **note_info,
            "filepath": path,  # Return the actual path used
            "status": "success",
            "message": f"Notes saved successfully to {path}"
        }
        
    except Exception as e:
        error_msg = f"Error saving notes: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {
            "id": "",
            "title": "",
            "filepath": "",
            "status": "error",
            "message": error_msg
        }

@my_mcp.tool
def get_notes(note_id: str) -> Dict:
    """
    Get a note by ID.
    
    Args:
        note_id: ID of the note to retrieve
        
    Returns:
        Dict: {
            "id": str,  # Note ID
            "title": str,  # Note title
            "content": str,  # Note content
            "status": str,  # "success" or "error"
            "message": str  # Additional status message
        }
    """
    try:
        note = _get_note_impl(note_id)
        if note:
            return {
                **note,
                "status": "success",
                "message": "Note retrieved successfully"
            }
        return {
            "id": "",
            "title": "",
            "content": "",
            "status": "error",
            "message": f"Note with ID {note_id} not found"
        }
    except Exception as e:
        error_msg = f"Error retrieving note: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {
            "id": "",
            "title": "",
            "content": "",
            "status": "error",
            "message": error_msg
        }

@my_mcp.tool
def read_notes(path: str = None, note_id: str = None) -> Dict:
    """
    Read notes from a file or by note ID. Supports Markdown with YAML front matter.

    Args:
        path (str): Path to the file containing the notes
        note_id (str): ID of the note to retrieve (alternative to path)

    Returns:
        Dict: {
            "content": str,  # The notes content
            "title": str,  # Note title (if available)
            "status": str,  # "success" or "error"
            "message": str,  # Additional status message
            "metadata": Dict  # Any metadata from YAML front matter
        }
    """
    try:
        if note_id:
            # First try to get the note from our internal storage
            note = _get_note_impl(note_id)
            if note:
                return {
                    "content": note.get("content", ""),
                    "title": note.get("title", ""),
                    "status": "success",
                    "message": "Note retrieved successfully",
                    "metadata": note.get("metadata", {})
                }
            
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for YAML front matter
                metadata = {}
                note_content = content
                
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        try:
                            import yaml
                            yaml_content = parts[1].strip()
                            metadata = yaml.safe_load(yaml_content) or {}
                            note_content = parts[2].strip()
                        except Exception as e:
                            print(f"Error parsing YAML front matter: {str(e)}", file=sys.stderr)
                
                return {
                    "content": note_content,
                    "title": metadata.get('title', os.path.splitext(os.path.basename(path))[0]),
                    "status": "success",
                    "message": "File read successfully",
                    "metadata": metadata
                }
        
        return {
            "content": "",
            "title": "",
            "status": "error",
            "message": "No valid path or note_id provided",
            "metadata": {}
        }
        
    except Exception as e:
        error_msg = f"Error reading notes: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {
            "content": "",
            "title": "",
            "status": "error",
            "message": error_msg,
            "metadata": {}
        }

if __name__ == "__main__":
    print("Starting MCP server on http://0.0.0.0:8000")
    my_mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
