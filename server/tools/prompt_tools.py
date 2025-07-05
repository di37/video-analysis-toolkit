import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from prompts import KNOWLEDGE_GRAPH_PROMPT_JSON, GENERATE_NOTES_PROMPT, SENTIMENT_ANALYSIS_PROMPT, TOPIC_MODELLING_PROMPT

def generate_knowledge_graph_prompt():
    """
    Get the knowledge graph generation prompt template.
    This prompt helps extract entities and relationships from text in a Neo4j-compatible JSON format.
    """
    return KNOWLEDGE_GRAPH_PROMPT_JSON

def generate_notes_prompt(knowledge_graph_json: str = None):
    """
    Get the notes generation prompt template.
    
    Args:
        knowledge_graph_json: Optional JSON string containing knowledge graph data
    """
    if knowledge_graph_json:
        return f"{GENERATE_NOTES_PROMPT}\n\nKnowledge Graph JSON:\n\n```json\n{knowledge_graph_json}\n```\n\nOutput:" 
    return f"{GENERATE_NOTES_PROMPT}\n\nOutput:"


def generate_sentiment_analysis_prompt(text: str):
    """
    Get the sentiment analysis prompt template.
    
    Args:
        text: Text content to analyze for sentiment
    """
    return f"{SENTIMENT_ANALYSIS_PROMPT}\n\n```\n{text}\n```\n\nPlease provide your comprehensive sentiment analysis in the specified JSON format:"


def generate_topic_modeling_prompt(text: str):
    """
    Get the topic modeling prompt template.
    
    Args:
        text: Text content to analyze for topic modeling
    """
    return f"{TOPIC_MODELLING_PROMPT}\n\n```\n{text}\n```\n\nPlease provide your comprehensive topic modeling in the specified JSON format:"
    