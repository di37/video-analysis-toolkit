import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
import json
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
import tempfile
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="MCP Powered YouTube Video Analysis Toolkit", page_icon="🚀", layout="wide")

# MCP Chat Functions
async def get_response_async(message, conversation_history=None):
    """Get response from MCP agent with full conversation context"""
    try:
        config_path = "/app/config/mcpServers.json"
        client = MCPClient.from_config_file(config_path)
        
        # Get model name and API key from session state
        model_name = st.session_state.get('model_name', 'gpt-4.1')
        api_key = st.session_state.get('api_key', '')
        
        # Use API key if provided, otherwise fall back to environment variable
        if api_key:
            llm = ChatOpenAI(model=model_name, api_key=api_key)
        else:
            llm = ChatOpenAI(model=model_name)
        
        agent = MCPAgent(llm=llm, client=client, max_steps=10)
        
        # Build context with conversation history
        if conversation_history and len(conversation_history) > 1:
            context = "Previous conversation:\n"
            for msg in conversation_history[:-1]:  # All messages except the current one
                role = "Human" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"
            context += f"\nCurrent question: {message}"
            query = context
        else:
            query = message
        
        result = await agent.run(query)
        return result
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def get_response(message, conversation_history=None):
    """Wrapper to run async function with conversation history"""
    try:
        return asyncio.run(get_response_async(message, conversation_history))
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(get_response_async(message, conversation_history))

# Knowledge Graph Functions
@st.cache_data
def load_knowledge_graph(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_knowledge_graph_from_upload(uploaded_file):
    return json.load(uploaded_file)

def create_networkx_graph(kg_data):
    G = nx.Graph()
    
    for node in kg_data['nodes']:
        properties = node['properties'].copy()
        properties['name'] = properties.get('name', node['id'])
        G.add_node(node['id'], 
                  label=node['label'],
                  **properties)
    
    for rel in kg_data['relationships']:
        G.add_edge(rel['source'], rel['target'], 
                  relationship=rel['type'],
                  **rel.get('properties', {}))
    
    return G

def get_node_colors(node_types=None):
    # Default colors for common node types
    default_colors = {
        'Person': '#FF6B6B',
        'Course': '#4ECDC4', 
        'Technology': '#45B7D1',
        'Concept': '#96CEB4',
        'Process': '#FFEAA7'
    }
    
    if node_types is None:
        return default_colors
    
    # Generate additional colors for node types not in defaults
    additional_colors = [
        '#FF9F43', '#54A0FF', '#5F27CD', '#00D2D3', 
        '#FF6B6B', '#C44569', '#F8B500', '#6C5CE7',
        '#A29BFE', '#FD79A8', '#E17055', '#00B894',
        '#0984E3', '#6C5CE7', '#A29BFE', '#FDCB6E'
    ]
    
    colors = default_colors.copy()
    color_index = 0
    
    for node_type in node_types:
        if node_type not in colors:
            colors[node_type] = additional_colors[color_index % len(additional_colors)]
            color_index += 1
    
    return colors

def create_interactive_graph(G, kg_data):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Get all unique node types from the data
    unique_node_types = list(set([node['label'] for node in kg_data['nodes']]))
    node_colors = get_node_colors(unique_node_types)
    
    for node in G.nodes():
        node_data = G.nodes[node]
        label = node_data['label']
        name = node_data['name']
        color = node_colors.get(label, '#888888')
        
        title = f"<b>{name}</b><br>Type: {label}<br>"
        for key, value in node_data.items():
            if key not in ['label', 'name']:
                title += f"{key.title()}: {value}<br>"
        
        net.add_node(node, label=name, color=color, title=title, size=20)
    
    for edge in G.edges():
        rel_type = G.edges[edge]['relationship']
        net.add_edge(edge[0], edge[1], label=rel_type, color="#888888", width=2)
    
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

def create_node_table(kg_data):
    nodes_df = pd.DataFrame([
        {
            'ID': node['id'],
            'Name': node['properties'].get('name', node['id']),
            'Type': node['label'],
            'Description': node['properties'].get('description', 'N/A')
        }
        for node in kg_data['nodes']
    ])
    return nodes_df

def create_relationship_table(kg_data, G):
    relationships_df = pd.DataFrame([
        {
            'Source': G.nodes[rel['source']]['name'],
            'Relationship': rel['type'],
            'Target': G.nodes[rel['target']]['name'],
            'Properties': ', '.join([f"{k}: {v}" for k, v in rel.get('properties', {}).items()]) or 'None'
        }
        for rel in kg_data['relationships']
    ])
    return relationships_df

def render_chat_tab():
    """Render the MCP Chat tab"""
    st.title("💬 YouTube Video Chatbot")
    
    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Get bot response with conversation history
        with st.spinner("Thinking..."):
            response = get_response(prompt, st.session_state.messages)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(response)
        
        st.rerun()

def render_knowledge_graph_tab():
    """Render the Knowledge Graph tab"""
    st.title("🕸️ Knowledge Graph Visualizer")
    st.markdown("Interactive visualization of your knowledge graph data")
    
    # Get selected knowledge graph from sidebar
    selected_kg = st.session_state.get('selected_kg', None)
    
    if not selected_kg:
        st.info("Please select a knowledge graph from the sidebar to visualize.")
        st.markdown("### How to get knowledge graphs:")
        st.markdown("1. Use the YouTube Video Chat to analyze a video")
        st.markdown("2. Ask the chatbot to generate a knowledge graph")
        st.markdown("3. The knowledge graph will be saved and available in the sidebar dropdown")
        st.markdown("### Expected JSON Format:")
        st.code('''
{
  "nodes": [
    {
      "id": "node1",
      "label": "Person",
      "properties": {
        "name": "John Doe",
        "description": "A person"
      }
    }
  ],
  "relationships": [
    {
      "source": "node1",
      "target": "node2", 
      "type": "KNOWS",
      "properties": {}
    }
  ]
}
        ''', language="json")
        return
    
    # Load the selected knowledge graph
    kg_dir = '/app/data/knowledge_graphs'
    kg_path = os.path.join(kg_dir, selected_kg)
    
    if not os.path.exists(kg_path):
        st.error(f"Selected knowledge graph file not found: {selected_kg}")
        return
    
    try:
        kg_data = load_knowledge_graph(kg_path)
        st.success(f"Successfully loaded knowledge graph: {selected_kg}")
    except json.JSONDecodeError:
        st.error("Invalid JSON format in the selected file.")
        return
    except Exception as e:
        st.error(f"Error loading knowledge graph: {str(e)}")
        return
    
    if kg_data:
        try:
            G = create_networkx_graph(kg_data)
            
            tab1, tab2, tab3 = st.tabs(["📊 Graph Visualization", "📋 Nodes", "🔗 Relationships"])
            
            with tab1:
                st.subheader("Interactive Knowledge Graph")
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    st.subheader("Legend")
                    # Get unique node types from the actual data
                    unique_node_types = list(set([node['label'] for node in kg_data['nodes']]))
                    colors = get_node_colors(unique_node_types)
                    
                    # Only show colors for node types that exist in the data
                    for node_type in unique_node_types:
                        color = colors[node_type]
                        st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; margin-right: 10px;"></div>{node_type}</div>', unsafe_allow_html=True)
                
                with col1:
                    html_file = create_interactive_graph(G, kg_data)
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=600)
                    os.unlink(html_file)
            
            with tab2:
                st.subheader("All Nodes")
                nodes_df = create_node_table(kg_data)
                st.dataframe(nodes_df, use_container_width=True)
                
                st.subheader("Filter by Node Type")
                selected_type = st.selectbox("Select node type:", ['All'] + list(set([node['label'] for node in kg_data['nodes']])))
                
                if selected_type != 'All':
                    filtered_nodes = nodes_df[nodes_df['Type'] == selected_type]
                    st.dataframe(filtered_nodes, use_container_width=True)
            
            with tab3:
                st.subheader("All Relationships")
                relationships_df = create_relationship_table(kg_data, G)
                st.dataframe(relationships_df, use_container_width=True)
                
                st.subheader("Filter by Relationship Type")
                selected_rel_type = st.selectbox("Select relationship type:", ['All'] + list(set([rel['type'] for rel in kg_data['relationships']])))
                
                if selected_rel_type != 'All':
                    filtered_rels = relationships_df[relationships_df['Relationship'] == selected_rel_type]
                    st.dataframe(filtered_rels, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing knowledge graph: {str(e)}")

def render_notes_tab():
    """Render the Generated Notes tab"""
    st.title("📝 Generated Notes")
    st.markdown("View and manage your generated video analysis notes")
    
    # Get selected notes from sidebar
    selected_notes = st.session_state.get('selected_notes', None)
    
    if not selected_notes:
        st.info("Please select notes from the sidebar to view their content.")
        st.markdown("### How to get notes:")
        st.markdown("1. Use the YouTube Video Chat to analyze a video")
        st.markdown("2. Ask the chatbot to generate notes from a transcript")
        st.markdown("3. The notes will be saved as .md files and available in the sidebar dropdown")
        return
    
    # Load the selected notes file
    notes_dir = '/app/data/notes_md'
    notes_path = os.path.join(notes_dir, selected_notes)
    
    if not os.path.exists(notes_path):
        st.error(f"Selected notes file not found: {selected_notes}")
        return
    
    try:
        with open(notes_path, 'r', encoding='utf-8') as f:
            notes_content = f.read()
        st.success(f"Successfully loaded notes: {selected_notes}")
    except Exception as e:
        st.error(f"Error loading notes: {str(e)}")
        return
    
    if notes_content:
        # Create tabs for different views
        view_tab1, view_tab2 = st.tabs(["📖 Rendered View", "📝 Raw Markdown"])
        
        with view_tab1:
            st.markdown(notes_content)
        
        with view_tab2:
            st.code(notes_content, language="markdown")
        
        # Download button
        st.download_button(
            label="💾 Download Notes",
            data=notes_content,
            file_name=selected_notes,
            mime="text/markdown"
        )

def render_transcripts_tab():
    """Render the Saved Transcripts tab"""
    st.title("📄 Saved Transcripts")
    st.markdown("View and manage your saved video transcripts")
    
    # Get selected transcript from sidebar
    selected_transcript = st.session_state.get('selected_transcript', None)
    
    if not selected_transcript:
        st.info("Please select a transcript from the sidebar to view its content.")
        st.markdown("### How to get transcripts:")
        st.markdown("1. Use the YouTube Video Chat to analyze a video")
        st.markdown("2. Ask the chatbot to get a YouTube transcript")
        st.markdown("3. The transcript will be saved and available in the sidebar dropdown")
        return
    
    # Load the selected transcript
    transcripts_dir = '/app/data/transcripts'
    transcript_path = os.path.join(transcripts_dir, selected_transcript)
    
    if not os.path.exists(transcript_path):
        st.error(f"Selected transcript file not found: {selected_transcript}")
        return
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Error loading transcript: {str(e)}")
        return
    
    # Create layout with metadata and content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display metadata
        st.subheader("📊 Transcript Info")
        
        st.write(f"**File:** {selected_transcript}")
        st.write(f"**ID:** {data.get('id', 'N/A')}")
        st.write(f"**Source:** {data.get('source', 'N/A')}")
        st.write(f"**Created:** {data.get('created_at', 'N/A')}")
        
        # Display metadata if available
        metadata = data.get('metadata', {})
        if metadata:
            st.write("**Metadata:**")
            for key, value in metadata.items():
                st.write(f"  - {key}: {value}")
        
        # Word count
        content = data.get('content', '')
        word_count = len(content.split()) if content else 0
        st.write(f"**Word Count:** {word_count}")
    
    with col2:
        if content:
            st.subheader("📖 Transcript Content")
            
            # Create tabs for different views
            content_tab1, content_tab2 = st.tabs(["📝 Formatted View", "📄 Raw Text"])
            
            with content_tab1:
                # Display content with better formatting
                st.markdown("### Transcript:")
                
                # Split content into paragraphs for better readability
                paragraphs = content.split('\n\n')
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        st.markdown(f"**[{i+1}]** {paragraph.strip()}")
                        st.markdown("") # Add spacing
            
            with content_tab2:
                # Display raw text in a text area for easy copying
                st.text_area(
                    "Raw transcript text:",
                    content,
                    height=400,
                    key="raw_transcript_content"
                )
            
            # Download buttons
            st.subheader("💾 Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as text file
                st.download_button(
                    label="📄 Download as TXT",
                    data=content,
                    file_name=f"transcript_{data.get('id', 'unknown')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Download as JSON file
                st.download_button(
                    label="📋 Download as JSON", 
                    data=json.dumps(data, indent=2, ensure_ascii=False),
                    file_name=selected_transcript,
                    mime="application/json"
                )
        else:
            st.warning("This transcript appears to be empty.")

def render_sentiment_analysis_tab():
    """Render the Sentiment Analysis tab"""
    st.title("📊 Sentiment Analysis")
    st.markdown("Analyze and visualize sentiment patterns from your video content")
    
    # Get selected sentiment analysis from sidebar
    selected_sentiment = st.session_state.get('selected_sentiment', None)
    
    if not selected_sentiment:
        st.info("Please select a sentiment analysis from the sidebar to visualize.")
        st.markdown("### How to get sentiment analysis:")
        st.markdown("1. Use the YouTube Video Chat to analyze a video")
        st.markdown("2. Ask the chatbot to perform sentiment analysis on a transcript")
        st.markdown("3. The sentiment analysis will be saved and available in the sidebar dropdown")
        return
    
    # Load the selected sentiment analysis
    sentiment_dir = '/app/data/sentiment_analysis'
    sentiment_path = os.path.join(sentiment_dir, selected_sentiment)
    
    if not os.path.exists(sentiment_path):
        st.error(f"Selected sentiment analysis file not found: {selected_sentiment}")
        return
    
    try:
        with open(sentiment_path, 'r', encoding='utf-8') as f:
            sentiment_data = json.load(f)
        st.success(f"Successfully loaded sentiment analysis: {selected_sentiment}")
    except Exception as e:
        st.error(f"Error loading sentiment analysis: {str(e)}")
        return
    
    if sentiment_data:
        # Create tabs for different visualizations
        overview_tab, segments_tab, emotions_tab, insights_tab = st.tabs([
            "📈 Overview", "🔍 Segments", "😊 Emotions", "💡 Insights"
        ])
        
        with overview_tab:
            st.subheader("Overall Sentiment Analysis")
            
            # Overall sentiment metrics
            overall = sentiment_data.get('overall_sentiment', {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Classification", overall.get('classification', 'N/A'))
            with col2:
                st.metric("Confidence", f"{overall.get('confidence', 0):.2f}")
            with col3:
                st.metric("Intensity", f"{overall.get('intensity', 0)}/10")
            with col4:
                st.metric("Polarity Score", f"{overall.get('polarity_score', 0):.2f}")
            
            # Sentiment trends visualization
            trends = sentiment_data.get('sentiment_trends', {})
            if trends:
                st.subheader("Sentiment Trajectory")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create a simple trajectory chart
                    trend_data = {
                        'Position': ['Beginning', 'Middle', 'End'],
                        'Sentiment': [trends.get('beginning', 'Unknown'), 
                                    trends.get('middle', 'Unknown'), 
                                    trends.get('end', 'Unknown')]
                    }
                    
                    # Map sentiment to numeric values for visualization
                    sentiment_map = {'Positive': 1, 'Negative': -1, 'Mixed': 0, 'Neutral': 0}
                    trend_data['Score'] = [sentiment_map.get(s, 0) for s in trend_data['Sentiment']]
                    
                    fig = px.line(trend_data, x='Position', y='Score', 
                                title="Sentiment Flow Throughout Content",
                                markers=True, line_shape='spline')
                    fig.update_layout(yaxis=dict(range=[-1.5, 1.5], 
                                               tickvals=[-1, 0, 1], 
                                               ticktext=['Negative', 'Neutral', 'Positive']))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Trajectory:** " + trends.get('trajectory', 'N/A'))
                    st.write("**Consistency:** " + trends.get('consistency', 'N/A'))
        
        with segments_tab:
            st.subheader("Sentiment by Segments")
            
            segments = sentiment_data.get('sentiment_segments', [])
            if segments:
                # Create segment visualization
                segment_df = pd.DataFrame([
                    {
                        'Segment': f"Segment {i+1}",
                        'Text': seg.get('text_excerpt', '')[:100] + '...',
                        'Sentiment': seg.get('sentiment', 'Unknown'),
                        'Intensity': seg.get('intensity', 0),
                        'Emotion': seg.get('primary_emotion', 'Unknown'),
                        'Reasoning': seg.get('reasoning', 'N/A')
                    }
                    for i, seg in enumerate(segments)
                ])
                
                # Segment intensity chart
                fig = px.bar(segment_df, x='Segment', y='Intensity', 
                           color='Sentiment', title="Sentiment Intensity by Segment",
                           hover_data=['Emotion', 'Text'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed segment table
                st.subheader("Detailed Segment Analysis")
                st.dataframe(segment_df, use_container_width=True)
                
                # Individual segment details
                st.subheader("Segment Details")
                selected_segment = st.selectbox("Select segment to view details:", 
                                              [f"Segment {i+1}" for i in range(len(segments))])
                
                if selected_segment:
                    idx = int(selected_segment.split()[1]) - 1
                    seg = segments[idx]
                    
                    st.write(f"**Text:** {seg.get('text_excerpt', 'N/A')}")
                    st.write(f"**Sentiment:** {seg.get('sentiment', 'N/A')}")
                    st.write(f"**Intensity:** {seg.get('intensity', 0)}/10")
                    st.write(f"**Primary Emotion:** {seg.get('primary_emotion', 'N/A')}")
                    st.write(f"**Reasoning:** {seg.get('reasoning', 'N/A')}")
        
        with emotions_tab:
            st.subheader("Emotional Analysis")
            
            emotional_breakdown = sentiment_data.get('emotional_breakdown', {})
            if emotional_breakdown:
                # Create emotion wheel/radar chart
                emotions = list(emotional_breakdown.keys())
                values = list(emotional_breakdown.values())
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=emotions,
                    fill='toself',
                    name='Emotional Profile'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    showlegend=False,
                    title="Emotional Breakdown (Wheel Chart)"
                )
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Emotion Scores")
                    for emotion, score in emotional_breakdown.items():
                        st.metric(emotion.title(), f"{score}/10")
                
                # Bar chart for emotions
                emotion_df = pd.DataFrame(list(emotional_breakdown.items()), 
                                        columns=['Emotion', 'Score'])
                fig2 = px.bar(emotion_df, x='Emotion', y='Score', 
                            title="Emotional Intensity Scores",
                            color='Score', color_continuous_scale='RdYlBu_r')
                st.plotly_chart(fig2, use_container_width=True)
        
        with insights_tab:
            st.subheader("Key Insights & Analysis")
            
            # Key phrases
            key_phrases = sentiment_data.get('key_phrases', {})
            if key_phrases:
                st.subheader("Key Phrases")
                
                phrase_tabs = st.tabs(["Positive", "Negative", "Neutral"])
                
                with phrase_tabs[0]:
                    positive_phrases = key_phrases.get('positive', [])
                    if positive_phrases:
                        for phrase in positive_phrases:
                            st.success(f"✅ {phrase}")
                    else:
                        st.info("No positive phrases identified")
                
                with phrase_tabs[1]:
                    negative_phrases = key_phrases.get('negative', [])
                    if negative_phrases:
                        for phrase in negative_phrases:
                            st.error(f"❌ {phrase}")
                    else:
                        st.info("No negative phrases identified")
                
                with phrase_tabs[2]:
                    neutral_phrases = key_phrases.get('neutral', [])
                    if neutral_phrases:
                        for phrase in neutral_phrases:
                            st.info(f"ℹ️ {phrase}")
                    else:
                        st.info("No neutral phrases identified")
            
            # Contextual insights
            contextual = sentiment_data.get('contextual_insights', {})
            if contextual:
                st.subheader("Contextual Analysis")
                
                themes = contextual.get('dominant_themes', [])
                if themes:
                    st.write("**Dominant Themes:**")
                    for theme in themes:
                        st.write(f"• {theme}")
                
                st.write(f"**Audience Perception:** {contextual.get('audience_perception', 'N/A')}")
                
                credibility = contextual.get('credibility_indicators', [])
                if credibility:
                    st.write("**Credibility Indicators:**")
                    for indicator in credibility:
                        st.write(f"• {indicator}")
                
                bias = contextual.get('bias_indicators', [])
                if bias:
                    st.write("**Potential Bias Indicators:**")
                    for indicator in bias:
                        st.write(f"• {indicator}")
            
            # Actionable insights
            actionable = sentiment_data.get('actionable_insights', [])
            if actionable:
                st.subheader("Actionable Insights")
                for i, insight in enumerate(actionable, 1):
                    st.write(f"**{i}.** {insight}")
            
            # Confidence metrics
            confidence = sentiment_data.get('confidence_metrics', {})
            if confidence:
                st.subheader("Analysis Confidence")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Analysis Confidence", 
                             f"{confidence.get('analysis_confidence', 0):.2f}")
                
                with col2:
                    ambiguities = confidence.get('potential_ambiguities', [])
                    if ambiguities:
                        st.write("**Potential Ambiguities:**")
                        for ambiguity in ambiguities:
                            st.write(f"• {ambiguity}")
                    
                    limitations = confidence.get('limitations', [])
                    if limitations:
                        st.write("**Analysis Limitations:**")
                        for limitation in limitations:
                            st.write(f"• {limitation}")
            
            # Download sentiment analysis
            st.subheader("💾 Export Analysis")
            st.download_button(
                label="📊 Download Sentiment Analysis (JSON)",
                data=json.dumps(sentiment_data, indent=2, ensure_ascii=False),
                file_name=selected_sentiment,
                mime="application/json"
            )

def render_topic_modeling_tab():
    """Render the Topic Modeling tab"""
    st.title("🎯 Topic Modeling")
    st.markdown("Analyze and classify educational content with detailed topic and domain insights")
    
    # Get selected topic modeling from sidebar
    selected_topic = st.session_state.get('selected_topic', None)
    
    if not selected_topic:
        st.info("Please select a topic modeling analysis from the sidebar to visualize.")
        st.markdown("### How to get topic modeling analysis:")
        st.markdown("1. Use the YouTube Video Chat to analyze educational content")
        st.markdown("2. Ask the chatbot to perform topic modeling on a transcript")
        st.markdown("3. The topic modeling analysis will be saved and available in the sidebar dropdown")
        return
    
    # Load the selected topic modeling analysis
    topic_dir = '/app/data/topic_modeling'
    topic_path = os.path.join(topic_dir, selected_topic)
    
    if not os.path.exists(topic_path):
        st.error(f"Selected topic modeling file not found: {selected_topic}")
        return
    
    try:
        with open(topic_path, 'r', encoding='utf-8') as f:
            topic_data = json.load(f)
        st.success(f"Successfully loaded topic modeling analysis: {selected_topic}")
    except Exception as e:
        st.error(f"Error loading topic modeling analysis: {str(e)}")
        return
    
    if topic_data:
        # Create tabs for different visualizations
        overview_tab, structure_tab, content_tab, technical_tab = st.tabs([
            "📋 Overview", "🎓 Educational Structure", "📚 Content Analysis", "⚙️ Technical Details"
        ])
        
        with overview_tab:
            st.subheader("Content Classification")
            
            # Content classification metrics
            classification = topic_data.get('content_classification', {})
            domain_info = topic_data.get('domain_info', {})
            summary = topic_data.get('summary', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Content Type", classification.get('type', 'N/A').title())
            with col2:
                st.metric("Approach", classification.get('approach', 'N/A').replace('_', ' ').title())
            with col3:
                st.metric("Primary Field", domain_info.get('primary_field', 'N/A'))
            with col4:
                st.metric("Teaching Style", classification.get('teaching_style', 'N/A').replace('_', ' ').title())
            
            # Domain information
            st.subheader("Domain Analysis")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Specific Domain:** {domain_info.get('specific_domain', 'N/A')}")
                st.write(f"**Interdisciplinary:** {'Yes' if domain_info.get('interdisciplinary', False) else 'No'}")
                
                related_fields = domain_info.get('related_fields', [])
                if related_fields:
                    st.write("**Related Fields:**")
                    for field in related_fields:
                        st.write(f"• {field}")
            
            with col2:
                if summary:
                    st.subheader("Quick Summary")
                    st.write(f"**Target Audience:** {summary.get('target_audience', 'N/A')}")
                    st.info(summary.get('one_line', 'No summary available'))
        
        with structure_tab:
            st.subheader("Educational Structure")
            
            educational = topic_data.get('educational_structure', {})
            topic_analysis = topic_data.get('topic_analysis', {})
            content_delivery = topic_data.get('content_delivery', {})
            
            # Level and prerequisites
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Course Information")
                st.metric("Content Level", educational.get('content_level', 'N/A').title())
                
                main_topic = topic_analysis.get('main_topic', 'N/A')
                st.write(f"**Main Topic:** {main_topic}")
                
                concepts = topic_analysis.get('concepts_covered', [])
                if concepts:
                    st.write("**Concepts Covered:**")
                    for concept in concepts[:5]:  # Show first 5
                        st.write(f"• {concept}")
                    if len(concepts) > 5:
                        st.write(f"... and {len(concepts) - 5} more")
            
            with col2:
                st.subheader("Prerequisites")
                prereqs = educational.get('prerequisites', {})
                
                required = prereqs.get('required_knowledge', [])
                if required:
                    st.write("**Required Knowledge:**")
                    for req in required:
                        st.write(f"• {req}")
                
                recommended = prereqs.get('recommended_background', [])
                if recommended:
                    st.write("**Recommended Background:**")
                    for rec in recommended:
                        st.write(f"• {rec}")
                
                tools = prereqs.get('tools_or_software', [])
                if tools:
                    st.write("**Tools/Software:**")
                    for tool in tools:
                        st.write(f"• {tool}")
            
            # Content delivery features
            st.subheader("Content Delivery Features")
            delivery_cols = st.columns(5)
            
            features = [
                ("Practical Examples", content_delivery.get('has_practical_examples', False)),
                ("Exercises", content_delivery.get('includes_exercises', False)),
                ("Theory", content_delivery.get('provides_theory', False)),
                ("Real-world Apps", content_delivery.get('uses_real_world_applications', False)),
                ("Demonstrations", content_delivery.get('includes_demonstrations', False))
            ]
            
            for i, (feature, has_feature) in enumerate(features):
                with delivery_cols[i]:
                    if has_feature:
                        st.success(f"✅ {feature}")
                    else:
                        st.info(f"❌ {feature}")
            
            # Learning outcomes
            outcomes = educational.get('learning_outcomes', [])
            if outcomes:
                st.subheader("Learning Outcomes")
                for i, outcome in enumerate(outcomes, 1):
                    st.write(f"**{i}.** {outcome}")
        
        with content_tab:
            st.subheader("Content Analysis")
            
            # Subtopics analysis
            subtopics = topic_analysis.get('subtopics', [])
            if subtopics:
                st.subheader("Subtopics Breakdown")
                
                subtopic_df = pd.DataFrame([
                    {
                        'Subtopic': sub.get('name', 'Unknown'),
                        'Coverage': sub.get('coverage', 'Unknown').title(),
                        'Is Prerequisite': 'Yes' if sub.get('is_prerequisite', False) else 'No'
                    }
                    for sub in subtopics
                ])
                
                # Coverage distribution chart
                coverage_counts = subtopic_df['Coverage'].value_counts()
                if not coverage_counts.empty:
                    fig = px.pie(values=coverage_counts.values, names=coverage_counts.index,
                               title="Subtopic Coverage Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(subtopic_df, use_container_width=True)
            
            # Skills and practical applications
            col1, col2 = st.columns(2)
            
            with col1:
                skills = topic_analysis.get('skills_taught', [])
                if skills:
                    st.subheader("Skills Taught")
                    for skill in skills:
                        st.write(f"• {skill}")
                
                # Tutorial-specific content
                tutorial = topic_data.get('tutorial_specific', {})
                if tutorial and tutorial.get('applies_if') == 'content_type is tutorial':
                    st.subheader("Tutorial Information")
                    st.write(f"**Project Outcome:** {tutorial.get('project_outcome', 'N/A')}")
                    
                    languages = tutorial.get('code_languages', [])
                    if languages:
                        st.write(f"**Programming Languages:** {', '.join(languages)}")
            
            with col2:
                # Practical applications
                practical = topic_data.get('practical_applications', {})
                if practical:
                    st.subheader("Practical Applications")
                    
                    examples = practical.get('real_world_examples', [])
                    if examples:
                        st.write("**Real-world Examples:**")
                        for example in examples:
                            st.write(f"• {example}")
                    
                    use_cases = practical.get('use_cases', [])
                    if use_cases:
                        st.write("**Use Cases:**")
                        for case in use_cases:
                            st.write(f"• {case}")
                    
                    st.metric("Industry Relevance", 
                             practical.get('industry_relevance', 'Unknown').replace('_', ' ').title())
            
            # Key takeaways
            takeaways = summary.get('key_takeaways', [])
            if takeaways:
                st.subheader("Key Takeaways")
                for i, takeaway in enumerate(takeaways, 1):
                    st.success(f"**{i}.** {takeaway}")
        
        with technical_tab:
            st.subheader("Technical Analysis")
            
            technical = topic_data.get('technical_elements', {})
            assessment = topic_data.get('assessment_practice', {})
            metadata = topic_data.get('metadata', {})
            
            # Technical elements
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Technical Elements")
                
                formulas = technical.get('formulas_equations', [])
                if formulas:
                    st.write("**Formulas/Equations:**")
                    for formula in formulas:
                        st.code(formula)
                
                algorithms = technical.get('algorithms_methods', [])
                if algorithms:
                    st.write("**Algorithms/Methods:**")
                    for algo in algorithms:
                        st.write(f"• {algo}")
                
                visualizations = technical.get('visualizations_mentioned', [])
                if visualizations:
                    st.write("**Visualizations Mentioned:**")
                    for viz in visualizations:
                        st.write(f"• {viz}")
            
            with col2:
                st.subheader("Assessment & Practice")
                
                practice_features = [
                    ("Exercises Provided", assessment.get('exercises_provided', False)),
                    ("Self Assessment", assessment.get('self_assessment', False)),
                    ("Assignments Mentioned", assessment.get('assignments_mentioned', False))
                ]
                
                for feature, has_feature in practice_features:
                    if has_feature:
                        st.success(f"✅ {feature}")
                    else:
                        st.info(f"❌ {feature}")
                
                problems = assessment.get('practice_problems', [])
                if problems:
                    st.write("**Practice Problems:**")
                    for problem in problems:
                        st.write(f"• {problem}")
            
            # Terminology
            terminology = technical.get('terminology', [])
            if terminology:
                st.subheader("Technical Terminology")
                
                term_df = pd.DataFrame([
                    {
                        'Term': term.get('term', 'Unknown'),
                        'Definition Provided': 'Yes' if term.get('definition_provided', False) else 'No'
                    }
                    for term in terminology
                ])
                
                st.dataframe(term_df, use_container_width=True)
            
            # Metadata
            st.subheader("Content Metadata")
            meta_col1, meta_col2 = st.columns(2)
            
            with meta_col1:
                st.write(f"**Estimated Duration:** {metadata.get('estimated_duration', 'N/A')}")
                st.write(f"**Part of Series:** {'Yes' if metadata.get('part_of_series', False) else 'No'}")
                st.write(f"**Engagement Style:** {metadata.get('engagement_style', 'N/A').replace('_', ' ').title()}")
            
            with meta_col2:
                series_info = metadata.get('series_info')
                if series_info:
                    st.write(f"**Series Info:** {series_info}")
                
                st.write(f"**Difficulty Progression:** {metadata.get('difficulty_progression', 'N/A').title()}")
                
                tags = topic_data.get('searchable_tags', [])
                if tags:
                    st.write("**Tags:** " + ", ".join(tags))
            
            # Download topic modeling analysis
            st.subheader("💾 Export Analysis")
            st.download_button(
                label="🎯 Download Topic Modeling Analysis (JSON)",
                data=json.dumps(topic_data, indent=2, ensure_ascii=False),
                file_name=selected_topic,
                mime="application/json"
            )


def main():
    st.title("🚀 MCP Powered YouTube Video Analysis Toolkit")
    st.sidebar.title("🚀 Navigation")
    
    # MCP Configuration section
    st.sidebar.subheader("⚙️ MCP Configuration")
    
    # Model name input
    model_name = st.sidebar.text_input(
        "Model Name",
        value=st.session_state.get('model_name', 'gpt-4.1'),
        help="Enter the model name to use for MCP operations"
    )
    st.session_state.model_name = model_name
    
    # API key input
    api_key = st.sidebar.text_input(
        "API Key",
        value=st.session_state.get('api_key', ''),
        type="password",
        help="Enter your OpenAI API key"
    )
    st.session_state.api_key = api_key
    
    st.sidebar.markdown("---")
    
    # Clear chat button in sidebar
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # File selection section in sidebar
    st.sidebar.subheader("📁 File Selection")
    
    # Transcript selection dropdown
    transcripts_dir = '/app/data/transcripts'
    transcript_files = glob.glob(f'{transcripts_dir}/*.json') if os.path.exists(transcripts_dir) else []
    transcript_files.sort(key=os.path.getmtime, reverse=True)
    
    transcript_options = ["None"] + [os.path.basename(f) for f in transcript_files]
    selected_transcript = st.sidebar.selectbox(
        "📄 Select Transcript:",
        transcript_options,
        help="Choose a saved transcript to view"
    )
    st.session_state.selected_transcript = selected_transcript if selected_transcript != "None" else None
    
    # Knowledge Graph selection dropdown
    kg_dir = '/app/data/knowledge_graphs'
    kg_files = glob.glob(f'{kg_dir}/knowledge_graph_*.json') if os.path.exists(kg_dir) else []
    kg_files.sort(key=os.path.getmtime, reverse=True)
    
    kg_options = ["None"] + [os.path.basename(f) for f in kg_files]
    selected_kg = st.sidebar.selectbox(
        "🕸️ Select Knowledge Graph:",
        kg_options,
        help="Choose a knowledge graph to visualize"
    )
    st.session_state.selected_kg = selected_kg if selected_kg != "None" else None
    
    # Notes selection dropdown
    notes_dir = '/app/data/notes_md'
    notes_files = glob.glob(f'{notes_dir}/*.md') if os.path.exists(notes_dir) else []
    notes_files.sort(key=os.path.getmtime, reverse=True)
    
    notes_options = ["None"] + [os.path.basename(f) for f in notes_files]
    selected_notes = st.sidebar.selectbox(
        "📝 Select Notes:",
        notes_options,
        help="Choose generated notes to view"
    )
    st.session_state.selected_notes = selected_notes if selected_notes != "None" else None
    
    # Sentiment Analysis selection dropdown
    sentiment_dir = '/app/data/sentiment_analysis'
    sentiment_files = glob.glob(f'{sentiment_dir}/sentiment_analysis_*.json') if os.path.exists(sentiment_dir) else []
    sentiment_files.sort(key=os.path.getmtime, reverse=True)
    
    sentiment_options = ["None"] + [os.path.basename(f) for f in sentiment_files]
    selected_sentiment = st.sidebar.selectbox(
        "📊 Select Sentiment Analysis:",
        sentiment_options,
        help="Choose a sentiment analysis to visualize"
    )
    st.session_state.selected_sentiment = selected_sentiment if selected_sentiment != "None" else None
    
    # Topic Modeling selection dropdown
    topic_dir = '/app/data/topic_modeling'
    topic_files = glob.glob(f'{topic_dir}/topic_modeling_*.json') if os.path.exists(topic_dir) else []
    topic_files.sort(key=os.path.getmtime, reverse=True)
    
    topic_options = ["None"] + [os.path.basename(f) for f in topic_files]
    selected_topic = st.sidebar.selectbox(
        "🎯 Select Topic Modeling:",
        topic_options,
        help="Choose a topic modeling analysis to visualize"
    )
    st.session_state.selected_topic = selected_topic if selected_topic != "None" else None
    
    # Create main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 YouTube Video Chat", "📄 Saved Transcripts", "🕸️ Knowledge Graph", "📝 Generated Notes", "📊 Sentiment Analysis", "🎯 Topic Modeling"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_transcripts_tab()
    
    with tab3:
        render_knowledge_graph_tab()
    
    with tab4:
        render_notes_tab()
    
    with tab5:
        render_sentiment_analysis_tab()
    
    with tab6:
        render_topic_modeling_tab()

if __name__ == "__main__":
    main()