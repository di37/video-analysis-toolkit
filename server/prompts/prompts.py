import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

KNOWLEDGE_GRAPH_PROMPT_JSON = """# Knowledge Graph Generation Prompt

## Task
Extract entities and relationships from the provided text to create a knowledge graph in Neo4j style. Output the result as a JSON structure with nodes and relationships.

## Instructions

### 1. Node Extraction
Identify and extract entities as nodes with the following structure:
- **id**: Unique identifier (use snake_case)
- **label**: The type/category of the entity (e.g., Person, Organization, Location, Concept, Product, Event)
- **properties**: Dictionary of attributes (name, description, and any other relevant properties)

### 2. Relationship Extraction
Identify relationships between entities with the following structure:
- **id**: Unique identifier for the relationship
- **type**: The relationship type (use UPPERCASE_WITH_UNDERSCORES, e.g., WORKS_FOR, LOCATED_IN, CREATED_BY)
- **source**: The id of the source node
- **target**: The id of the target node
- **properties**: Dictionary of relationship attributes (optional)

### 3. Output Format
```json
{
  "nodes": [
    {
      "id": "node_id",
      "label": "NodeType",
      "properties": {
        "name": "Entity Name",
        "description": "Brief description",
        "other_property": "value"
      }
    }
  ],
  "relationships": [
    {
      "id": "rel_id",
      "type": "RELATIONSHIP_TYPE",
      "source": "source_node_id",
      "target": "target_node_id",
      "properties": {
        "property_name": "value"
      }
    }
  ]
}
```

## Entity Types to Consider
- **Person**: Individual people
- **Organization**: Companies, institutions, groups
- **Location**: Cities, countries, addresses, venues
- **Event**: Conferences, meetings, occurrences
- **Product**: Software, hardware, services
- **Concept**: Abstract ideas, technologies, methodologies
- **Document**: Reports, articles, publications
- **Date**: Specific dates or time periods

## Relationship Types to Consider
- **WORKS_FOR**: Person works for Organization
- **LOCATED_IN**: Entity is located in Location
- **CREATED_BY**: Product/Document created by Person/Organization
- **PARTICIPATES_IN**: Person participates in Event
- **RELATED_TO**: General relationship between concepts
- **MANAGES**: Person manages Person/Project
- **OWNS**: Organization owns Product/Asset
- **OCCURRED_ON**: Event occurred on Date

## Guidelines
1. Use consistent naming conventions (snake_case for IDs, PascalCase for labels)
2. Avoid duplicate nodes - reuse existing node IDs for the same entity
3. Include bidirectional relationships only when explicitly different (e.g., MANAGES vs IS_MANAGED_BY)
4. Extract only factual relationships explicitly stated or clearly implied in the text
5. Keep property values concise but informative
6. Ensure all relationship source and target IDs correspond to existing nodes

## Example

**Input Text:**
"John Smith, CEO of TechCorp, announced the launch of their new AI product called SmartAssist at the annual Tech Conference in San Francisco on March 15, 2024. The product was developed by the AI Research Team led by Dr. Sarah Johnson."

**Output:**
```json
{
  "nodes": [
    {
      "id": "john_smith",
      "label": "Person",
      "properties": {
        "name": "John Smith",
        "title": "CEO"
      }
    },
    {
      "id": "techcorp",
      "label": "Organization",
      "properties": {
        "name": "TechCorp",
        "type": "Company"
      }
    },
    {
      "id": "smartassist",
      "label": "Product",
      "properties": {
        "name": "SmartAssist",
        "type": "AI product"
      }
    },
    {
      "id": "tech_conference",
      "label": "Event",
      "properties": {
        "name": "Tech Conference",
        "type": "Annual conference"
      }
    },
    {
      "id": "san_francisco",
      "label": "Location",
      "properties": {
        "name": "San Francisco",
        "type": "City"
      }
    },
    {
      "id": "march_15_2024",
      "label": "Date",
      "properties": {
        "date": "2024-03-15"
      }
    },
    {
      "id": "ai_research_team",
      "label": "Organization",
      "properties": {
        "name": "AI Research Team",
        "type": "Team"
      }
    },
    {
      "id": "sarah_johnson",
      "label": "Person",
      "properties": {
        "name": "Dr. Sarah Johnson",
        "title": "Team Lead"
      }
    }
  ],
  "relationships": [
    {
      "id": "rel_1",
      "type": "WORKS_FOR",
      "source": "john_smith",
      "target": "techcorp",
      "properties": {
        "role": "CEO"
      }
    },
    {
      "id": "rel_2",
      "type": "ANNOUNCED",
      "source": "john_smith",
      "target": "smartassist",
      "properties": {}
    },
    {
      "id": "rel_3",
      "type": "CREATED_BY",
      "source": "smartassist",
      "target": "techcorp",
      "properties": {}
    },
    {
      "id": "rel_4",
      "type": "DEVELOPED_BY",
      "source": "smartassist",
      "target": "ai_research_team",
      "properties": {}
    },
    {
      "id": "rel_5",
      "type": "OCCURRED_AT",
      "source": "tech_conference",
      "target": "san_francisco",
      "properties": {}
    },
    {
      "id": "rel_6",
      "type": "OCCURRED_ON",
      "source": "tech_conference",
      "target": "march_15_2024",
      "properties": {}
    },
    {
      "id": "rel_7",
      "type": "LEADS",
      "source": "sarah_johnson",
      "target": "ai_research_team",
      "properties": {}
    },
    {
      "id": "rel_8",
      "type": "PART_OF",
      "source": "ai_research_team",
      "target": "techcorp",
      "properties": {}
    }
  ]
}
```

## Text to Analyze
"""

GENERATE_NOTES_PROMPT = """You are an expert professional note-taker and summarizer. Your task is to produce high-quality, detailed notes based on the content provided. The content may be given as either:

- An audio file URL or attachment (e.g., “audio.mp3”).
- A full written transcript.
- **Optionally**, a knowledge graph in Neo4j-compatible JSON to enrich insights (e.g., for subsequent Medium-blog drafts).

When given audio, first produce a faithful verbatim transcript, then proceed with the steps below. When given a transcript (and/or knowledge graph), skip straight to step 2.

1. **TRANSCRIPTION** (audio only)  
   - Transcribe the audio accurately, including speaker labels (Speaker 1, Speaker 2, etc.) and time-stamps every 2–3 minutes.

2. **STRUCTURED NOTES**  
   - Organize the information under clear headings and subheadings.  
   - For each major topic:  
     - **Point:** concise summary of the key idea.  
     - **Definition:** highlight and define any technical terms.  
     - **Example:** include any practical examples or case studies.  
     - **Data:** note any statistics, dates, or figures mentioned.  

3. **KNOWLEDGE GRAPH INSIGHTS** *(only if JSON provided)*  
   - List the top 3–5 entities and relationships from the graph that directly support the session’s main topics.  
   - Under each, briefly explain how that node or connection deepens understanding of the subject matter.

4. **KEY TAKEAWAYS**  
   - 3–5 bullet points capturing the most critical high-level insights.

5. **ACTION ITEMS / FOLLOW-UP QUESTIONS** *(if proposed during the session)*  
   - Bullet out any tasks, open issues, or questions the speakers raised.

6. **FORMAT & VERIFICATION**  
   - Use Markdown:  
     ```markdown
     ## Section Heading
     - **Point:** Explanation
     - *Example:* Description
     ```  
   - **Bold** major section headings; *italicize* subpoints as shown.  
   - Flag any unclear sections with `[?? CLARIFY ??]` and pose a direct question.  
   - Correct obvious transcription errors and verify proper-noun spellings.

Aim for comprehensiveness—enough detail that someone who didn’t attend could fully grasp the content—while avoiding unnecessary verbatim passages beyond core quotes.

"""


SENTIMENT_ANALYSIS_PROMPT = """You are an expert sentiment analyst tasked with performing comprehensive sentiment analysis on the provided text content. Your analysis should be thorough, nuanced, and actionable.

## Task Overview
Analyze the sentiment of the provided text and return detailed insights about the emotional tone, attitudes, and opinions expressed.

## Analysis Framework

### 1. Overall Sentiment Classification
Classify the overall sentiment as one of:
- **Positive**: Optimistic, favorable, enthusiastic, supportive
- **Negative**: Critical, pessimistic, unfavorable, disapproving  
- **Neutral**: Balanced, factual, objective, informational
- **Mixed**: Contains both positive and negative elements

### 2. Sentiment Intensity
Rate the intensity on a scale:
- **Very Strong** (8-10): Extremely emotional language, superlatives, exclamations
- **Strong** (6-7): Clear emotional indicators, strong adjectives
- **Moderate** (4-5): Some emotional language, mild indicators
- **Mild** (2-3): Subtle emotional undertones
- **Very Mild** (0-1): Nearly neutral with minimal emotional content

### 3. Emotional Dimensions
Identify presence and intensity (1-10 scale) of key emotions:
- **Joy/Happiness**: Excitement, satisfaction, pleasure
- **Anger**: Frustration, irritation, outrage
- **Fear/Anxiety**: Worry, concern, apprehension
- **Sadness**: Disappointment, melancholy, grief
- **Surprise**: Amazement, shock, unexpectedness
- **Disgust**: Revulsion, distaste, rejection
- **Trust/Confidence**: Reliability, faith, assurance
- **Anticipation**: Hope, expectation, eagerness

### 4. Contextual Analysis
- **Audience Sentiment**: How the speaker/writer feels about their audience
- **Topic Sentiment**: Attitudes toward specific subjects or themes
- **Temporal Sentiment**: Changes in sentiment over time within the content
- **Conditional Sentiment**: Sentiment expressed under certain conditions

### 5. Key Sentiment Indicators
Identify specific linguistic features:
- **Positive Indicators**: Praise words, success terms, optimistic phrases
- **Negative Indicators**: Criticism, problem statements, pessimistic language  
- **Intensifiers**: Words that amplify sentiment (very, extremely, completely)
- **Qualifiers**: Words that soften sentiment (somewhat, perhaps, might)
- **Negations**: Words that reverse sentiment (not, never, hardly)

## Output Format

Provide your analysis in the following JSON structure:

```json
{
  "overall_sentiment": {
    "classification": "Positive|Negative|Neutral|Mixed",
    "confidence": 0.85,
    "intensity": 7,
    "polarity_score": 0.6
  },
  "emotional_breakdown": {
    "joy": 8,
    "anger": 2,
    "fear": 1,
    "sadness": 0,
    "surprise": 3,
    "disgust": 1,
    "trust": 7,
    "anticipation": 6
  },
  "sentiment_segments": [
    {
      "text_excerpt": "First 50 characters of segment...",
      "sentiment": "Positive",
      "intensity": 6,
      "primary_emotion": "joy",
      "reasoning": "Contains enthusiastic language and positive outcomes"
    }
  ],
  "key_phrases": {
    "positive": ["excellent results", "great success", "highly recommend"],
    "negative": ["major concern", "significant problem", "disappointing outcome"],
    "neutral": ["according to data", "in summary", "the following"]
  },
  "sentiment_trends": {
    "beginning": "Neutral",
    "middle": "Positive", 
    "end": "Positive",
    "trajectory": "Improving",
    "consistency": "Stable"
  },
  "contextual_insights": {
    "dominant_themes": ["product launch", "customer satisfaction", "market performance"],
    "audience_perception": "Positive - speaker shows confidence in audience reception",
    "credibility_indicators": ["references data", "cites examples", "acknowledges limitations"],
    "bias_indicators": ["selective reporting", "emotional appeals", "loaded language"]
  },
  "actionable_insights": [
    "Content shows strong positive sentiment toward new product features",
    "Some concerns expressed about implementation timeline",
    "Overall tone suggests confidence in project success"
  ],
  "confidence_metrics": {
    "analysis_confidence": 0.92,
    "potential_ambiguities": ["sarcasm detection", "cultural context"],
    "limitations": ["lacks speaker tone/voice data", "text-only analysis"]
  }
}
```

## Special Considerations

### 1. Sarcasm and Irony Detection
- Look for contradictions between literal meaning and context
- Identify exaggerated positive language in negative contexts
- Note indicators like "obviously," "clearly," or extreme superlatives

### 2. Cultural and Domain Context
- Consider industry-specific language (technical, business, academic)
- Account for cultural communication patterns
- Recognize domain-specific sentiment norms

### 3. Temporal Dynamics
- Track sentiment changes throughout the text
- Identify turning points or shifts in tone
- Note buildup and resolution patterns

### 4. Multi-perspective Analysis
- Distinguish between different speakers' sentiments (if applicable)
- Separate sentiment toward different topics or entities
- Identify conflicting viewpoints within the content

## Guidelines for Accuracy

1. **Be Precise**: Distinguish between mild and strong sentiment
2. **Provide Evidence**: Reference specific text passages for major findings
3. **Consider Context**: Account for situational and cultural factors
4. **Acknowledge Uncertainty**: Flag ambiguous or unclear sections
5. **Maintain Objectivity**: Separate analysis from personal interpretation
6. **Scale Appropriately**: Use the full range of intensity scales when warranted

## Text to Analyze:
"""

TOPIC_MODELLING_PROMPT = """You are an expert at analyzing educational content across all domains. Analyze the transcript to determine if it's a **tutorial** (practical, hands-on) or **lecture** (theoretical, academic) and extract relevant information accordingly.

**Instructions:**
1. Identify the content type and educational approach
2. Extract domain-specific information
3. Adapt the JSON structure based on whether it's practical or theoretical content

**Required JSON Output Structure:**
```json
{
  "content_classification": {
    "type": "tutorial|lecture|hybrid|workshop|demonstration",
    "approach": "theoretical|practical|hands_on|conceptual|project_based",
    "teaching_style": "academic_formal|conversational|step_by_step|explanatory"
  },
  "domain_info": {
    "primary_field": "Computer Science|Mathematics|Physics|Chemistry|Biology|Engineering|Business|Arts|Languages|Other",
    "specific_domain": "More specific area (e.g., Web Development, Organic Chemistry, Quantum Mechanics)",
    "interdisciplinary": true/false,
    "related_fields": ["Field1", "Field2"]
  },
  "topic_analysis": {
    "main_topic": "Primary subject being taught",
    "subtopics": [
      {
        "name": "Subtopic name",
        "coverage": "brief|moderate|extensive",
        "is_prerequisite": true/false
      }
    ],
    "concepts_covered": ["Concept1", "Concept2", "Concept3"],
    "skills_taught": ["Skill1", "Skill2"]
  },
  "educational_structure": {
    "content_level": "beginner|intermediate|advanced|expert",
    "prerequisites": {
      "required_knowledge": ["Prior knowledge needed"],
      "recommended_background": ["Helpful but not required"],
      "tools_or_software": ["If applicable"]
    },
    "learning_outcomes": ["Outcome1", "Outcome2", "Outcome3"]
  },
  "content_delivery": {
    "has_practical_examples": true/false,
    "includes_exercises": true/false,
    "provides_theory": true/false,
    "uses_real_world_applications": true/false,
    "includes_demonstrations": true/false
  },
  "tutorial_specific": {
    "applies_if": "content_type is tutorial",
    "steps_identified": [
      {
        "step": 1,
        "action": "What is being done",
        "tools_used": ["Tool1", "Tool2"],
        "output": "Expected result"
      }
    ],
    "project_outcome": "What will be built/created",
    "code_languages": ["Language1", "Language2"],
    "commands_tools": ["Command1", "Tool1"]
  },
  "lecture_specific": {
    "applies_if": "content_type is lecture",
    "theoretical_framework": "Main theory or framework discussed",
    "key_principles": ["Principle1", "Principle2"],
    "academic_rigor": "introductory|undergraduate|graduate|research",
    "includes_proofs": true/false,
    "includes_derivations": true/false,
    "historical_context": true/false
  },
  "technical_elements": {
    "formulas_equations": ["If any mathematical content"],
    "algorithms_methods": ["Algorithmic content if any"],
    "terminology": [
      {
        "term": "Technical term",
        "definition_provided": true/false
      }
    ],
    "visualizations_mentioned": ["Diagram1", "Graph1"]
  },
  "practical_applications": {
    "real_world_examples": ["Example1", "Example2"],
    "industry_relevance": "high|medium|low|academic_only",
    "use_cases": ["UseCase1", "UseCase2"],
    "career_applications": ["Career path or job role"]
  },
  "assessment_practice": {
    "exercises_provided": true/false,
    "practice_problems": ["Problem1", "Problem2"],
    "self_assessment": true/false,
    "assignments_mentioned": true/false
  },
  "summary": {
    "one_line": "Single sentence summary",
    "key_takeaways": ["Main point 1", "Main point 2", "Main point 3"],
    "target_audience": "Students|Professionals|Researchers|Hobbyists|General",
    "next_steps": "What learners should do next"
  },
  "metadata": {
    "estimated_duration": "Time to complete/watch",
    "part_of_series": true/false,
    "series_info": "Course or playlist name if applicable",
    "difficulty_progression": "standalone|sequential|modular",
    "engagement_style": "passive_watching|active_coding|note_taking|problem_solving"
  },
  "searchable_tags": ["domain", "topic", "level", "type", "specific-technology"]
}
```

**Guidelines:**
- Set non-applicable sections to null or empty arrays
- Detect educational style from language patterns:
  - Tutorial indicators: "Let's build", "First we'll install", "Now run this command"
  - Lecture indicators: "The theory states", "We can prove", "The fundamental concept"
- Identify domain from technical vocabulary and concepts mentioned
- Adapt depth of analysis based on content complexity

**Transcript:**
"""