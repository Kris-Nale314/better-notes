# 🧭 The Planner Agent: Orchestrating Intelligence

## What Is the Planner Agent?

The Planner Agent (formerly called the Instructor Agent) is a meta-agent that designs and orchestrates how other agents analyze documents. Unlike the specialized agents that process content directly, the Planner creates context-aware, document-specific instructions that guide the entire analysis process.

Think of the Planner as the director of a movie - it doesn't appear on screen, but it coordinates all the actors to create a cohesive performance.

## How It Works

The Planner receives three key inputs:
1. **Document Metadata** - Summary, structure, length, document type
2. **User Preferences** - Detail level, focus areas, custom instructions
3. **Configuration** - Agent roles and capabilities from JSON config files

It then generates a tailored plan containing:
- **Agent Instructions** - Specific guidance for each specialized agent
- **Emphasis Areas** - What each agent should prioritize for this document
- **Analysis Strategy** - How to approach this particular document type

```python
plan = planner.create_plan(
    document_info=document_info,
    user_preferences=user_preferences,
    crew_type="issues"
)
```

## Why It Matters

The Planner transforms Better Notes from a static pipeline into an adaptive system:

1. **Document-Aware Processing**
   - Meeting transcripts get different instructions than technical reports
   - Long documents receive different strategies than short ones

2. **User Preference Integration**
   - Detail level changes how agents process information
   - Focus areas (technical, process, etc.) influence agent priorities

3. **Configurability Without Code Changes**
   - New analysis types can be created through configuration alone
   - Behavior can be adjusted without modifying agent implementations

4. **System Adaptability**
   - The same agents can approach different documents differently
   - The system can evolve without architectural changes

## Architecture Integration

The Planner sits between the Orchestrator and specialized agents:

```
Orchestrator → Planner → Specialized Agents
```

1. Orchestrator analyzes document and collects user preferences
2. Planner creates document-specific instructions
3. Instructions are distributed to specialized agents
4. Agents execute with tailored guidance

## Implementation Details

The PlannerAgent:
- Inherits from BaseAgent with type="planner" and crew_type="meta"
- Creates JSON-structured instructions for each agent
- Includes fallback mechanisms if planning fails
- Provides metadata about its planning decisions

## Future Potential

The Planner enables future capabilities like:
- Multi-document analysis with cross-referencing instructions
- Adaptive processing based on content complexity
- Dynamic agent allocation depending on document needs
- Learning from previous analyses to improve future planning

## Summary

The Planner Agent represents the "brain" of Better Notes - it's what allows the system to think and adapt rather than just execute a fixed workflow. By separating planning from execution, Better Notes achieves both flexibility and specialization, making it capable of handling diverse document types with consistently high-quality results.