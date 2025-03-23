# 🧠 Planner Agent Overview

## What is the Planner Agent?

The **Planner Agent** (previously called the Instructor Agent) is a **meta-agent** responsible for designing and coordinating the behavior of all other agents in a multi-agent analysis crew. It does not analyze the document itself—instead, it plans how other agents will perform their specialized roles, adapting to the document context and user preferences.

> Think of the Planner as the director of a play: it assigns roles, gives each actor their lines, and sets the tone for the performance based on the script (the document) and the audience (the user).

---

## What Does It Do?

The Planner Agent performs the following critical functions:

### 1. **Analyzes Metadata and Preferences**
- Ingests:
  - Summary of the document (typically the first 2,000 tokens)
  - Document structure and stats
  - User-defined instructions (e.g., "focus on risks")
  - Configuration JSON defining agent roles and goals

### 2. **Generates Specialized Instructions**
- Dynamically generates tailored instructions for each agent in the crew:
  - **Extractor Agent**
  - **Aggregator Agent**
  - **Evaluator Agent**
  - **Formatter Agent**
  - **Reviewer Agent** (optional)
- Provides two key components per agent:
  - `instructions`: Detailed task-specific guidance
  - `emphasis`: Key aspects to prioritize (e.g., severity, clarity)

### 3. **Supports Configurable Crews**
- Uses a config-driven design (`issues_config.json`, `actions_config.json`, etc.)
- Supports different crew types ("issues", "actions", "risks") by interpreting the `crew_type` parameter

### 4. **Provides Fallbacks**
- If LLM generation fails or is incomplete, it auto-generates default instructions based on heuristics and config values

---

## Why Use a Planner Agent?

### ✅ **Abstraction & Reusability**
- Centralizes orchestration logic so you can build many different crews (issues, risks, actions) with the same building blocks

### 🧩 **Modular Design**
- Keeps agent logic clean and focused—each agent does its job, and the Planner tells them what that job is

### 📐 **Adaptability**
- Each analysis is customized per document and user intent without modifying agent internals

### 🚀 **LLM-Enhanced Planning**
- Leverages LLMs to build dynamic task flows and guidance tailored to:
  - Document type (transcript, report, etc.)
  - Detail level (Essential → Comprehensive)
  - Domain-specific focus areas (e.g., security, budget, quality)

---

## Why It Matters in This App

The Planner is what makes **Better Notes** truly scalable and intelligent:
- It enables **macro-chunking** of long-form transcripts (~50K+ tokens) and provides intelligent context for each crew
- It separates **planning from doing**, allowing developers to create new assessment types without rewriting agent logic
- It creates a foundation for more advanced capabilities like **agent collaboration**, **multi-document comparison**, and **chain-of-thought diagnostics**

---

## Status

🛠️ The Planner Agent is actively being developed and enhanced.
- Future versions will include:
  - More advanced role assignment logic
  - Task graphs and dependencies
  - Multi-modal planning (e.g., combining documents with user input)

---

## Summary

The Planner Agent is the intelligence layer that makes Better Notes modular, adaptive, and powerful. It transforms a static set of AI agents into a dynamic, document-aware crew that can flexibly respond to real-world content and goals.

> It’s not just assigning roles—it’s designing how intelligence flows through the system.