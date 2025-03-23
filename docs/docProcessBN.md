# Document Processing in Better Notes

<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logo.svg" alt="Better-Notes logo" width="120px"/>
  <h3>Understanding the Macro-Chunking Approach</h3>
</div>

## Introduction

Document processing is a critical foundation of Better Notes' analytical capabilities. Unlike traditional document AI systems that use simplistic chunking or shallow parsing, Better Notes implements a sophisticated document handling pipeline that preserves context while managing token limitations. This document explains our approach to document processing and why it matters.

## The Macro-Chunking Paradigm

### What is Macro-Chunking?

Macro-chunking is Better Notes' approach to processing large documents that preserves meaningful context while managing the token limitations of large language models:

- Documents are divided into chunks of 7,000-10,000 tokens (roughly 2,000-3,000 words)
- Each chunk maintains complete paragraphs, sections, and contextual units
- This results in approximately 5-15 chunks for most documents
- Processing is parallelized across chunks for efficiency

### Macro-Chunking vs. Traditional RAG Chunking

<table>
<tr>
  <th>Aspect</th>
  <th>Better Notes Macro-Chunking</th>
  <th>Traditional RAG Chunking</th>
</tr>
<tr>
  <td><strong>Chunk Size</strong></td>
  <td>7,000-10,000 tokens</td>
  <td>100-500 tokens</td>
</tr>
<tr>
  <td><strong>Document Division</strong></td>
  <td>By logical sections when possible</td>
  <td>By fixed token count regardless of content</td>
</tr>
<tr>
  <td><strong>Context Preservation</strong></td>
  <td>Preserves paragraph and section integrity</td>
  <td>Often breaks mid-paragraph or mid-sentence</td>
</tr>
<tr>
  <td><strong>Number of Chunks</strong></td>
  <td>~10 chunks per average document</td>
  <td>Hundreds of chunks per document</td>
</tr>
<tr>
  <td><strong>Processing Approach</strong></td>
  <td>Each chunk analyzed comprehensively</td>
  <td>Chunks retrieved selectively by relevance</td>
</tr>
<tr>
  <td><strong>Context Window Usage</strong></td>
  <td>Uses most of LLM context window</td>
  <td>Uses small portion of context window</td>
</tr>
</table>

### Why Macro-Chunking Matters

The macro-chunking approach delivers several important benefits:

1. **Contextual Understanding**: Larger chunks allow the model to see patterns, relationships, and references that span multiple paragraphs
2. **Section Awareness**: Maintaining section boundaries preserves the document's organizational logic
3. **Reduced Fragmentation**: Fewer, larger chunks means less fragmentation of ideas and themes
4. **Comprehensive Processing**: Every part of the document receives full analytical attention
5. **Efficient Use of Context Windows**: Modern LLMs have context windows of 16K-100K tokens, making macro-chunking viable

## Document Processing Pipeline

Better Notes processes documents through several stages:

### 1. Document Loading and Preprocessing

- Document text is extracted and normalized
- Basic statistics are gathered (word count, paragraph count, etc.)
- Special document types (e.g., meeting transcripts) are identified
- Token counts are estimated for accurate chunking

### 2. Document Analysis

Before chunking, a quick overall analysis is performed to:
- Identify document type and purpose
- Extract high-level metadata (participants, dates, etc.)
- Determine document tone and formality
- Identify key sections and organizational structure

This information guides both chunking strategies and subsequent analysis.

### 3. Intelligent Chunking

Documents are then divided using a hierarchical approach:

```
DocumentChunker
├── Structure Detection
│   ├── Heading Recognition
│   ├── Section Boundary Identification
│   └── Natural Break Detection
├── Chunk Size Calculation
│   ├── Token Counting
│   └── Adaptive Sizing
└── Chunk Metadata Enhancement
    ├── Position Tagging (early, middle, late)
    ├── Section Context Preservation
    └── Cross-Reference Tracking
```

The chunking algorithm prefers natural document boundaries (sections, major paragraph breaks) when dividing text, falling back to paragraph boundaries when necessary.

### 4. Chunk Metadata Enhancement

Each chunk is enhanced with metadata that helps subsequent processing:

- **Position Information**: Where the chunk falls in the document (early, middle, late)
- **Section Context**: What sections are contained or partially contained
- **Chunk Type**: Whether it's structural, topical, or transitional content
- **Reference Map**: References to content in other chunks

This metadata enables chunks to be processed with awareness of their place in the larger document.

### 5. Parallel Processing

Chunks are processed in parallel with appropriate constraints:

- **Rate Limiting**: Respects API rate limits while maximizing throughput
- **Priority Processing**: Critical sections can be prioritized
- **Progressive Updates**: Results flow through the system as they become available
- **Adaptive Concurrency**: Adjusts parallel processing based on document size and complexity

## Document Type Awareness

The system recognizes different document types and adjusts processing accordingly:

### Meeting Transcripts

For meeting transcripts, the system:
- Identifies speakers and their roles
- Recognizes dialogue patterns and turn-taking
- Tracks discussion topics over time
- Identifies decisions, action assignments, and commitments

### Technical Documents

For technical documents, the system:
- Preserves specification details and requirements
- Maintains reference integrity for technical terms
- Tracks dependencies between components
- Preserves hierarchical technical relationships

### Strategic Reports

For strategic documents, the system:
- Identifies objectives, strategies, and tactics
- Preserves the relationship between goals and measures
- Maintains risk and opportunity context
- Tracks timeline and milestone information

## Chunk Position Awareness

The system processes chunks differently based on their position in the document:

- **Introduction Chunks**: Focus on identifying context, purpose, and scope
- **Body Chunks**: Emphasis on detailed content extraction and relationships
- **Conclusion Chunks**: Attention to summaries, recommendations, and next steps

This position awareness helps extract the right information from each part of the document.

## Technical Implementation

The document processing system is implemented in the `lean` folder of the codebase:

- `chunker.py`: Implements the macro-chunking algorithm
- `document.py`: Handles document loading and analysis
- `async_openai_adapter.py`: Manages API communication with rate limiting

Preprocessing is highly optimized to minimize token usage while maximizing information preservation.

## Advanced Features

### Cross-Reference Resolution

The system tracks and resolves cross-references between chunks:
- "As mentioned earlier..." references are connected to their sources
- "See section X below" references are mapped to their targets
- Pronouns with antecedents in other chunks are resolved when possible

### Handling Large Documents

For exceptionally large documents (100K+ tokens), the system implements a hierarchical approach:
- First-pass macro-chunking divides the document into major sections
- Second-pass analysis identifies the most critical sections
- Processing focuses on prioritized sections with appropriate context

### Document Structure Preservation

The system preserves structural elements that aid in analysis:
- Heading hierarchy
- List structures (bullet points, numbered lists)
- Table relationships
- Block quotes and callouts

## Future Directions

The document processing system is being enhanced with several planned improvements:

1. **Multi-Document Analysis**: Handling sets of related documents with cross-document awareness
2. **Multimedia Content**: Processing documents with embedded charts, images, and other non-text elements
3. **Document Evolution Tracking**: Analyzing changes across document versions
4. **Customizable Chunking Strategies**: User-defined chunking rules for specialized document types

## Conclusion

Better Notes' sophisticated document processing approach sets it apart from traditional document AI systems. By using macro-chunking, intelligent preprocessing, and context-aware analysis, it achieves a deeper understanding of document content and structure that enables superior analytical results.

This foundation enables the specialized agent crews to perform their analytical work with richer context, greater coherence, and more comprehensive understanding than would be possible with traditional approaches.