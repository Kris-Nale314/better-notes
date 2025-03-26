# 🔧 Better Notes: Configuration Guide

<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logo.svg" alt="Better-Notes logo" width="120px"/>
  <h3>Customizing Your Agent Crews!</h3>
  <p>Make Better Notes truly yours by teaching the agents to think your way</p>
</div>

Better Notes uses a flexible configuration system that lets you customize how your AI agent teams analyze documents. Think of it as training your own specialized analyst team! This guide shows you how to remix existing configurations and create new ones without touching any code.

## 📋 Table of Contents
- [🧩 Understanding Configuration Files](#-understanding-configuration-files)
- [🔥 Modifying Issue Severity Levels](#-modifying-issue-severity-levels)
- [🏷️ Customizing Categories](#️-customizing-categories)
- [🚀 Creating a New Crew Type](#-creating-a-new-crew-type)
- [⚡ Advanced Agent Configuration](#-advanced-agent-configuration)

## 🧩 Understanding Configuration Files

Better Notes uses JSON configuration files (think of them as instruction manuals for your agent teams). These files live in the `config` directory with names like `issues_config.json`.

Each configuration file defines:
- Core crew attributes 🧠
- Analysis definitions (severity levels, categories) 📊
- Workflow stages and agent roles 🔄
- User options (detail levels, focus areas) 🎛️
- Report formatting instructions 📝

Let's dive in and see how we can customize these!

## 🔥 Modifying Issue Severity Levels

Want your own custom severity scale? Whether you need corporate-friendly terms or industry-specific severity ratings, it's easy to change!

### 🌟 Default Severity Levels

```json
"severity_levels": {
  "critical": "Immediate threat requiring urgent attention",
  "high": "Significant impact requiring prompt attention",
  "medium": "Moderate impact that should be addressed",
  "low": "Minor impact with limited consequences"
}
```

### 💥 Example: Adding a "Blocker" Level

Need something even more urgent than critical? Add a "blocker" level:

```json
"severity_levels": {
  "blocker": "🚫 Existential threat that blocks all progress",
  "critical": "🔴 Immediate threat requiring urgent attention",
  "high": "🟠 Significant impact requiring prompt attention",
  "medium": "🟡 Moderate impact that should be addressed",
  "low": "🟢 Minor impact with limited consequences"
}
```

### 💰 Example: Using Impact-Based Severity

Working in finance? Try impact-based severity levels:

```json
"severity_levels": {
  "critical": "💸 Financial impact > $1M or regulatory consequences",
  "high": "💰 Financial impact $250K-$1M or major delay",
  "medium": "💵 Financial impact $50K-$250K or moderate delay",
  "low": "💲 Financial impact < $50K or minor inconvenience"
}
```

> 💡 **Pro Tip:** Remember to update the `report_format.sections` array too!
> 
> ```json
> "sections": [
>   "Executive Summary",
>   "🚫 Blocker Issues",  
>   "🔴 Critical Issues",
>   "🟠 High-Priority Issues", 
>   "🟡 Medium-Priority Issues",
>   "🟢 Low-Priority Issues"
> ]
> ```

## 🏷️ Customizing Categories

Categories help organize issues by their domain or type. Make them match your team's language!

### 🌈 Default Categories

```json
"categories": [
  "technical", "process", "resource", "quality", "risk", "compliance"
]
```

### 💻 Example: Software Development Categories

Building software? Try these categories:

```json
"categories": [
  "🔒 security", "⚡ performance", "😊 usability", "🔧 maintainability", 
  "♿ accessibility", "🌐 compatibility", "📈 scalability", "🏆 reliability"
]
```

### 💼 Example: Business Analysis Categories

For business documents, you might prefer:

```json
"categories": [
  "🎯 strategic", "⚙️ operational", "💰 financial", "⚖️ legal", 
  "🏪 market", "👥 organizational", "🥊 competitive", "🛍️ customer"
]
```

<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logic.svg" alt="Better-Notes Logic" width="70%"/>
  <p><em>Each agent in the crew will use your custom categories</em></p>
</div>

## 🚀 Creating a New Crew Type

Why stop at finding issues? Create entirely new crews for different kinds of analysis!

### 🔍 Steps to Create a New Crew Type

1. Create a new configuration file (e.g., `opportunities_config.json`) 📄
2. Define the core structure for the new crew type 🏗️
3. Implement a new crew class (optional for advanced customization) 🧪

### ✨ Example: Opportunities Crew Configuration

Here's a complete example for an "Opportunities Crew" that identifies potential improvements:

```json
{
  "crew_type": "opportunities",
  "description": "Identifies potential improvements, innovations, and growth areas in documents",
  
  "opportunity_definition": {
    "description": "Any potential improvement, innovation, or growth area that could enhance value, efficiency, or quality",
    "value_levels": {
      "transformative": "🌟 Could fundamentally transform business outcomes",
      "significant": "⭐ Could deliver substantial improvements to key metrics",
      "moderate": "✨ Could provide meaningful improvements to specific areas",
      "minor": "💫 Could offer incremental improvements or quick wins"
    },
    "categories": [
      "💰 revenue", "📉 cost", "😊 experience", "⚡ efficiency", 
      "💡 innovation", "🤝 partnership", "🧠 talent"
    ]
  },
  
  "workflow": {
    "enabled_stages": ["document_analysis", "chunking", "planning", 
                      "extraction", "aggregation", "evaluation", 
                      "formatting", "review"],
    "agent_roles": {
      "planner": {
        "description": "Plans the analysis approach for identifying opportunities",
        "primary_task": "Create tailored instructions for each agent based on document type and user preferences"
      },
      "extractor": {
        "description": "Identifies opportunities from document chunks",
        "primary_task": "Find all potential opportunities, assign initial value assessment, and provide relevant context",
        "output_schema": {
          "title": "Concise opportunity label",
          "description": "Detailed explanation of the opportunity",
          "value_level": "Initial value assessment (transformative/significant/moderate/minor)",
          "category": "Opportunity category from the defined list",
          "context": "Relevant information from the document",
          "effort": "Estimated implementation effort (high/medium/low)"
        }
      },
      "aggregator": {
        "description": "Combines and enhances opportunities from all chunks",
        "primary_task": "Consolidate similar opportunities while preserving important variations"
      },
      "evaluator": {
        "description": "Assesses opportunity value and priority",
        "primary_task": "Analyze each opportunity's potential impact, feasibility, and assign final value rating"
      },
      "formatter": {
        "description": "Creates the structured report",
        "primary_task": "Organize opportunities by value and category into a clear report"
      },
      "reviewer": {
        "description": "Ensures quality and alignment with user needs",
        "primary_task": "Verify report quality and alignment with user preferences"
      }
    }
  },
  
  "user_options": {
    "detail_levels": {
      "essential": "Focus only on the highest-value opportunities",
      "standard": "Balanced analysis of worthwhile opportunities",
      "comprehensive": "In-depth analysis of all potential opportunities"
    },
    "focus_areas": {
      "revenue": "💲 New revenue streams, pricing, customer acquisition",
      "cost": "💰 Cost reduction, efficiency improvements, waste elimination",
      "experience": "😊 Customer experience, employee experience, usability",
      "efficiency": "⚡ Process improvements, automation, optimization",
      "innovation": "💡 New products, services, business models",
      "partnership": "🤝 Strategic alliances, integration, ecosystem",
      "talent": "🧠 Skills development, recruiting, retention, culture"
    }
  },
  
  "report_format": {
    "sections": [
      "Executive Summary",
      "🌟 Transformative Opportunities",
      "⭐ Significant Opportunities", 
      "✨ Moderate Opportunities",
      "💫 Minor Opportunities"
    ],
    "opportunity_presentation": {
      "title": "Clear, descriptive title",
      "value_level": "Visual indicator of value",
      "description": "Full opportunity description",
      "potential_impact": "Potential benefits and outcomes",
      "category": "Opportunity category",
      "effort": "Implementation effort estimate"
    }
  }
}
```

> 💡 **Quick Start:** Copy, modify, save as `opportunities_config.json`, and reload the app!

## ⚡ Advanced Agent Configuration

Each agent in your crew can be customized with specific instructions and behaviors!

### 🛠️ Modifying Agent Instructions

Want an agent to focus on something specific? Change its `description` and `primary_task`:

```json
"extractor": {
  "description": "🔒 Identifies security vulnerabilities in code snippets",
  "primary_task": "Find all potential security issues, assign CVSS scores, and provide remediation suggestions",
  "output_schema": {
    "title": "Vulnerability name",
    "cvss_score": "CVSS score from 0.0-10.0",
    "description": "Detailed explanation of the vulnerability",
    "affected_components": "List of affected components",
    "remediation": "Steps to fix the vulnerability"
  }
}
```

### 🧪 Adding Custom Agent Behavior

For power users, you can get even more creative:

1. Define custom fields in the agent configuration 📝
2. Create a specialized agent class that extends `BaseAgent` 🧬
3. Register the agent class in your crew's `_init_agent_factory` method 🔌

Example of adding emphasis fields to agent instructions:

```json
"aggregator": {
  "description": "Combines and deduplicates issues from all chunks",
  "primary_task": "Consolidate similar issues while preserving important distinctions",
  "emphasis_fields": [
    "security_focus", "compliance_requirements", "performance_thresholds"
  ]
}
```

<table>
<tr>
  <td width="50%" valign="top">
    <h3>🤓 For Coding Enthusiasts</h3>
    <p>Want to go even deeper? You can create your own agent classes that inherit from BaseAgent!</p>
    
```python
class SecurityExtractorAgent(BaseAgent):
    """Specialized agent for security vulnerabilities."""
    
    async def process(self, context):
        # Custom security extraction logic
        return security_findings
```
  </td>
  <td width="50%" valign="top">
    <h3>🎮 Try It Out</h3>
    <p>Test your custom configurations with different document types:</p>
    <ul>
      <li>🔒 Security audit reports</li>
      <li>📊 Financial statements</li>
      <li>📝 Product requirements</li>
      <li>📅 Project plans</li>
      <li>🔬 Research papers</li>
    </ul>
    <p>See how your agents adapt to different content!</p>
  </td>
</tr>
</table>

## 🎉 Make It Your Own!

Better Notes' configuration system lets you tailor almost every aspect of how agents analyze documents. It's like having your own AI analysis team that you can train to think exactly how you want!

By tweaking these JSON files, you can:
- 🎯 Target specific types of analysis
- 🔤 Use your organization's terminology
- 🏢 Align with your company's priorities
- 🧩 Create entirely new analysis approaches

**Have fun experimenting!** The beauty of Better Notes is that you can try different configurations easily without writing a single line of code. Let your agents learn to see documents through your eyes!

<div align="center">
  <p><em>Better Notes: Where AI agents work like you think</em></p>
</div>