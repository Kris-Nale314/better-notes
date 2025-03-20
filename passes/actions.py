"""
Action items extraction module for Better Notes.
Provides specialized handling for action item extraction passes.
"""

import os
import json
import logging
import asyncio
import re
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta

from lean.async_openai_adapter import AsyncOpenAIAdapter
from passes.passes import create_pass_processor, TemplatedPass

logger = logging.getLogger(__name__)

class ActionItemExtractor:
    """
    Specialized handler for action item extraction from documents.
    Uses the passes system for core functionality but adds action-specific
    processing and formatting.
    """
    
    def __init__(self, llm_client, options=None):
        """
        Initialize the action item extractor.
        
        Args:
            llm_client: LLM client for text analysis
            options: Processing options
        """
        self.llm_client = llm_client
        self.options = options
        
        # Load configuration first to ensure it exists
        self.config = self._load_config()
        
        # Check if configurations directory exists
        config_dir = Path("passes/configurations")
        if not config_dir.exists():
            logger.warning(f"Creating missing configurations directory: {config_dir}")
            config_dir.mkdir(exist_ok=True, parents=True)
            
        # Check if action_items.json exists
        config_path = config_dir / "action_items.json"
        if not config_path.exists():
            logger.warning(f"Action items configuration not found at {config_path}")
            # If config is loaded but file doesn't exist, save it
            if self.config:
                try:
                    with open(config_path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                    logger.info(f"Created action items configuration at {config_path}")
                except Exception as e:
                    logger.error(f"Error creating configuration file: {e}")
        
        # Create the underlying pass processor
        self.processor = create_pass_processor("action_items", llm_client, options)
        
        # Fallback if processor creation fails
        if not self.processor and self.config:
            logger.warning("Falling back to direct TemplatedPass creation")
            from passes.passes import TemplatedPass
            try:
                self.processor = TemplatedPass(str(config_path), llm_client, options)
                logger.info("Successfully created pass processor using fallback method")
            except Exception as e:
                logger.error(f"Fallback processor creation failed: {e}")
                self.processor = None
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load action items configuration.
        
        Returns:
            Configuration dictionary
        """
        # Try to load from file first
        config_path = Path("passes/configurations/action_items.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    logger.info(f"Loading action items configuration from {config_path}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading action items configuration from file: {e}")
                # Fall through to default configuration
                
        # Default configuration if file not found or error loading
        logger.info("Using default action items configuration")
        return {
            "pass_type": "action_items",
            "purpose": "Extract tasks, assignments, follow-up items, and commitments from the document",
            "instructions": "Identify any tasks, assignments, commitments, or follow-up activities mentioned in the document. Extract the task, owner, deadline, and contextual details.",
            "process_by_chunks": True,
            "chunk_prompt_template": """
            You are an expert at identifying action items. Your task is to analyze the following document section and extract any tasks, commitments, follow-up items, or assignments.

            Document Section:
            {chunk_text}

            Position in document: {chunk_position}

            For each action item you identify, please provide:
            1. The specific task or action to be performed
            2. The owner or person responsible for the task (if mentioned)
            3. Any deadline or timeframe (if mentioned)
            4. A brief description or context for the action

            Look for:
            - Explicit assignments: "John will prepare the report"
            - Commitments: "I'll take care of that" 
            - Tasks with deadlines: "Complete the review by Friday"
            - Follow-up items: "We need to circle back on this"
            - Decisions requiring action: "We decided to update the policy"

            Be thorough but focus on clear action items rather than vague statements.

            Return the results in JSON format:
            ```json
            {
              "actions": [
                {
                  "task": "The specific task",
                  "owner": "The person responsible (if mentioned)",
                  "deadline": "The timeframe (if mentioned)",
                  "description": "Additional context"
                }
              ]
            }
            ```
            """
        }
    
    async def process_document(self,
                             document_text: str,
                             document_info: Optional[Dict[str, Any]] = None,
                             progress_callback: Optional[Callable] = None,
                             prior_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document for action item extraction.
        
        Args:
            document_text: Document text
            document_info: Document metadata
            progress_callback: Progress callback function
            prior_result: Prior processing results
            
        Returns:
            Action item extraction results
        """
        if not self.processor:
            logger.error("Action item processor not initialized")
            return {"error": "Action item processor not initialized"}
            
        # Process document using underlying pass processor
        if progress_callback:
            progress_callback(0.1, "Starting action item extraction...")
            
        raw_result = await self.processor.process_document(
            document_text=document_text,
            document_info=document_info,
            progress_callback=progress_callback,
            prior_result=prior_result
        )
        
        if progress_callback:
            progress_callback(0.9, "Formatting action item results...")
            
        # Extract and format results
        formatted_result = self._format_results(raw_result)
        
        if progress_callback:
            progress_callback(1.0, "Action item extraction complete")
            
        return formatted_result
    
    def _format_results(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format action item extraction results.
        
        Args:
            raw_result: Raw results from pass processor
            
        Returns:
            Formatted results
        """
        # Extract actions from result
        actions = []
        
        if "result" in raw_result:
            result_data = raw_result["result"]
            
            # Check different possible action item keys
            for key in ["actions", "action_items"]:
                if key in result_data and isinstance(result_data[key], list):
                    actions = result_data[key]
                    break
        
        # Normalize action item fields
        normalized_actions = []
        for action in actions:
            normalized = {
                "task": action.get("task", ""),
                "owner": action.get("owner", action.get("assignee", "")),
                "deadline": action.get("deadline", action.get("due_date", "")),
                "description": action.get("description", action.get("context", ""))
            }
            
            # Skip if no task
            if not normalized["task"]:
                continue
                
            normalized_actions.append(normalized)
        
        # Get priority threshold from options
        priority_threshold = None
        if self.options and hasattr(self.options, 'get_pass_options'):
            pass_options = self.options.get_pass_options("action_items")
            if "priority_threshold" in pass_options:
                priority_threshold = pass_options["priority_threshold"].lower()
        
        # Group actions by owner
        actions_by_owner = {}
        unassigned_actions = []
        
        for action in normalized_actions:
            owner = action.get("owner", "")
            if owner and owner.lower() not in ["none", "n/a", "unassigned", "unknown"]:
                if owner not in actions_by_owner:
                    actions_by_owner[owner] = []
                actions_by_owner[owner].append(action)
            else:
                unassigned_actions.append(action)
        
        # Sort actions by deadline
        all_actions = sorted(
            normalized_actions,
            key=lambda x: self._parse_deadline(x.get("deadline", ""))
        )
        
        # Generate timeline
        timeline = self._generate_timeline(normalized_actions)
        
        # Create formatted output if format_template is available
        formatted_output = ""
        if "format_template" in self.config:
            try:
                formatted_output = self._apply_format_template(
                    self.config["format_template"],
                    {
                        "actions": all_actions
                    }
                )
            except Exception as e:
                logger.error(f"Error formatting action items: {e}")
        
        # Create result with enhanced metadata
        return {
            "actions": all_actions,
            "actions_by_owner": actions_by_owner,
            "unassigned_actions": unassigned_actions,
            "timeline": timeline,
            "formatted_output": formatted_output,
            "total_actions": len(all_actions),
            "total_assigned": len(normalized_actions) - len(unassigned_actions),
            "total_unassigned": len(unassigned_actions),
            "owners": list(actions_by_owner.keys())
        }
    
    def _parse_deadline(self, deadline_str: str) -> datetime:
        """
        Parse a deadline string into a datetime for sorting.
        
        Args:
            deadline_str: String representing a deadline
            
        Returns:
            Datetime object (uses far future for no deadline)
        """
        if not deadline_str:
            # Return far future date for undefined deadlines
            return datetime.now() + timedelta(days=36500)
            
        # Common date patterns
        patterns = [
            # Specific dates
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', lambda m: datetime(
                int(m.group(3)) if len(m.group(3)) > 2 else 2000 + int(m.group(3)),
                int(m.group(2)),
                int(m.group(1))
            )),
            # Day names
            (r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', lambda m: self._next_weekday(m.group(1))),
            # Tomorrow
            (r'tomorrow', lambda m: datetime.now() + timedelta(days=1)),
            # Next week
            (r'next week', lambda m: datetime.now() + timedelta(days=7)),
            # End of week/month
            (r'end of (week|month)', lambda m: datetime.now() + timedelta(days=7 if m.group(1) == 'week' else 30)),
            # X days/weeks
            (r'(\d+) (day|week)s?', lambda m: datetime.now() + timedelta(
                days=int(m.group(1)) if m.group(2) == 'day' else int(m.group(1)) * 7
            ))
        ]
        
        # Try to match each pattern
        deadline_lower = deadline_str.lower()
        for pattern, parser in patterns:
            match = re.search(pattern, deadline_lower)
            if match:
                try:
                    return parser(match)
                except:
                    pass
        
        # Default to middle future
        return datetime.now() + timedelta(days=365)
    
    def _next_weekday(self, day_name: str) -> datetime:
        """
        Get the date of the next occurrence of a weekday.
        
        Args:
            day_name: Name of the day
            
        Returns:
            Datetime of next occurrence
        """
        days = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        target_day = days.get(day_name.lower(), 0)
        now = datetime.now()
        days_ahead = target_day - now.weekday()
        
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
            
        return now + timedelta(days=days_ahead)
    
    def _generate_timeline(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate a timeline of action items by deadline.
        
        Args:
            actions: List of action items
            
        Returns:
            Timeline dictionary
        """
        # Group actions by timeframe
        timeline = {
            "today": [],
            "this_week": [],
            "next_week": [],
            "this_month": [],
            "future": [],
            "no_deadline": []
        }
        
        now = datetime.now()
        for action in actions:
            deadline = action.get("deadline", "")
            if not deadline:
                timeline["no_deadline"].append(action)
                continue
                
            # Try to parse deadline
            try:
                date = self._parse_deadline(deadline)
                
                # Determine timeframe
                if date.date() == now.date():
                    timeline["today"].append(action)
                elif date.date() <= (now + timedelta(days=7)).date():
                    timeline["this_week"].append(action)
                elif date.date() <= (now + timedelta(days=14)).date():
                    timeline["next_week"].append(action)
                elif date.date() <= (now + timedelta(days=30)).date():
                    timeline["this_month"].append(action)
                elif date < (now + timedelta(days=36500)):  # Not the "no deadline" sentinel
                    timeline["future"].append(action)
                else:
                    timeline["no_deadline"].append(action)
            except:
                # If parsing fails, treat as no deadline
                timeline["no_deadline"].append(action)
                
        return timeline
    
    def _apply_format_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Apply a format template to data.
        
        Args:
            template: Template string
            data: Data to format
            
        Returns:
            Formatted string
        """
        # Simple template rendering implementation
        # Replace variables: {{variable}}
        import re
        
        result = template
        
        # Replace simple variables
        for variable in re.findall(r'{{([^{]+?)}}', result):
            var_name = variable.strip()
            # Handle nested paths with dot notation
            parts = var_name.split('.')
            
            # Get value from data
            value = data
            try:
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = ""
                        break
                # Replace in template
                result = result.replace('{{' + variable + '}}', str(value))
            except Exception:
                # Replace with empty string if error
                result = result.replace('{{' + variable + '}}', "")
        
        # Handle for loops: {% for item in items %} ... {% endfor %}
        for loop_match in re.finditer(
            r'{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%}(.*?){%\s*endfor\s*%}',
            result,
            re.DOTALL
        ):
            loop_var, collection_name, loop_content = loop_match.groups()
            
            # Get the collection
            if collection_name in data and isinstance(data[collection_name], list):
                collection = data[collection_name]
                
                # Process loop content for each item
                rendered_items = []
                for item in collection:
                    item_content = loop_content
                    
                    # Replace item variables
                    for var_match in re.finditer(r'{{' + loop_var + r'\.(\w+)}}', item_content):
                        item_prop = var_match.group(1)
                        if item_prop in item:
                            item_content = item_content.replace(
                                '{{' + loop_var + '.' + item_prop + '}}',
                                str(item[item_prop])
                            )
                    
                    # Handle conditionals within the loop
                    for cond_match in re.finditer(
                        r'{%\s*if\s+' + loop_var + r'\.(\w+)\s*%}(.*?){%\s*endif\s*%}',
                        item_content,
                        re.DOTALL
                    ):
                        cond_prop, cond_content = cond_match.groups()
                        if cond_prop in item and item[cond_prop]:
                            # Condition is true, keep content
                            item_content = item_content.replace(
                                '{% if ' + loop_var + '.' + cond_prop + ' %}' + cond_content + '{% endif %}',
                                cond_content
                            )
                        else:
                            # Condition is false, remove content
                            item_content = item_content.replace(
                                '{% if ' + loop_var + '.' + cond_prop + ' %}' + cond_content + '{% endif %}',
                                ''
                            )
                    
                    rendered_items.append(item_content)
                
                # Replace the entire loop with rendered items
                loop_replacement = ''.join(rendered_items)
                result = result.replace(
                    '{% for ' + loop_var + ' in ' + collection_name + ' %}' + loop_content + '{% endfor %}',
                    loop_replacement
                )
            else:
                # Collection not found, remove the loop
                result = result.replace(
                    '{% for ' + loop_var + ' in ' + collection_name + ' %}' + loop_content + '{% endfor %}',
                    ''
                )
        
        return result
    
    def process_document_sync(self,
                           document_text: str,
                           document_info: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[Callable] = None,
                           prior_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document synchronously.
        
        Args:
            document_text: Document text
            document_info: Document metadata
            progress_callback: Progress callback function
            prior_result: Prior processing results
            
        Returns:
            Action item extraction results
        """
        # Create event loop
        loop = asyncio.new_event_loop()
        try:
            # Run async function
            return loop.run_until_complete(
                self.process_document(
                    document_text=document_text,
                    document_info=document_info,
                    progress_callback=progress_callback,
                    prior_result=prior_result
                )
            )
        finally:
            # Clean up
            loop.close()


def create_action_extractor(llm_client, options=None) -> ActionItemExtractor:
    """
    Create an action item extractor instance.
    
    Args:
        llm_client: LLM client for text analysis
        options: Processing options
        
    Returns:
        Action item extractor instance
    """
    return ActionItemExtractor(llm_client, options)