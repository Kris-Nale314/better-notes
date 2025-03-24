"""
Issues Test Script - Debug tool for Better Notes.
Provides enhanced logging and step-by-step execution to identify issues
in the document processing pipeline.
"""

import os
import sys
import time
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("debug_log.txt", mode="w")
    ]
)

logger = logging.getLogger("issues_test")
logger.info("Starting Issues Test Script")

# Attempt to import key components with detailed error reporting
logger.info("Attempting to import core components")

def import_with_logging(module_name, class_name=None):
    """Import a module or class with detailed logging."""
    try:
        logger.info(f"Importing {module_name}{f'.{class_name}' if class_name else ''}")
        if class_name:
            module = __import__(module_name, fromlist=[class_name])
            component = getattr(module, class_name)
            logger.info(f"Successfully imported {module_name}.{class_name}")
            return component
        else:
            module = __import__(module_name, fromlist=["*"])
            logger.info(f"Successfully imported {module_name}")
            return module
    except ImportError as e:
        logger.error(f"Failed to import {module_name}{f'.{class_name}' if class_name else ''}: {e}")
        logger.error(f"Current sys.path: {sys.path}")
        return None
    except AttributeError as e:
        logger.error(f"Module {module_name} doesn't have attribute {class_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing {module_name}{f'.{class_name}' if class_name else ''}: {e}")
        logger.error(traceback.format_exc())
        return None

# Try to import core components
ConfigManager = import_with_logging("config_manager", "ConfigManager")
ProcessingOptions = import_with_logging("config_manager", "ProcessingOptions")
UniversalLLMAdapter = import_with_logging("universal_llm_adapter", "UniversalLLMAdapter")
OrchestratorFactory = import_with_logging("orchestrator_factory", "OrchestratorFactory")

# Check if imports were successful
if not all([ConfigManager, ProcessingOptions, UniversalLLMAdapter, OrchestratorFactory]):
    logger.critical("Failed to import one or more required components. Exiting.")
    sys.exit(1)

# Try to import document processor components
try:
    DocumentAnalyzer = import_with_logging("lean.document", "DocumentAnalyzer")
    if not DocumentAnalyzer:
        DocumentAnalyzer = import_with_logging("document", "DocumentAnalyzer")
except:
    logger.error("Could not import DocumentAnalyzer from any location")
    DocumentAnalyzer = None

try:
    DocumentChunker = import_with_logging("lean.chunker", "DocumentChunker")
    if not DocumentChunker:
        DocumentChunker = import_with_logging("chunker", "DocumentChunker")
except:
    logger.error("Could not import DocumentChunker from any location")
    DocumentChunker = None

# Determine test document to use
SAMPLE_TEXT = """
This is a test document for Better Notes.

# Issues Analysis

The current implementation has several issues:

## Critical Issues
1. The DocumentAnalyzer may not be loading correctly
2. The ConfigManager error with field() method needs to be addressed

## High-Priority Issues
1. The Orchestrator has complex import paths that can fail
2. Error handling is silencing important issues
3. The result_formatting.py needs better type handling

## Medium-Priority Issues
1. Too many fallback paths make debugging difficult
2. Streamlit UI needs to handle different result formats better

Let's fix these issues to make Better Notes more reliable.
"""

def test_config_manager():
    """Test ConfigManager functionality."""
    logger.info("Testing ConfigManager")
    
    try:
        # Create ConfigManager
        config_manager = ConfigManager()
        logger.info(f"ConfigManager initialized with base_dir: {config_manager.base_dir}")
        
        # List config directory contents
        config_dir = os.path.join(config_manager.base_dir, "config")
        if os.path.exists(config_dir):
            logger.info(f"Config directory contents: {os.listdir(config_dir)}")
        else:
            logger.warning(f"Config directory not found: {config_dir}")
        
        # Try loading issues config
        issues_config = config_manager.get_config("issues")
        if issues_config:
            logger.info(f"Successfully loaded issues config: {list(issues_config.keys())}")
        else:
            logger.warning("Failed to load issues config")
        
        # Try processing options
        options = config_manager.get_processing_options()
        logger.info(f"Default processing options: {options}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing ConfigManager: {e}")
        logger.error(traceback.format_exc())
        return False

def test_document_analyzer(text=SAMPLE_TEXT):
    """Test DocumentAnalyzer functionality."""
    logger.info("Testing DocumentAnalyzer")
    
    if DocumentAnalyzer is None:
        logger.error("DocumentAnalyzer could not be imported")
        return False
    
    try:
        # Create LLM client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
        
        llm_client = UniversalLLMAdapter(api_key=api_key)
        
        # Create analyzer
        analyzer = DocumentAnalyzer(llm_client)
        logger.info("DocumentAnalyzer created successfully")
        
        # Analyze preview
        import asyncio
        result = asyncio.run(analyzer.analyze_preview(text))
        logger.info(f"Analysis result: {result}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing DocumentAnalyzer: {e}")
        logger.error(traceback.format_exc())
        return False

def test_document_chunker(text=SAMPLE_TEXT):
    """Test DocumentChunker functionality."""
    logger.info("Testing DocumentChunker")
    
    if DocumentChunker is None:
        logger.error("DocumentChunker could not be imported")
        return False
    
    try:
        # Create chunker
        chunker = DocumentChunker()
        logger.info("DocumentChunker created successfully")
        
        # Chunk document
        chunks = chunker.chunk_document(text, min_chunks=3)
        logger.info(f"Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i}: {len(chunk['text'])} chars, position: {chunk.get('position')}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing DocumentChunker: {e}")
        logger.error(traceback.format_exc())
        return False

def test_orchestrator(text=SAMPLE_TEXT):
    """Test Orchestrator functionality."""
    logger.info("Testing Orchestrator")
    
    try:
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
        
        # Create config manager
        config_manager = ConfigManager()
        
        # Create orchestrator
        orchestrator = OrchestratorFactory.create_orchestrator(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.2,
            verbose=True,
            config_manager=config_manager
        )
        
        logger.info("Orchestrator created successfully")
        
        # Define progress callback
        def progress_callback(progress, message):
            logger.info(f"Progress: {progress:.2f} - {message}")
        
        # Process document
        options = {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.2,
            "crews": ["issues"],
            "min_chunks": 3,
            "max_chunk_size": 2000,
            "enable_reviewer": True,
            "detail_level": "standard"
        }
        
        logger.info("Starting document processing")
        result = orchestrator.process_document(
            text,
            options=options,
            progress_callback=progress_callback
        )
        
        # Log result summary
        if isinstance(result, dict):
            if "error" in result:
                logger.error(f"Processing error: {result['error']}")
                return False
            
            logger.info(f"Processing successful, result keys: {list(result.keys())}")
            
            # Check for issues result
            if "issues" in result:
                issues_result = result["issues"]
                if isinstance(issues_result, dict):
                    logger.info(f"Issues result keys: {list(issues_result.keys())}")
                else:
                    logger.info(f"Issues result type: {type(issues_result)}")
        else:
            logger.warning(f"Unexpected result type: {type(result)}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Orchestrator: {e}")
        logger.error(traceback.format_exc())
        return False

def test_component_tree():
    """Analyze and log the component tree to identify import issues."""
    logger.info("Analyzing component tree")
    
    # List of components to check
    components = [
        {"module": "universal_llm_adapter", "class": "UniversalLLMAdapter"},
        {"module": "config_manager", "class": "ConfigManager"},
        {"module": "orchestrator", "class": "Orchestrator"},
        {"module": "orchestrator_factory", "class": "OrchestratorFactory"},
        {"module": "lean.document", "class": "DocumentAnalyzer", "alternate_module": "document"},
        {"module": "lean.chunker", "class": "DocumentChunker", "alternate_module": "chunker"},
        {"module": "agents.planner", "class": "PlannerAgent", "alternate_module": "planner"},
        {"module": "crews.issues_crew", "class": "IssuesCrew", "alternate_module": "issues_crew"},
    ]
    
    # Check each component
    for comp in components:
        module_name = comp["module"]
        class_name = comp["class"]
        
        try:
            module = __import__(module_name, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            logger.info(f"✅ Found {module_name}.{class_name}")
            
            # If this is a class, inspect its methods
            if hasattr(class_obj, "__dict__"):
                methods = [m for m in dir(class_obj) if not m.startswith("_") and callable(getattr(class_obj, m))]
                logger.info(f"   Methods: {', '.join(methods[:5])}{'...' if len(methods) > 5 else ''}")
        except ImportError:
            # Try alternate module if available
            if "alternate_module" in comp:
                try:
                    alt_module = comp["alternate_module"]
                    module = __import__(alt_module, fromlist=[class_name])
                    class_obj = getattr(module, class_name)
                    logger.info(f"✅ Found {alt_module}.{class_name} (alternate)")
                except Exception as e:
                    logger.warning(f"❌ Not found: {module_name}.{class_name} or {alt_module}.{class_name}")
            else:
                logger.warning(f"❌ Not found: {module_name}.{class_name}")
        except Exception as e:
            logger.warning(f"❌ Error checking {module_name}.{class_name}: {e}")
    
    # Print Python path
    logger.info(f"Python path: {sys.path}")
    
    # List files in current directory
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    # Check for common directories
    for directory in ["lean", "agents", "crews", "config"]:
        if os.path.exists(directory):
            logger.info(f"Directory {directory} exists, contents: {os.listdir(directory)}")
        else:
            logger.warning(f"Directory {directory} does not exist")

def main():
    """Run all tests and collect results."""
    logger.info("=" * 50)
    logger.info("BETTER NOTES COMPONENT TEST")
    logger.info("=" * 50)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info("=" * 50)
    
    # Analyze component tree first
    logger.info("COMPONENT TREE ANALYSIS")
    test_component_tree()
    logger.info("=" * 50)
    
    # Run individual component tests
    results = {
        "ConfigManager": test_config_manager(),
        "DocumentAnalyzer": test_document_analyzer(),
        "DocumentChunker": test_document_chunker(),
        "Orchestrator": test_orchestrator()
    }
    
    # Print summary
    logger.info("=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {component}")
    logger.info("=" * 50)
    
    # Give recommendations based on results
    logger.info("RECOMMENDATIONS:")
    
    if not results["ConfigManager"]:
        logger.info("- Fix ConfigManager: Check the fields definition in the ProcessingOptions class")
    
    if not results["DocumentAnalyzer"]:
        logger.info("- Fix DocumentAnalyzer: Check that the module is in your Python path")
        logger.info("  Possible fix: Make sure 'lean' directory is in your Python path")
    
    if not results["DocumentChunker"]:
        logger.info("- Fix DocumentChunker: Check that the module is in your Python path")
        logger.info("  Possible fix: Make sure 'lean' directory is in your Python path")
    
    if not results["Orchestrator"]:
        if not (results["DocumentAnalyzer"] and results["DocumentChunker"]):
            logger.info("- Fix Orchestrator: First fix DocumentAnalyzer and DocumentChunker")
        else:
            logger.info("- Fix Orchestrator: Check for initialization errors in the pipeline")
    
    # If all tests passed
    if all(results.values()):
        logger.info("All components are working correctly!")
    
    logger.info("=" * 50)
    logger.info("See debug_log.txt for detailed debugging information")
    
    return results

if __name__ == "__main__":
    main()