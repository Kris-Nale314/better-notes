# lean/orchestrator.py
"""
Orchestrator for document processing pipeline.
Manages the flow from document input to summary output.
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union

from .async_openai_adapter import AsyncOpenAIAdapter
from .document import DocumentAnalyzer
from .chunker import DocumentChunker
from .summarizer import ChunkSummarizer
from .synthesizer import Synthesizer
from .booster import Booster
from .options import ProcessingOptions

logger = logging.getLogger(__name__)

class SummarizerFactory:
    """Factory for creating the summarization pipeline components."""
    
    @staticmethod
    def create_pipeline(api_key: str, options: ProcessingOptions) -> Dict[str, Any]:
        """
        Create the summarization pipeline based on the provided options.
        
        Args:
            api_key: OpenAI API key
            options: Processing options
            
        Returns:
            Dictionary containing pipeline components
        """
        # Create the OpenAI adapter
        llm_client = AsyncOpenAIAdapter(
            model=options.model_name,
            api_key=api_key,
            temperature=options.temperature
        )
        
        # Create core components
        analyzer = DocumentAnalyzer(llm_client)
        chunker = DocumentChunker()
        summarizer = ChunkSummarizer(llm_client)
        synthesizer = Synthesizer(llm_client)
        
        # Create booster if needed
        if options.max_concurrent_chunks > 1:
            booster = Booster(
                cache_dir=".cache",
                max_workers=options.max_concurrent_chunks,
                enable_caching=getattr(options, "enable_caching", False)
            )
        else:
            booster = None
        
        # Create orchestrator
        orchestrator = Orchestrator(
            llm_client=llm_client,
            analyzer=analyzer,
            chunker=chunker,
            summarizer=summarizer,
            synthesizer=synthesizer,
            booster=booster,
            options=options
        )
        
        # Return all components in a dictionary
        return {
            'llm_client': llm_client,
            'analyzer': analyzer,
            'chunker': chunker,
            'summarizer': summarizer,
            'synthesizer': synthesizer,
            'booster': booster,
            'orchestrator': orchestrator
        }


class Orchestrator:
    """
    Orchestrates the document processing workflow.
    Coordinates the analyzer, chunker, summarizer, and synthesizer components.
    """
    
    def __init__(self, 
                 llm_client: AsyncOpenAIAdapter,
                 analyzer: DocumentAnalyzer,
                 chunker: DocumentChunker,
                 summarizer: ChunkSummarizer,
                 synthesizer: Synthesizer,
                 booster: Optional[Booster],
                 options: ProcessingOptions):
        """
        Initialize the orchestrator with pipeline components.
        
        Args:
            llm_client: OpenAI client for LLM operations
            analyzer: Document analyzer for initial analysis
            chunker: Document chunker for dividing text
            summarizer: Chunk summarizer for processing chunks
            synthesizer: Synthesizer for combining chunk summaries
            booster: Optional booster for parallel processing
            options: Processing options
        """
        self.llm_client = llm_client
        self.analyzer = analyzer
        self.chunker = chunker
        self.summarizer = summarizer
        self.synthesizer = synthesizer
        self.booster = booster
        self.options = options
        
        # Initialize refiner if available
        try:
            from ui_utils.refiner import SummaryRefiner
            self.refiner = SummaryRefiner(llm_client)
            self.has_refiner = True
        except ImportError:
            self.has_refiner = False
            logger.info("SummaryRefiner not available, refinement features disabled")
    
    async def process_document(self, 
                              document_text: str,
                              progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Process a document asynchronously through the pipeline.
        
        Args:
            document_text: The document text to process
            progress_callback: Optional callback for progress updates
                Function signature: callback(progress_fraction, status_message)
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # --- Step 1: Document Analysis ---
            self._update_progress(progress_callback, 0.1, "Analyzing document...")
            
            document_info = await self.analyzer.analyze_preview(
                document_text[:self.options.preview_length],
                preview_length=self.options.preview_length
            )
            document_info['original_text_length'] = len(document_text)
            document_info['user_instructions'] = self.options.user_instructions
            
            self._update_progress(progress_callback, 0.2, "Document analysis complete")
            
            # --- Step 2: Document Chunking ---
            self._update_progress(progress_callback, 0.2, "Chunking document...")
            
            chunks = self.chunker.chunk_document(
                document_text, 
                min_chunks=self.options.min_chunks,
                max_chunk_size=self.options.max_chunk_size
            )
            document_info['total_chunks'] = len(chunks)
            
            self._update_progress(
                progress_callback, 
                0.3, 
                f"Document chunked into {len(chunks)} chunks"
            )
            
            # --- Step 3: Chunk Summarization ---
            self._update_progress(progress_callback, 0.3, "Summarizing chunks...")
            
            # Process chunks in parallel if booster is available
            if self.booster and len(chunks) > 1:
                chunk_summaries = await self.booster.process_in_parallel(
                    chunks,
                    self._summarize_chunk_async,
                    max_concurrency=self.options.max_concurrent_chunks
                )
            else:
                # Process chunks sequentially
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    summary = await self._summarize_chunk_async(chunk)
                    chunk_summaries.append(summary)
                    
                    # Update progress for each chunk
                    chunk_progress = 0.3 + (0.4 * (i + 1) / len(chunks))
                    self._update_progress(
                        progress_callback,
                        chunk_progress,
                        f"Summarized chunk {i+1}/{len(chunks)}"
                    )
            
            # Ensure chunks are in the correct order
            chunk_summaries.sort(key=lambda x: x.get('chunk_index', 0))
            
            self._update_progress(progress_callback, 0.7, "Chunk summarization complete")
            
            # --- Step 4: Summary Synthesis ---
            self._update_progress(progress_callback, 0.7, "Creating final summary...")
            
            synthesis_result = await self.synthesizer.synthesize_summaries(
                chunk_summaries,
                document_info,
                self.options.detail_level
            )
            
            self._update_progress(progress_callback, 0.9, "Summary synthesis complete")
            
            # --- Step 5: Final Result Preparation ---
            self._update_progress(progress_callback, 0.9, "Preparing final result...")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create the final result dictionary
            result = {
                'summary': synthesis_result.get('summary', ''),
                'metadata': {
                    'model': self.options.model_name,
                    'processing_time_seconds': processing_time,
                    'chunks_processed': len(chunks),
                    'detail_level': self.options.detail_level
                },
                'document_info': document_info
            }
            
            # Add executive summary if available
            if 'executive_summary' in synthesis_result:
                result['executive_summary'] = synthesis_result['executive_summary']
            
            # Add chunk summaries if requested
            if self.options.include_metadata:
                result['chunk_summaries'] = chunk_summaries
            
            self._update_progress(progress_callback, 1.0, "Processing complete")
            return result
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}", exc_info=True)
            return {
                'error': str(e),
                'metadata': {
                    'success': False,
                    'processing_time_seconds': time.time() - start_time
                }
            }
    
    async def _summarize_chunk_async(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize a single chunk asynchronously.
        
        Args:
            chunk: The chunk to summarize
            
        Returns:
            Summarized chunk result
        """
        try:
            # Ensure document_info is passed to summarize_chunk
            document_info = {}
            return await self.summarizer.summarize_chunk(chunk, document_info)
        except Exception as e:
            logger.error(f"Error summarizing chunk: {e}", exc_info=True)
            return {
                'chunk_index': chunk.get('index', 0),
                'error': str(e),
                'summary': f"Error processing this section: {str(e)}"
            }
    
    def _update_progress(self, 
                       callback: Optional[Callable[[float, str], None]],
                       progress: float,
                       message: str) -> None:
        """
        Update progress through callback if provided.
        
        Args:
            callback: Progress callback function
            progress: Progress as a value between 0.0 and 1.0
            message: Status message
        """
        if callback:
            try:
                callback(progress, message)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")
        
        # Log progress
        logger.info(f"Progress {progress:.0%}: {message}")
    
    def process_document_sync(self, 
                            document_text: str,
                            progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Process a document synchronously.
        This is a wrapper around the async method for simpler interfaces.
        
        Args:
            document_text: The document text to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results
        """
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async method in the event loop
            result = loop.run_until_complete(
                self.process_document(document_text, progress_callback)
            )
            return result
        except Exception as e:
            logger.error(f"Error in synchronous document processing: {e}", exc_info=True)
            return {
                'error': str(e),
                'metadata': {
                    'success': False
                }
            }
        finally:
            # Always close the loop to free resources
            loop.close()
    
    async def refine_summary(self, 
                           result: Dict[str, Any],
                           refinement_type: str,
                           custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Refine a summary result.
        
        Args:
            result: The summary result to refine
            refinement_type: Type of refinement ('more_detail', 'more_concise', etc.)
            custom_instructions: Optional custom instructions for refinement
            
        Returns:
            Refined summary result
        """
        if not self.has_refiner:
            logger.warning("Summary refinement requested but refiner is not available")
            return result
        
        try:
            refined_result = await self.refiner.refine_summary(
                result,
                refinement_type,
                custom_instructions
            )
            return refined_result
        except Exception as e:
            logger.error(f"Error refining summary: {e}", exc_info=True)
            return result  # Return original result if refinement fails
    
    def refine_summary_sync(self, 
                         result: Dict[str, Any],
                         refinement_type: str,
                         custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Refine a summary result synchronously.
        
        Args:
            result: The summary result to refine
            refinement_type: Type of refinement
            custom_instructions: Optional custom instructions
            
        Returns:
            Refined summary result
        """
        if not self.has_refiner:
            logger.warning("Summary refinement requested but refiner is not available")
            return result
        
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async method in the event loop
            refined_result = loop.run_until_complete(
                self.refine_summary(result, refinement_type, custom_instructions)
            )
            return refined_result
        except Exception as e:
            logger.error(f"Error in synchronous summary refinement: {e}", exc_info=True)
            return result  # Return original result if refinement fails
        finally:
            # Always close the loop to free resources
            loop.close()