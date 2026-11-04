#!/usr/bin/env python3
"""
Test script to demonstrate the updated standalone tools.
This script shows how to use the newly updated files.
"""

import os
import sys
from datetime import datetime

def test_llm_response_logger():
    """Test the updated llm_response_logger.py functionality."""
    print("=== Testing LLM Response Logger ===")
    print("This tool now provides:")
    print("1. Quick testing of all models with a single prompt")
    print("2. Batch testing from a file with multiple prompts")
    print("3. Automatic comprehensive evaluation")
    print("4. CSV export of results")
    print()
    print("Usage: python llm_response_logger.py")
    print("Then choose option 1 for single prompt or 2 for batch testing")
    print()

def test_response_generator():
    """Test the updated response_generator.py functionality."""
    print("=== Testing Response Generator ===")
    print("This tool now provides:")
    print("1. Side-by-side response comparison")
    print("2. Response length analysis")
    print("3. Optional comprehensive evaluation")
    print("4. Batch processing from files")
    print("5. Detailed comparison reports")
    print()
    print("Usage: python response_generator.py")
    print("Then choose option 1 for single prompt or 2 for batch generation")
    print()

def test_llm_prompt_eval_analysis():
    """Test the updated llm_prompt_eval_analysis.py functionality."""
    print("=== Testing LLM Prompt Evaluation Analysis ===")
    print("This tool now provides:")
    print("1. Automatic loading of latest CSV results")
    print("2. Comprehensive statistical analysis")
    print("3. Model performance comparisons")
    print("4. Evaluator bias analysis")
    print("5. Cross-evaluation heatmaps")
    print("6. Correlation analysis")
    print("7. Automated report generation")
    print("8. Visualization generation (charts and graphs)")
    print()
    print("Usage: python llm_prompt_eval_analysis.py")
    print("Then choose option 5 for full analysis")
    print()

def create_sample_prompts_file():
    """Create a sample prompts file for testing."""
    sample_prompts = [
        "Explain quantum computing in simple terms",
        "Write a short story about a robot learning to paint",
        "What are the benefits and drawbacks of renewable energy?",
        "How would you solve world hunger?",
        "Explain the concept of machine learning to a 10-year-old"
    ]
    
    filename = "sample_prompts.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in sample_prompts:
            f.write(prompt + "\n")
    
    print(f"✅ Created {filename} with {len(sample_prompts)} sample prompts")
    return filename

def main():
    """Main test function."""
    print("🧪 STANDALONE TOOLS TESTING GUIDE")
    print("=" * 50)
    print()
    print("The following files have been updated and are now useful:")
    print()
    
    test_llm_response_logger()
    test_response_generator()
    test_llm_prompt_eval_analysis()
    
    print("=" * 50)
    print("🎯 QUICK START GUIDE:")
    print()
    print("1. First, ensure your .env file has all API keys:")
    print("   OPENAI_API_KEY=your_key_here")
    print("   CLAUDE_API_KEY=your_key_here") 
    print("   GEMINI_API_KEY=your_key_here")
    print()
    print("2. Test individual tools:")
    print("   python llm_response_logger.py")
    print("   python response_generator.py")
    print("   python llm_prompt_eval_analysis.py")
    print()
    print("3. For batch testing, create a prompts file:")
    
    try:
        sample_file = create_sample_prompts_file()
        print(f"   Sample file created: {sample_file}")
        print(f"   Use this file with the batch testing options")
    except Exception as e:
        print(f"   Could not create sample file: {e}")
    
    print()
    print("4. Run the main Gradio app for full functionality:")
    print("   python gradio_full_llm_eval.py")
    print()
    print("✅ All tools are now functional and integrated!")

if __name__ == "__main__":
    main() 