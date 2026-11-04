import csv
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai
from round_robin_evaluator import comprehensive_round_robin_evaluation, save_comprehensive_results
from datetime import datetime
from model_config import (
    OPENAI_MODEL,
    CLAUDE_MODEL,
    GEMINI_MODEL,
    MODEL_KEY_GPT,
    MODEL_KEY_CLAUDE,
    MODEL_KEY_GEMINI,
)

# Load API keys from .env
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gpt4_response(prompt):
    """Get response from GPT-4."""
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with GPT-4: {e}")
        return None

def get_claude_response(prompt):
    """Get response from Claude."""
    try:
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error with {MODEL_KEY_CLAUDE}: {e}")
        return None

def get_gemini_response(prompt):
    """Get response from Gemini."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error with Gemini: {e}")
        return None

def quick_test_all_models():
    """Quick test function to test all models and save results."""
    print("=== LLM Response Logger - Quick Test Tool ===\n")
    
    # Get prompt from user
    prompt = input("Enter your test prompt: ")
    if not prompt.strip():
        print("No prompt provided. Exiting.")
        return
    
    print(f"\nTesting prompt: '{prompt}'")
    print("=" * 50)
    
    # Collect responses from all models
    responses = {}
    
    print("\n1. Getting GPT-4 response...")
    gpt_response = get_gpt4_response(prompt)
    if gpt_response:
        responses[MODEL_KEY_GPT] = gpt_response
        print("GPT-4 response received")
        print(f"   Preview: {gpt_response[:100]}...")
    else:
        print("GPT-4 failed")
    
    print("\n2. Getting Claude response...")
    claude_response = get_claude_response(prompt)
    if claude_response:
        responses[MODEL_KEY_CLAUDE] = claude_response
        print("Claude response received")
        print(f"   Preview: {claude_response[:100]}...")
    else:
        print("Claude failed")
    
    print("\n3. Getting Gemini response...")
    gemini_response = get_gemini_response(prompt)
    if gemini_response:
        responses[MODEL_KEY_GEMINI] = gemini_response
        print("Gemini response received")
        print(f"   Preview: {gemini_response[:100]}...")
    else:
        print("Gemini failed")
    
    if not responses:
        print("\nNo models responded successfully. Check your API keys.")
        return
    
    print(f"\nSuccessfully collected {len(responses)} responses")
    
    # Perform comprehensive evaluation
    print("\n4. Performing comprehensive evaluation...")
    try:
        comprehensive_results = comprehensive_round_robin_evaluation(responses, prompt)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = save_comprehensive_results(comprehensive_results, prompt, timestamp)
        
        if csv_file:
            print(f"Results saved to: {csv_file}")
        
        # Display summary
        print("\n=== EVALUATION SUMMARY ===")
        for model, data in comprehensive_results.items():
            avg_scores = data.get('average_scores', {})
            print(f"\n{model}:")
            print(f"  Helpfulness: {avg_scores.get('helpfulness', 'N/A')}")
            print(f"  Correctness: {avg_scores.get('correctness', 'N/A')}")
            print(f"  Coherence: {avg_scores.get('coherence', 'N/A')}")
            print(f"  Evaluators: {list(data.get('evaluations', {}).keys())}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    print("\n=== Test completed ===")

def batch_test_from_file(filename):
    """Test multiple prompts from a file."""
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return
    
    print(f"=== Batch Testing from {filename} ===")
    
    with open(filename, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(prompts)} prompts to test")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Testing Prompt {i}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")
        
        # Quick test for this prompt
        responses = {}
        
        gpt_response = get_gpt4_response(prompt)
        if gpt_response:
            responses[MODEL_KEY_GPT] = gpt_response
        
        claude_response = get_claude_response(prompt)
        if claude_response:
            responses[MODEL_KEY_CLAUDE] = claude_response
        
        gemini_response = get_gemini_response(prompt)
        if gemini_response:
            responses[MODEL_KEY_GEMINI] = gemini_response
        
        if responses:
            try:
                comprehensive_results = comprehensive_round_robin_evaluation(responses, prompt)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = save_comprehensive_results(comprehensive_results, prompt, f"{timestamp}_prompt_{i}")
                print(f"Results saved for prompt {i}")
            except Exception as e:
                print(f"Evaluation failed for prompt {i}: {e}")
        else:
            print(f"No responses for prompt {i}")
    
    print("\n=== Batch testing completed ===")

if __name__ == "__main__":
    print("LLM Response Logger - Standalone Testing Tool")
    print("1. Quick test with single prompt")
    print("2. Batch test from file")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == "1":
        quick_test_all_models()
    elif choice == "2":
        filename = input("Enter filename with prompts (one per line): ").strip()
        batch_test_from_file(filename)
    else:
        print("Invalid choice. Running quick test...")
        quick_test_all_models()
