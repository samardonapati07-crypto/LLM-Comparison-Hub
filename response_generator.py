import os
from dotenv import load_dotenv
from universal_model_wrapper import universal_model_responses

# Load API keys from .env
load_dotenv()

def generate_all_responses(prompt, resume=None, job_description=None):
    return universal_model_responses(prompt, resume, job_description)

def generate_all_responses_with_reasoning(prompt, selected_models=None, resume=None, job_description=None):
    """
    Generate responses from all selected models with reasoning.
    Uses the universal model wrapper for enhanced functionality.
    """
    # Get responses from the universal wrapper (respect model selection)
    all_responses = universal_model_responses(
        prompt, resume, job_description, selected_models
    )
    
    # Filter by selected models if specified
    if selected_models:
        filtered_responses = {
            model: data 
            for model, data in all_responses.items() 
            if model in selected_models
        }
        return filtered_responses
    
    return all_responses
