import os
from openai import OpenAI
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import csv
import json
import re
from model_config import (
    OPENAI_MODEL,
    CLAUDE_MODEL,
    GEMINI_MODEL,
    MODEL_KEY_GPT,
    MODEL_KEY_CLAUDE,
    MODEL_KEY_GEMINI,
)

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def safe_parse_json(text):
    """Extract and parse JSON from a possibly noisy LLM output."""
    try:
        match = re.search(r'{.*}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[Safe JSON Parse Error] {e}")
    return None

def evaluate_response(evaluator_model, prompt, target_model, response_text):
    """Evaluate a response using the specified evaluator model."""
    evaluation_prompt = (
        f"You are an AI tasked with evaluating another model's response.\n"
        f"Here is the original prompt: \"{prompt}\"\n"
        f"Here is the response from {target_model}: \"{response_text}\"\n\n"
        f"Evaluate this response on the following criteria from 0 (worst) to 1 (best):\n"
        f"- Helpfulness\n- Correctness\n- Coherence\n- Tone\n- Accuracy\n"
        f"- Relevance\n- Completeness\n- Clarity\n\n"
        f"Return ONLY a valid JSON object with the following keys:\n"
        f"{{\n"
        f"  \"helpfulness\": <float>,\n"
        f"  \"correctness\": <float>,\n"
        f"  \"coherence\": <float>,\n"
        f"  \"tone_score\": <float>,\n"
        f"  \"accuracy\": <float>,\n"
        f"  \"relevance\": <float>,\n"
        f"  \"completeness\": <float>,\n"
        f"  \"clarity\": <float>,\n"
        f"  \"reasoning\": \"explanation\",\n"
        f"  \"notes\": \"additional remarks\"\n"
        f"}}"
    )

    try:
        if evaluator_model == MODEL_KEY_GPT:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.3
            )
            result = response.choices[0].message.content
        elif evaluator_model == MODEL_KEY_CLAUDE:
            response = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            result = response.content[0].text
        elif evaluator_model == MODEL_KEY_GEMINI:
            model = genai.GenerativeModel(GEMINI_MODEL)
            eval_response = model.generate_content(evaluation_prompt)
            result = eval_response.text
        else:
            print(f"Unknown evaluator model: {evaluator_model}")
            return None

        parsed = safe_parse_json(result)
        if parsed:
            return parsed
        else:
            print(f"Failed to parse JSON from {evaluator_model} evaluation")
            return None

    except Exception as e:
        print(f"Error in {evaluator_model} evaluation: {str(e)}")
        return None

def comprehensive_round_robin_evaluation(responses_dict, prompt):
    print("\nStarting comprehensive round-robin evaluation...")

    evaluation_matrix = {
        MODEL_KEY_GPT: [MODEL_KEY_CLAUDE, MODEL_KEY_GEMINI],
        MODEL_KEY_CLAUDE: [MODEL_KEY_GPT, MODEL_KEY_GEMINI],
        MODEL_KEY_GEMINI: [MODEL_KEY_GPT, MODEL_KEY_CLAUDE],
    }

    comprehensive_results = {}

    for target_model, response_text in responses_dict.items():
        print(f"\nCollecting evaluations for {target_model}...")
        comprehensive_results[target_model] = {
            'response': response_text,
            'evaluations': {},
            'average_scores': {}
        }

        for evaluator in evaluation_matrix[target_model]:
            print(f"  {evaluator} evaluating {target_model}...")
            evaluation = evaluate_response(evaluator, prompt, target_model, response_text)
            if evaluation:
                comprehensive_results[target_model]['evaluations'][evaluator] = evaluation
                print(f"    {evaluator} evaluation completed")
            else:
                print(f"    {evaluator} evaluation failed")

        if comprehensive_results[target_model]['evaluations']:
            metrics = ['helpfulness', 'correctness', 'coherence', 'tone_score',
                       'accuracy', 'relevance', 'completeness', 'clarity']
            for metric in metrics:
                scores = [
                    eval_data[metric]
                    for eval_data in comprehensive_results[target_model]['evaluations'].values()
                    if metric in eval_data and isinstance(eval_data[metric], (int, float))
                ]
                comprehensive_results[target_model]['average_scores'][metric] = round(sum(scores) / len(scores), 3) if scores else 0.5

    print(f"\nComprehensive evaluation completed for {len(comprehensive_results)} models")
    return comprehensive_results

def save_comprehensive_results(comprehensive_results, prompt, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"results/comprehensive_eval_{timestamp}.csv"
    os.makedirs("results", exist_ok=True)

    rows = []
    for model, data in comprehensive_results.items():
        avg_scores = data.get('average_scores', {})
        for evaluator, evaluation in data.get('evaluations', {}).items():
            row = {
                'timestamp': timestamp,
                'prompt': prompt,
                'target_model': model,
                'evaluator': evaluator,
                'response': data['response'],
                'helpfulness': evaluation.get('helpfulness', 0.5),
                'correctness': evaluation.get('correctness', 0.5),
                'coherence': evaluation.get('coherence', 0.5),
                'tone_score': evaluation.get('tone_score', 0.5),
                'accuracy': evaluation.get('accuracy', 0.5),
                'relevance': evaluation.get('relevance', 0.5),
                'completeness': evaluation.get('completeness', 0.5),
                'clarity': evaluation.get('clarity', 0.5),
                'reasoning': evaluation.get('reasoning', ''),
                'notes': evaluation.get('notes', ''),
                'avg_helpfulness': avg_scores.get('helpfulness', 0.5),
                'avg_correctness': avg_scores.get('correctness', 0.5),
                'avg_coherence': avg_scores.get('coherence', 0.5),
                'avg_tone_score': avg_scores.get('tone_score', 0.5),
                'avg_accuracy': avg_scores.get('accuracy', 0.5),
                'avg_relevance': avg_scores.get('relevance', 0.5),
                'avg_completeness': avg_scores.get('completeness', 0.5),
                'avg_clarity': avg_scores.get('clarity', 0.5)
            }
            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {filename}")
        return filename
    else:
        print("No results to save")
        return None
