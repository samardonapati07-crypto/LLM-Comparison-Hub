---
title: LLM Comparison Hub
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: false
---

# LLM Comparison Hub

A comprehensive tool for comparing responses from multiple Large Language Models (GPT-4, Claude Sonnet 4, Gemini 3 Flash) with built-in evaluation, analysis, and visualization capabilities.

**Live Demo:** [https://huggingface.co/spaces/chunchu-08/LLM-Comparison-Hub](https://huggingface.co/spaces/chunchu-08/LLM-Comparison-Hub)

## Overview

This application provides a complete LLM comparison and evaluation system that generates responses from multiple models, performs round-robin evaluations where each model evaluates all others, and provides comprehensive analysis with interactive visualizations.

## Key Features

- **Multi-Model Response Generation**: Dynamically generate responses from any combination of GPT-4, Claude Sonnet 4, and Gemini 3 Flash using a simple model selector.
- **Dynamic Round-Robin Evaluation**: A robust evaluation system where selected models evaluate each other. If a model is deselected, the evaluation logic adapts automatically.
- **Real-time Query Detection**: Automatically detects if a prompt requires current information and fetches it using a Google search fallback.
- **ATS Scoring**: Performs detailed resume vs. job description matching and scoring.
- **Interactive Data Analysis & Visualization**: Generates consistent, professionally styled charts (Heatmap, Radar, Bar) for all prompt types.
- **Batch Processing**: Handles multiple prompts from CSV files.
- **Modular Architecture**: A clean, production-ready codebase with a new `universal_model_wrapper.py` that centralizes core logic.
- **Gradio Web Interface**: A user-friendly web UI with a model selector to easily choose which LLMs to run.
- **Export Capabilities**: Download a ZIP bundle with all evaluation results and interactive HTML charts.
- **Automated Deployment**: GitHub Actions for continuous deployment to Hugging Face Spaces.

## Project Architecture

The architecture has been refactored for simplicity and robustness.

### Core Application Files

- **`app.py`** - Main Gradio web interface, including UI logic and the model selector.
- **`model_config.py`** - API model IDs and UI/routing keys (GPT-4, Claude Sonnet 4, Gemini 3 Flash).
- **`universal_model_wrapper.py`** - **New core module!** Centralizes all LLM API calls, real-time detection, search fallback, and ATS/general prompt logic.
- **`response_generator.py`** - A simplified wrapper that interfaces between the app and the `universal_model_wrapper`.
- **`round_robin_evaluator.py`** - A dynamic evaluation engine that adapts to the models selected in the UI.
- **`llm_prompt_eval_analysis.py`** - Data analysis and visualization engine.
- **`llm_response_logger.py`** - Quick testing and logging tool.

### Supporting Modules

- **`search_fallback.py`**: This file is kept for reference, but its core functionality has been integrated into `universal_model_wrapper.py` for a more robust, self-contained architecture.

## Usage

### Web Interface (Recommended)

Launch the Gradio web interface:
```bash
python app.py
```

The interface provides:
- **Input Section**: Enter prompts, upload files, and use the **Model Selector** checkboxes to choose which LLMs to run.
- **Results Tabs**: View responses, evaluations, search results, and interactive visualizations.
- **Export Options**: Download results as ZIP bundles with interactive HTML charts.
- **Real-time Features**: Automatic query detection and search enhancement.

### Model Selection
The UI now includes a set of checkboxes allowing you to select any combination of models (GPT-4, Claude Sonnet 4, Gemini 3 Flash) for a given query. The application, including the round-robin evaluation, will dynamically adapt to your selection.

## Technical Architecture

### Design Principles
- **Centralized Logic**: The new `universal_model_wrapper.py` acts as a single source of truth for model interaction.
- **Dynamic & Robust**: The evaluation system is no longer static and adapts to user input, preventing crashes when models are deselected.
- **Separation of Concerns**: Each file has a clear, specific responsibility.
- **Clean Code**: Production-ready and easy to maintain.
- **Hugging Face Compatible**: No external browser dependencies for chart generation.

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `model_config.py` | API model IDs and UI/routing keys for each provider. |
| `app.py` | UI orchestration, including the model selector and deployment. |
| `universal_model_wrapper.py` | Handles all LLM calls, prompt logic, and search. |
| `response_generator.py` | Connects the UI to the universal wrapper. |
| `round_robin_evaluator.py` | Dynamically evaluates the currently selected models. |
| `llm_prompt_eval_analysis.py` | Data analysis and visualization. |

## Installation

### Prerequisites

- Python 3.8 or higher
- API keys for OpenAI, Anthropic, and Google Generative AI

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLM-Compare-Hub
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   CLAUDE_API_KEY=your_claude_key_here
   GEMINI_API_KEY=your_gemini_key_here
   GOOGLE_API_KEY=your_google_key_here
   GOOGLE_CSE_ID=your_google_cse_id_here
   ```

## API Requirements

### Required APIs
- **OpenAI API**: For GPT-4-class responses and ATS scoring
- **Anthropic API**: For Claude Sonnet 4 responses
- **Google Generative AI**: For Gemini 3 Flash responses

### Optional APIs
- **Google Custom Search**: For real-time query enhancement

## Evaluation Metrics

The system evaluates responses on eight comprehensive criteria:

- **Helpfulness**: How useful and informative is the response?
- **Correctness**: How accurate and factually correct is the response?
- **Coherence**: How well-structured and logical is the response?
- **Tone Score**: How appropriate and professional is the tone?
- **Accuracy**: How precise and detailed is the information?
- **Relevance**: How well does the response address the prompt?
- **Completeness**: How comprehensive is the response?
- **Clarity**: How clear and easy to understand is the response?

## ATS Scoring System

When a resume and job description are provided, the system performs ATS (Applicant Tracking System) scoring:

- **Keyword Matching**: Identifies relevant skills and qualifications
- **Section Weighting**: Prioritizes important sections
- **Semantic Similarity**: Analyzes meaning and context
- **Recency/Frequency**: Considers experience relevance
- **Penalty Detection**: Identifies potential issues
- **Aggregation**: Provides overall match score

## Output and Results

### Generated Files
- **CSV Files**: Comprehensive evaluation results with timestamps
- **Analysis Reports**: Detailed analysis and insights
- **Interactive Visualizations**: Interactive HTML charts and graphs
- **Export Bundles**: ZIP files containing all results and interactive charts

### File Naming Convention
- `evaluation_YYYYMMDD_HHMMSS.csv` - Evaluation results
- `batch_YYYYMMDD_HHMMSS/` - Results directory
- `heatmap.html`, `radar.html`, `barchart.html` - Interactive visualization files
- `bundle.zip` - Complete export package
