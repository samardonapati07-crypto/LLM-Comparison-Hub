# app.py – Final Updated Version with Unified Visualization (Model selection-safe + Visualization Fixes)
import gradio as gr
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import zipfile
import json
from datetime import datetime
from dotenv import load_dotenv

from response_generator import generate_all_responses_with_reasoning
from round_robin_evaluator import comprehensive_round_robin_evaluation
from model_config import OPENAI_MODEL, MODEL_KEYS_ORDERED, MODEL_KEY_GPT, MODEL_KEY_CLAUDE, MODEL_KEY_GEMINI

load_dotenv()
# Plotly >=6.1: pio.defaults; older: kaleido.scope (deprecated after 6.2)
try:
    pio.defaults.default_format = "png"
except AttributeError:
    pio.kaleido.scope.default_format = "png"

metrics = ['helpfulness', 'correctness', 'coherence', 'tone_score',
           'accuracy', 'relevance', 'completeness', 'clarity']

def extract_text_from_resume(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        import fitz
        with fitz.open(file.name) as doc:
            return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        import docx
        doc = docx.Document(file.name)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        return file.read().decode('utf-8')
    return ""

def ats_score_advanced(response, resume, jd):
    prompt = f"""
You are a professional ATS scoring engine. Compare the generated response to the candidate's resume and job description using:
1. Keyword Matching
2. Section Weighting
3. Semantic Similarity
4. Recency/Frequency
5. Penalty Detection
6. Aggregation

Resume:
{resume}

Job Description:
{jd}

Response:
{response}

Return JSON:
{{"ats_score": <0-100>, "strengths": ["..."], "gaps": ["..."], "suggestions": ["..."]}}
"""
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        res = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(res.choices[0].message.content.strip())
    except:
        return {"ats_score": 50, "strengths": [], "gaps": [], "suggestions": ["Check formatting."]}

def create_visualizations(df, results_dir):
    html_files = []
    summary = df.groupby('target_model')[metrics].mean().reset_index()
    font_style = dict(family="Arial, sans-serif", size=12, color="black")

    # 1. Heatmap with professional styling
    heatmap = px.imshow(
        summary[metrics].values,
        x=metrics,
        y=summary['target_model'],
        labels=dict(x="Metric", y="Model", color="Score"),
        title="<b>Heatmap: Metrics Across Models</b>",
        color_continuous_scale='Viridis'
    )
    heatmap.update_layout(
        margin=dict(l=80, r=40, t=80, b=120),
        xaxis_tickangle=-45,
        title_font=dict(size=18, family="Arial, sans-serif"),
        font=font_style
    )
    heatmap_path = os.path.join(results_dir, "heatmap.html")
    heatmap.write_html(heatmap_path)
    html_files.append(heatmap_path)

    # 2. Radar Chart with professional styling
    radar = go.Figure()
    for _, row in summary.iterrows():
        radar.add_trace(go.Scatterpolar(
            r=list(row[metrics]),
            theta=metrics,
            fill='toself',
            name=row['target_model']
        ))
    radar.update_layout(
        title="<b>Radar Chart: Model Score Profiles</b>",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend_title_text='Models',
        title_font=dict(size=18, family="Arial, sans-serif"),
        font=font_style,
        margin=dict(l=60, r=60, t=80, b=80)
    )
    radar_path = os.path.join(results_dir, "radar.html")
    radar.write_html(radar_path)
    html_files.append(radar_path)

    # 3. Bar Chart with professional styling
    bar = px.bar(
        summary.melt(id_vars='target_model'),
        x='variable',
        y='value',
        color='target_model',
        barmode='group',
        title="<b>Bar Chart: Metric Comparison</b>",
        labels={'variable': 'Metric', 'value': 'Score', 'target_model': 'Model'}
    )
    bar.update_layout(
        margin=dict(l=60, r=20, t=80, b=120),
        xaxis_tickangle=-45,
        legend_title_text='Model',
        title_font=dict(size=18, family="Arial, sans-serif"),
        font=font_style
    )
    bar_path = os.path.join(results_dir, "barchart.html")
    bar.write_html(bar_path)
    html_files.append(bar_path)

    return (heatmap, radar, bar), html_files

def _empty_eval_plots():
    """Placeholders so Gradio always receives three Plot outputs when there are no evaluation rows."""

    def one(title):
        fig = go.Figure()
        fig.add_annotation(
            text="No evaluation data (API errors or empty evaluators).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13),
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=280,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        return fig

    return [one("Heatmap"), one("Radar"), one("Bar chart")]

def format_ats_feedback(score, strengths, gaps, suggestions):
    color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
    return f"""
### ATS Match Score: ~{score}% {color}

#### **Strengths / High Matches:**
{chr(10).join([f"* {s}" for s in strengths]) if strengths else "* None found."}

#### **Partial or Missing:**
{chr(10).join([f"* {g}" for g in gaps]) if gaps else "* None mentioned."}

#### **How to Improve ATS Score:**
{chr(10).join([f"1. {s}" for s in suggestions]) if suggestions else "1. Add missing skills."}
"""

def process_prompt(prompt, enable_realtime, enable_eval, enable_analysis, user_file, model_selection):
    selected_models = [m for m, enabled in zip(MODEL_KEYS_ORDERED, model_selection) if enabled]
    resume_text = ""
    job_description = prompt
    batch_mode = user_file and hasattr(user_file, 'name') and user_file.name.endswith(".csv")
    resume_mode = user_file and hasattr(user_file, 'name') and user_file.name.lower().endswith(('.pdf', '.docx', '.txt'))

    prompts = [prompt]
    ats_summary_texts = []
    search_results = ""

    if batch_mode:
        df_batch = pd.read_csv(user_file.name)
        prompts = df_batch['prompt'].dropna().tolist()
    elif resume_mode:
        resume_text = extract_text_from_resume(user_file)

    all_rows, all_charts = [], []
    zip_path, ats_table_markdown = None, ""
    responses = {}

    for prompt_text in prompts:
        responses = generate_all_responses_with_reasoning(
            prompt_text,
            selected_models,
            resume_text if resume_mode else None,
            job_description if resume_mode else None
        )

        if responses:
            first_response = list(responses.values())[0]
            search_results = first_response.get('search_results', '')
            is_ats = first_response.get('is_ats', False)

        ats_rows = []
        for model in responses:
            model_resp = responses[model]['response']
            model_reasoning = responses[model]['reasoning']
            responses[model]['ats_embed'] = f"### Response\n\n{model_resp}\n\n---\n\n**Explainability:**\n{model_reasoning}"

            if resume_mode and is_ats:
                try:
                    ats_result = ats_score_advanced(model_resp, resume_text, prompt_text)
                    ats_rows.append(f"| {model} | {ats_result['ats_score']} | {', '.join(ats_result.get('strengths', []))} | {', '.join(ats_result.get('suggestions', []))} |")
                except:
                    ats_rows.append(f"| {model} | N/A | N/A | N/A |")

        if ats_rows:
            ats_table_markdown = "| Model | Score | Strengths | Suggestions |\n|-------|-------|-----------|-------------|\n" + "\n".join(ats_rows)

        # Always run evaluation to generate chart data
        compact = {k: v['response'] for k, v in responses.items()}
        eval_result = comprehensive_round_robin_evaluation(compact, prompt_text)
        for model, data in eval_result.items():
            for evaluator, scores in data['evaluations'].items():
                row = {
                    'prompt': prompt_text,
                    'target_model': model,
                    'evaluator': evaluator,
                    'response': responses[model]['response'],
                    'explainability': responses[model]['reasoning']
                }
                row.update({k: scores.get(k, 0.5) for k in metrics})
                row.update({f"avg_{k}": data['average_scores'].get(k, 0.5) for k in metrics})
                all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    if not df_all.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/batch_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "evaluation.csv")
        df_all.to_csv(csv_path, index=False)
        (heatmap, radar, bar), chart_paths = create_visualizations(df_all, results_dir)
        all_charts = [heatmap, radar, bar]
        zip_path = os.path.join(results_dir, "bundle.zip")
        batch_out = os.path.join(results_dir, "batch_prompts_output.csv")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(csv_path, arcname="evaluation.csv")
            for chart in chart_paths:
                zipf.write(chart, arcname=os.path.basename(chart))
            if batch_mode:
                df_batch['ATS Summary'] = ats_summary_texts
                df_batch.to_csv(batch_out, index=False)
                zipf.write(batch_out, arcname="batch_prompts_output.csv")
    else:
        all_charts = _empty_eval_plots()

    # Conditional UI updates
    eval_table = df_all[['target_model', 'evaluator'] + metrics] if not df_all.empty and enable_eval else pd.DataFrame()
    ats_md = ats_table_markdown if resume_mode else ""

    def _tab_markdown(model_name):
        if model_name not in responses:
            return f"*{model_name} was not selected.*"
        r = responses[model_name]
        return r.get("ats_embed", r.get("response", ""))

    return (
        _tab_markdown(MODEL_KEY_GPT),
        _tab_markdown(MODEL_KEY_CLAUDE),
        _tab_markdown(MODEL_KEY_GEMINI),
        search_results or "N/A",
        all_charts[0],
        all_charts[1],
        all_charts[2],
        eval_table,
        ats_md,
        zip_path,
    )

def download_results(path):
    return path if path and os.path.exists(path) else None

def create_interface():
    with gr.Blocks(title="LLM Comparison Hub") as demo:
        gr.Markdown("""
# LLM Comparison Hub
This app compares LLM responses using round-robin evaluations, with real-time query detection and comprehensive analysis.

**How to use:**
- Enter a prompt (JD or query)
- Upload a resume (PDF/DOCX/TXT) or a CSV with prompts
- Select models
- Click evaluate

**Features:**
- Real-time web search fallback
- Resume vs JD ATS scoring (optional)
- Batch CSV prompt evaluation
- Visualizations (Heatmap, Radar, Bar)
- ZIP export of all results
""")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Enter Prompt", lines=4)
                user_file = gr.File(label="Upload Resume or CSV", file_types=[".pdf", ".docx", ".txt", ".csv"])
                model_selector = gr.CheckboxGroup(
                    label="Select Models",
                    choices=MODEL_KEYS_ORDERED,
                    value=list(MODEL_KEYS_ORDERED),
                )
                enable_realtime = gr.Checkbox(label="Enable real-time detection", value=True)
                enable_eval = gr.Checkbox(label="Enable evaluation", value=True)
                enable_analysis = gr.Checkbox(label="Enable analysis (currently not used)", value=True)
                submit = gr.Button("Run Evaluation")

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab(MODEL_KEY_GPT): gpt_out = gr.Markdown()
                    with gr.Tab(MODEL_KEY_CLAUDE): claude_out = gr.Markdown()
                    with gr.Tab(MODEL_KEY_GEMINI): gemini_out = gr.Markdown()
                    with gr.Tab("Evaluation Table"): df_out = gr.Dataframe()
                    with gr.Tab("ATS Evaluation"): ats_summary = gr.Markdown()
                    with gr.Tab("Search Results"): search_out = gr.Markdown()
                    with gr.Tab("Visualizations"):
                        heatmap_plot = gr.Plot()
                        radar_plot = gr.Plot()
                        bar_plot = gr.Plot()
                export_btn = gr.Button("Download ZIP Bundle")
                zip_output = gr.File(file_types=[".zip"], interactive=False, visible=True)

        submit.click(
            fn=process_prompt,
            inputs=[prompt, enable_realtime, enable_eval, enable_analysis, user_file, model_selector],
            outputs=[gpt_out, claude_out, gemini_out, search_out, heatmap_plot, radar_plot, bar_plot, df_out, ats_summary, zip_output]
        )
        export_btn.click(download_results, inputs=[zip_output], outputs=[zip_output])

    return demo

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
