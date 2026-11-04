import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import numpy as np

def load_latest_csv():
    """Load the most recent CSV file from the results directory."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("Results directory not found.")
        return None
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in results directory.")
        return None
    
    # Get the most recent file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"Loading latest file: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        return df, latest_file
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None

def load_specific_csv(filename):
    """Load a specific CSV file."""
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return None, None
    
    try:
        df = pd.read_csv(filename)
        return df, filename
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None

def analyze_evaluation_data(df):
    """Analyze evaluation data and generate insights."""
    print("=== LLM Evaluation Data Analysis ===\n")
    
    if df is None or df.empty:
        print("No data to analyze.")
        return
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Number of evaluations: {len(df)}")
    
    # Basic statistics
    print("\n=== BASIC STATISTICS ===")
    
    # Check if we have comprehensive evaluation data
    if 'target_model' in df.columns and 'evaluator' in df.columns:
        print("\nModel Performance Summary:")
        model_stats = df.groupby('target_model').agg({
            'helpfulness': ['mean', 'std'],
            'correctness': ['mean', 'std'],
            'coherence': ['mean', 'std'],
            'clarity': ['mean', 'std']
        }).round(3)
        print(model_stats)
        
        print("\nEvaluator Bias Analysis:")
        evaluator_stats = df.groupby('evaluator').agg({
            'helpfulness': ['mean', 'std'],
            'correctness': ['mean', 'std'],
            'coherence': ['mean', 'std'],
            'clarity': ['mean', 'std']
        }).round(3)
        print(evaluator_stats)
        
        # Cross-evaluation analysis
        print("\n=== CROSS-EVALUATION ANALYSIS ===")
        cross_eval = df.groupby(['target_model', 'evaluator']).agg({
            'helpfulness': 'mean',
            'correctness': 'mean',
            'coherence': 'mean',
            'clarity': 'mean'
        }).round(3)
        print(cross_eval)
        
    else:
        # Simple analysis for basic CSV format
        print("\nModel Performance (Simple Format):")
        if 'model' in df.columns:
            model_performance = df.groupby('model').agg({
                'helpfulness': ['mean', 'std'],
                'correctness': ['mean', 'std'],
                'coherence': ['mean', 'std'],
                'clarity': ['mean', 'std']
            }).round(3)
            print(model_performance)

def generate_visualizations(df, output_dir="analysis_results"):
    """Generate visualizations from evaluation data."""
    if df is None or df.empty:
        print("No data for visualization.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    if 'target_model' in df.columns:
        metrics = ['helpfulness', 'correctness', 'coherence', 'clarity']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Create box plot
            sns.boxplot(data=df, x='target_model', y=metric, ax=ax)
            ax.set_title(f'{metric.title()} Scores by Model')
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Evaluator Bias Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Evaluator Bias Analysis', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Create box plot
            sns.boxplot(data=df, x='evaluator', y=metric, ax=ax)
            ax.set_title(f'{metric.title()} Scores by Evaluator')
            ax.set_xlabel('Evaluator')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/evaluator_bias_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of Cross-Evaluations
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values='helpfulness', 
            index='target_model', 
            columns='evaluator', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=0.5, 
                   fmt='.3f', cbar_kws={'label': 'Average Helpfulness Score'})
        plt.title('Cross-Evaluation Heatmap (Helpfulness)')
        plt.xlabel('Evaluator')
        plt.ylabel('Target Model')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cross_evaluation_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Correlation Matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.3f', square=True)
            plt.title('Metric Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_matrix_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def generate_report(df, filename):
    """Generate a comprehensive analysis report."""
    if df is None or df.empty:
        print("No data for report generation.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"analysis_results/evaluation_report_{timestamp}.txt"
    
    os.makedirs("analysis_results", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("LLM EVALUATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data source: {filename}\n")
        f.write(f"Total records: {len(df)}\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Columns: {list(df.columns)}\n\n")
        
        # Model performance
        if 'target_model' in df.columns:
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            model_stats = df.groupby('target_model').agg({
                'helpfulness': ['mean', 'std', 'count'],
                'correctness': ['mean', 'std'],
                'coherence': ['mean', 'std'],
                'clarity': ['mean', 'std']
            }).round(3)
            
            f.write(str(model_stats) + "\n\n")
            
            # Best performing model
            avg_scores = df.groupby('target_model')[['helpfulness', 'correctness', 'coherence', 'clarity']].mean()
            best_model = avg_scores.mean(axis=1).idxmax()
            f.write(f"Best performing model (overall): {best_model}\n")
            f.write(f"Average score: {avg_scores.mean(axis=1).max():.3f}\n\n")
            
            # Evaluator analysis
            f.write("EVALUATOR ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            evaluator_stats = df.groupby('evaluator').agg({
                'helpfulness': ['mean', 'std'],
                'correctness': ['mean', 'std'],
                'coherence': ['mean', 'std'],
                'clarity': ['mean', 'std']
            }).round(3)
            
            f.write(str(evaluator_stats) + "\n\n")
            
            # Most lenient/strict evaluator
            evaluator_means = df.groupby('evaluator')[['helpfulness', 'correctness', 'coherence', 'clarity']].mean()
            most_lenient = evaluator_means.mean(axis=1).idxmax()
            most_strict = evaluator_means.mean(axis=1).idxmin()
            
            f.write(f"Most lenient evaluator: {most_lenient}\n")
            f.write(f"Most strict evaluator: {most_strict}\n\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 15 + "\n")
        
        if 'target_model' in df.columns:
            # Model strengths
            f.write("Model Strengths:\n")
            for metric in ['helpfulness', 'correctness', 'coherence', 'clarity']:
                best_model_metric = df.groupby('target_model')[metric].mean().idxmax()
                best_score = df.groupby('target_model')[metric].mean().max()
                f.write(f"  {metric.title()}: {best_model_metric} ({best_score:.3f})\n")
            
            f.write("\nModel Weaknesses:\n")
            for metric in ['helpfulness', 'correctness', 'coherence', 'clarity']:
                worst_model_metric = df.groupby('target_model')[metric].mean().idxmin()
                worst_score = df.groupby('target_model')[metric].mean().min()
                f.write(f"  {metric.title()}: {worst_model_metric} ({worst_score:.3f})\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Report generation completed.\n")
    
    print(f"Analysis report saved to: {report_file}")

def main():
    """Main function for the analysis tool."""
    print("=== LLM Prompt Evaluation Analysis Tool ===\n")
    print("1. Analyze latest CSV file")
    print("2. Analyze specific CSV file")
    print("3. Generate visualizations")
    print("4. Generate comprehensive report")
    print("5. Full analysis (load + analyze + visualize + report)")
    
    choice = input("\nChoose option (1-5): ").strip()
    
    df = None
    filename = None
    
    if choice == "1":
        df, filename = load_latest_csv()
        if df is not None:
            analyze_evaluation_data(df)
    
    elif choice == "2":
        file_path = input("Enter CSV file path: ").strip()
        df, filename = load_specific_csv(file_path)
        if df is not None:
            analyze_evaluation_data(df)
    
    elif choice == "3":
        df, filename = load_latest_csv()
        if df is not None:
            generate_visualizations(df)
    
    elif choice == "4":
        df, filename = load_latest_csv()
        if df is not None:
            generate_report(df, filename)
    
    elif choice == "5":
        df, filename = load_latest_csv()
        if df is not None:
            print("\n" + "="*50)
            analyze_evaluation_data(df)
            print("\n" + "="*50)
            generate_visualizations(df)
            print("\n" + "="*50)
            generate_report(df, filename)
            print("\n" + "="*50)
            print("Full analysis completed!")
    
    else:
        print("Invalid choice. Running full analysis...")
        df, filename = load_latest_csv()
        if df is not None:
            analyze_evaluation_data(df)
            generate_visualizations(df)
            generate_report(df, filename)

if __name__ == "__main__":
    main()
