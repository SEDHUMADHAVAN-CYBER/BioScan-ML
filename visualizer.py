"""
Visualization Module
Creates plots and charts for model evaluation and feature importance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


class Visualizer:
    """
    Class to create various visualizations for ML results
    """
    
    def __init__(self):
        """Initialize visualizer with style settings"""
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
    def plot_confusion_matrix(self, cm, model_name, target_names):
        """
        Create confusion matrix heatmap
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            target_names: Names of target classes
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, features, importances, top_n=10):
        """
        Create feature importance bar chart
        
        Args:
            features: List of feature names
            importances: List of importance values
            top_n: Number of top features to display
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(importances / importances.max())
        
        bars = ax.barh(y_pos, importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Important Biomarkers (Random Forest)', 
                     fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(importance, i, f' {importance:.4f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, comparison_df):
        """
        Create grouped bar chart comparing model metrics
        
        Args:
            comparison_df: DataFrame with model comparison data
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics))
        width = 0.35
        
        # Plot bars for each model
        models = comparison_df['Model'].tolist()
        colors = ['#2E86AB', '#A23B72']
        
        for i, model in enumerate(models):
            values = comparison_df[comparison_df['Model'] == model][metrics].values[0]
            offset = width * (i - len(models)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_comparison(self, comparison_df):
        """
        Create interactive plotly chart for model comparison
        
        Args:
            comparison_df: DataFrame with model comparison data
            
        Returns:
            plotly figure
        """
        # Melt dataframe for plotly
        df_melted = comparison_df.melt(
            id_vars=['Model'], 
            var_name='Metric', 
            value_name='Score'
        )
        
        # Create grouped bar chart
        fig = px.bar(
            df_melted,
            x='Metric',
            y='Score',
            color='Model',
            barmode='group',
            title='Interactive Model Performance Comparison',
            color_discrete_sequence=['#2E86AB', '#A23B72'],
            text='Score'
        )
        
        # Update layout
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            xaxis_title='Metrics',
            yaxis_title='Score',
            yaxis_range=[0, 1.1],
            font=dict(size=12),
            title_font_size=16,
            legend_title_text='Model',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_metrics_radar(self, comparison_df):
        """
        Create radar chart for model comparison
        
        Args:
            comparison_df: DataFrame with model comparison data
            
        Returns:
            plotly figure
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = go.Figure()
        
        colors = ['#2E86AB', '#A23B72']
        
        for i, row in comparison_df.iterrows():
            values = row[metrics].tolist()
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=row['Model'],
                line_color=colors[i],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title='Model Performance Radar Chart',
            title_font_size=16
        )
        
        return fig
    
    def create_summary_table(self, comparison_df):
        """
        Create a formatted summary table
        
        Args:
            comparison_df: DataFrame with model comparison data
            
        Returns:
            Styled DataFrame
        """
        # Format values to 4 decimal places
        formatted_df = comparison_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:.4f}')
        
        return formatted_df
