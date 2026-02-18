"""
Biomarker Prediction Web Application
A complete ML application for predicting biomarkers using breast cancer dataset
Author: B.Tech Final Year Project
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from ml_model import BiomarkerPredictor
from visualizer import Visualizer
import time


# Page configuration
st.set_page_config(
    page_title="Biomarker Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """
    Main function to run the Streamlit application
    """
    
    # Header
    st.markdown('<div class="main-header">🧬 Biomarker Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning Application for Cancer Biomarker Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/dna.png", width=100)
        st.title("⚙️ Configuration")
        
        # Dataset selection
        st.subheader("1️⃣ Dataset Selection")
        dataset_option = st.radio(
            "Choose dataset:",
            ["Use Default Dataset (Breast Cancer)", "Upload Custom CSV"]
        )
        
        uploaded_file = None
        if dataset_option == "Upload Custom CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Last column should be the target variable"
            )
        
        # Model configuration
        st.subheader("2️⃣ Model Configuration")
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Percentage of data to use for testing"
        ) / 100
        
        top_n_features = st.slider(
            "Top N Important Features",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Number of top biomarkers to display"
        )
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.info(
            """
            This application uses Machine Learning to predict cancer biomarkers.
            
            **Models Used:**
            - Random Forest Classifier
            - Support Vector Machine (SVM)
            
            **Metrics Evaluated:**
            - Accuracy
            - Precision
            - Recall
            - F1-Score
            """
        )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset Overview", 
        "🤖 Model Training", 
        "📈 Results & Analysis",
        "📋 Documentation"
    ])
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Tab 1: Dataset Overview
    with tab1:
        st.header("📊 Dataset Overview")
        
        if dataset_option == "Use Default Dataset (Breast Cancer)":
            if st.button("🔄 Load Default Dataset", type="primary"):
                with st.spinner("Loading dataset..."):
                    processor = DataProcessor()
                    X, y, feature_names, target_names = processor.load_default_dataset()
                    
                    # Store in session state
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.feature_names = feature_names
                    st.session_state.target_names = target_names
                    st.session_state.processor = processor
                    st.session_state.data_loaded = True
                    
                    st.success("✅ Dataset loaded successfully!")
        
        elif uploaded_file is not None:
            if st.button("🔄 Load Custom Dataset", type="primary"):
                with st.spinner("Loading custom dataset..."):
                    try:
                        processor = DataProcessor()
                        X, y, feature_names, target_names = processor.load_custom_dataset(uploaded_file)
                        
                        # Store in session state
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.feature_names = feature_names
                        st.session_state.target_names = target_names
                        st.session_state.processor = processor
                        st.session_state.data_loaded = True
                        
                        st.success("✅ Custom dataset loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading dataset: {str(e)}")
        
        # Display dataset information
        if st.session_state.data_loaded:
            st.markdown("---")
            
            # Dataset statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📦 Total Samples", st.session_state.X.shape[0])
            with col2:
                st.metric("🔢 Features", st.session_state.X.shape[1])
            with col3:
                st.metric("🎯 Classes", len(np.unique(st.session_state.y)))
            with col4:
                st.metric("📊 Target Names", len(st.session_state.target_names))
            
            st.markdown("---")
            
            # Feature statistics
            st.subheader("📈 Feature Statistics")
            stats_df = st.session_state.processor.get_feature_statistics(st.session_state.X)
            st.dataframe(stats_df, use_container_width=True, height=400)
            
            # Class distribution
            st.subheader("🎯 Class Distribution")
            class_counts = pd.Series(st.session_state.y).value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart(class_counts)
            with col2:
                for idx, count in class_counts.items():
                    st.write(f"**{st.session_state.target_names[idx]}:** {count} samples ({count/len(st.session_state.y)*100:.1f}%)")
    
    # Tab 2: Model Training
    with tab2:
        st.header("🤖 Model Training")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load a dataset first from the 'Dataset Overview' tab.")
        else:
            st.markdown("""
                <div class="info-box">
                    <strong>ℹ️ Training Information</strong><br>
                    This will train two models: Random Forest and Support Vector Machine (SVM).
                    The training process includes data preprocessing, feature scaling, and model evaluation.
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Preprocessing
                status_text.text("Step 1/3: Preprocessing data...")
                progress_bar.progress(10)
                
                X_train, X_test, y_train, y_test = st.session_state.processor.preprocess_data(
                    st.session_state.X,
                    st.session_state.y,
                    test_size=test_size
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                progress_bar.progress(30)
                time.sleep(0.5)
                
                # Step 2: Training
                status_text.text("Step 2/3: Training models...")
                predictor = BiomarkerPredictor()
                predictor.train_models(X_train, y_train)
                
                st.session_state.predictor = predictor
                progress_bar.progress(70)
                time.sleep(0.5)
                
                # Step 3: Evaluation
                status_text.text("Step 3/3: Evaluating models...")
                results = predictor.evaluate_all_models(X_test, y_test)
                
                st.session_state.results = results
                st.session_state.models_trained = True
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed!")
                
                st.markdown("""
                    <div class="success-box">
                        <strong>✅ Success!</strong><br>
                        Models trained and evaluated successfully. Check the 'Results & Analysis' tab for detailed results.
                    </div>
                """, unsafe_allow_html=True)
                
                # Display quick results
                st.subheader("📊 Quick Results")
                comparison_df = predictor.get_comparison_dataframe()
                st.dataframe(comparison_df, use_container_width=True)
    
    # Tab 3: Results & Analysis
    with tab3:
        st.header("📈 Results & Analysis")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train the models first from the 'Model Training' tab.")
        else:
            visualizer = Visualizer()
            
            # Model comparison
            st.subheader("🏆 Model Performance Comparison")
            comparison_df = st.session_state.predictor.get_comparison_dataframe()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                    use_container_width=True
                )
            
            with col2:
                # Find best model
                best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
                best_accuracy = comparison_df['Accuracy'].max()
                
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>🥇 Best Model</h3>
                        <h2>{best_model}</h2>
                        <p>Accuracy: {best_accuracy:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualization options
            viz_option = st.selectbox(
                "Select Visualization",
                ["Bar Chart Comparison", "Interactive Comparison", "Radar Chart"]
            )
            
            if viz_option == "Bar Chart Comparison":
                fig = visualizer.plot_model_comparison(comparison_df)
                st.pyplot(fig)
            elif viz_option == "Interactive Comparison":
                fig = visualizer.plot_interactive_comparison(comparison_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = visualizer.plot_metrics_radar(comparison_df)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Confusion matrices
            st.subheader("🔍 Confusion Matrices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Random Forest**")
                rf_cm = st.session_state.results['Random Forest']['confusion_matrix']
                fig_rf = visualizer.plot_confusion_matrix(
                    rf_cm, 
                    "Random Forest",
                    st.session_state.target_names
                )
                st.pyplot(fig_rf)
            
            with col2:
                st.markdown("**Support Vector Machine**")
                svm_cm = st.session_state.results['SVM']['confusion_matrix']
                fig_svm = visualizer.plot_confusion_matrix(
                    svm_cm,
                    "SVM",
                    st.session_state.target_names
                )
                st.pyplot(fig_svm)
            
            st.markdown("---")
            
            # Feature importance
            st.subheader(f"🧬 Top {top_n_features} Important Biomarkers")
            
            top_features, top_importances = st.session_state.predictor.get_feature_importance(
                st.session_state.feature_names,
                top_n=top_n_features
            )
            
            fig_importance = visualizer.plot_feature_importance(top_features, top_importances, top_n_features)
            st.pyplot(fig_importance)
            
            # Feature importance table
            st.subheader("📋 Feature Importance Table")
            importance_df = pd.DataFrame({
                'Biomarker': top_features,
                'Importance Score': top_importances
            })
            importance_df['Rank'] = range(1, len(top_features) + 1)
            importance_df = importance_df[['Rank', 'Biomarker', 'Importance Score']]
            
            st.dataframe(importance_df, use_container_width=True)
    
    # Tab 4: Documentation
    with tab4:
        st.header("📋 Documentation")
        
        st.markdown("""
        ## 🎯 Project Overview
        
        This is a complete Machine Learning web application for **Biomarker Prediction** using the Breast Cancer dataset.
        It's designed as a B.Tech final year project demonstration.
        
        ### 🔬 Features
        
        1. **Data Processing**
           - Load default breast cancer dataset or upload custom CSV
           - Automatic preprocessing and feature scaling
           - Train-test split with stratification
        
        2. **Machine Learning Models**
           - Random Forest Classifier
           - Support Vector Machine (SVM)
        
        3. **Evaluation Metrics**
           - Accuracy
           - Precision
           - Recall
           - F1-Score
           - Confusion Matrix
        
        4. **Visualizations**
           - Model comparison charts
           - Confusion matrices
           - Feature importance graphs
           - Interactive plots
        
        ### 📊 Dataset Information
        
        **Default Dataset: Breast Cancer Wisconsin**
        - **Samples:** 569
        - **Features:** 30 biomarkers
        - **Classes:** 2 (Malignant, Benign)
        - **Source:** sklearn.datasets
        
        ### 🚀 How to Use
        
        1. **Load Dataset**
           - Go to "Dataset Overview" tab
           - Choose default dataset or upload custom CSV
           - Click "Load Dataset"
        
        2. **Train Models**
           - Go to "Model Training" tab
           - Configure test size in sidebar
           - Click "Train Models"
        
        3. **View Results**
           - Go to "Results & Analysis" tab
           - Explore model comparisons
           - View confusion matrices
           - Analyze feature importance
        
        ### 🛠️ Technical Stack
        
        - **Python 3.8+**
        - **Streamlit** - Web interface
        - **scikit-learn** - ML models
        - **pandas** - Data manipulation
        - **matplotlib/seaborn** - Visualizations
        - **plotly** - Interactive charts
        
        ### 📦 Installation
        
        ```bash
        pip install -r requirements.txt
        ```
        
        ### ▶️ Run Application
        
        ```bash
        streamlit run app.py
        ```
        
        ### 👨‍💻 Code Structure
        
        - `app.py` - Main Streamlit application
        - `data_processor.py` - Data loading and preprocessing
        - `ml_model.py` - ML model training and evaluation
        - `visualizer.py` - Visualization functions
        - `requirements.txt` - Dependencies
        
        ### 📈 Model Details
        
        **Random Forest Classifier**
        - n_estimators: 100
        - max_depth: 10
        - Provides feature importance
        
        **Support Vector Machine**
        - Kernel: RBF
        - C: 1.0
        - Gamma: scale
        
        ### 🎓 Project Credits
        
        **B.Tech Final Year Project**  
        **Domain:** Machine Learning & Bioinformatics  
        **Application:** Cancer Biomarker Prediction
        
        ---
        
        ### 📞 Support
        
        For questions or issues, please refer to the code comments or contact the development team.
        """)


if __name__ == "__main__":
    main()
