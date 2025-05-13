import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.preprocessing import StandardScaler
import os
from . import Config

def analyze_feature_importance(df, feature_cols, output_dir=None):
    """
    Analyze feature importance using multiple methods and visualize results
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        output_dir: Directory to save visualizations (optional)
    
    Returns:
        DataFrame with feature importance rankings
    """
    if output_dir is None:
        output_dir = os.path.join(Config.ROOT_DIR, "feature_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df['label'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Method 1: Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    rf_importances = pd.DataFrame({
        'Feature': feature_cols,
        'RF_Importance': rf.feature_importances_
    }).sort_values('RF_Importance', ascending=False)
    
    # Method 2: Mutual Information
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    mi_importances = pd.DataFrame({
        'Feature': feature_cols,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    # Method 3: Recursive Feature Elimination
    rfe = RFE(estimator=rf, n_features_to_select=len(feature_cols))
    rfe.fit(X_scaled, y)
    rfe_importances = pd.DataFrame({
        'Feature': feature_cols,
        'RFE_Rank': rfe.ranking_
    }).sort_values('RFE_Rank')
    
    # Combine all results
    results = rf_importances.merge(mi_importances, on='Feature')
    results = results.merge(rfe_importances, on='Feature')
    
    # Add an overall rank based on average rank across methods
    results['RF_Rank'] = results['RF_Importance'].rank(ascending=False)
    results['MI_Rank'] = results['MI_Score'].rank(ascending=False)
    results['Overall_Rank'] = (results['RF_Rank'] + results['MI_Rank'] + results['RFE_Rank']) / 3
    results = results.sort_values('Overall_Rank')
    
    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='RF_Importance', y='Feature', data=results.sort_values('RF_Importance'))
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rf_importance.png'))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MI_Score', y='Feature', data=results.sort_values('MI_Score'))
    plt.title('Feature Importance (Mutual Information)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_importance.png'))
    
    # Correlation heatmap between features
    plt.figure(figsize=(14, 12))
    sns.heatmap(df[feature_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
    
    return results

def recommend_features(importance_df, threshold=None, num_features=None):
    """
    Recommend which features to keep based on importance analysis
    
    Args:
        importance_df: DataFrame with feature importance results
        threshold: Importance threshold (if provided)
        num_features: Number of top features to keep (if provided)
    
    Returns:
        List of recommended features
    """
    if num_features is not None:
        return importance_df.sort_values('Overall_Rank').head(num_features)['Feature'].tolist()
    elif threshold is not None:
        return importance_df[importance_df['RF_Importance'] > threshold]['Feature'].tolist()
    else:
        # Default: keep features with above-average importance
        avg_importance = importance_df['RF_Importance'].mean()
        return importance_df[importance_df['RF_Importance'] > avg_importance]['Feature'].tolist()
    



def analyze_features():
    # Load the dataframe
    data_file_path = os.path.join(Config.DATA_DIR, "dataframe.csv")
    df = pd.read_csv(data_file_path)
    
    # Get all features (combine all feature lists from config)
    all_features = Config.features + Config.jitter_shimmer_features + Config.spectral_features + Config.speech2text_features
    
    # Run feature importance analysis
    importance_results = analyze_feature_importance(df, all_features)
    
    # Display results
    print("\nFeature Importance Rankings:")
    print(importance_results[['Feature', 'RF_Importance', 'MI_Score', 'RFE_Rank', 'Overall_Rank']])
    
    # Get feature recommendations
    recommended_features = recommend_features(importance_results, num_features=8)
    print("\nRecommended features (top 8):")
    print(recommended_features)
    
    return importance_results, recommended_features


importance_results, recommended_features = analyze_features()
# Optionally use only the recommended features