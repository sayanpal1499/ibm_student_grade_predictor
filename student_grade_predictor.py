"""
Student Grade Prediction System
Predicts final grades based on demographic, social, and academic factors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class StudentGradePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.best_model_name = None
        
    def generate_sample_data(self, n_samples=1000):
        """Generate realistic synthetic student data"""
        np.random.seed(42)
        
        data = {
            # Demographics
            'age': np.random.randint(15, 20, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'urban_rural': np.random.choice(['Urban', 'Rural'], n_samples, p=[0.7, 0.3]),
            
            # Family Background
            'parent_education': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.4, 0.3]),
            'family_income': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.25, 0.5, 0.25]),
            'family_size': np.random.randint(2, 8, n_samples),
            'parent_support': np.random.randint(1, 6, n_samples),  # 1-5 scale
            
            # Academic Factors
            'study_time_hours': np.random.uniform(0.5, 8, n_samples),
            'attendance_percentage': np.random.uniform(50, 100, n_samples),
            'previous_grade': np.random.uniform(40, 95, n_samples),
            'homework_completion': np.random.uniform(40, 100, n_samples),
            'class_participation': np.random.randint(1, 6, n_samples),  # 1-5 scale
            
            # Social Factors
            'extracurricular_activities': np.random.randint(0, 5, n_samples),
            'social_activities_hours': np.random.uniform(0, 15, n_samples),
            'romantic_relationship': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'internet_access': np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2]),
            
            # Health and Lifestyle
            'health_status': np.random.randint(1, 6, n_samples),  # 1-5 scale
            'sleep_hours': np.random.uniform(4, 10, n_samples),
            'travel_time_mins': np.random.choice([15, 30, 45, 60, 90], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate final grade based on logical relationships
        final_grade = (
            df['previous_grade'] * 0.35 +
            df['study_time_hours'] * 2.5 +
            df['attendance_percentage'] * 0.15 +
            df['homework_completion'] * 0.12 +
            df['class_participation'] * 2 +
            df['parent_support'] * 1.5 +
            df['health_status'] * 1.2 +
            (df['parent_education'] == 'High').astype(int) * 3 +
            (df['family_income'] == 'High').astype(int) * 2 +
            (df['internet_access'] == 'Yes').astype(int) * 2 -
            df['social_activities_hours'] * 0.3 -
            (df['travel_time_mins'] / 30) +
            np.random.normal(0, 5, n_samples)  # Add noise
        )
        
        # Normalize to 0-100 scale
        df['final_grade'] = np.clip(final_grade, 0, 100)
        
        return df
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data: encode categorical variables"""
        df_processed = df.copy()
        
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Define models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.01)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'cv_rmse': cv_rmse
            }
            print(f"  Cross-Validation RMSE: {cv_rmse:.2f}")
        
        # Select best model
        self.best_model_name = min(results, key=lambda x: results[x]['cv_rmse'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n✓ Best Model: {self.best_model_name}")
        return results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)
        
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2
            })
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R² Score: {r2:.4f}")
        
        return pd.DataFrame(results)
    
    def plot_results(self, X_test, y_test, results_df):
        """Visualize model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison - RMSE
        ax1 = axes[0, 0]
        results_sorted = results_df.sort_values('RMSE')
        colors = ['#2ecc71' if model == self.best_model_name else '#3498db' 
                 for model in results_sorted['Model']]
        ax1.barh(results_sorted['Model'], results_sorted['RMSE'], color=colors)
        ax1.set_xlabel('RMSE (Lower is Better)')
        ax1.set_title('Model Comparison - Root Mean Squared Error')
        ax1.invert_yaxis()
        
        # 2. Model Comparison - R² Score
        ax2 = axes[0, 1]
        results_sorted = results_df.sort_values('R² Score', ascending=False)
        colors = ['#2ecc71' if model == self.best_model_name else '#3498db' 
                 for model in results_sorted['Model']]
        ax2.barh(results_sorted['Model'], results_sorted['R² Score'], color=colors)
        ax2.set_xlabel('R² Score (Higher is Better)')
        ax2.set_title('Model Comparison - R² Score')
        ax2.invert_yaxis()
        
        # 3. Actual vs Predicted (Best Model)
        ax3 = axes[1, 0]
        y_pred = self.best_model.predict(X_test)
        ax3.scatter(y_test, y_pred, alpha=0.5, s=30)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual Grade')
        ax3.set_ylabel('Predicted Grade')
        ax3.set_title(f'Actual vs Predicted - {self.best_model_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction Error Distribution
        ax4 = axes[1, 1]
        errors = y_test - y_pred
        ax4.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Prediction Error (Actual - Predicted)')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Error Distribution - {self.best_model_name}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'model_performance.png'")
        plt.show()
    
    def feature_importance_analysis(self, X, feature_names):
        """Analyze feature importance using the best model"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_)
        else:
            print("Feature importance not available for this model type")
            return
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Importance')
        plt.title(f'Top 15 Feature Importances - {self.best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved as 'feature_importance.png'")
        plt.show()
        
        return feature_importance_df
    
    def predict_single_student(self, student_data):
        """Predict grade for a single student"""
        student_df = pd.DataFrame([student_data])
        student_processed = self.preprocess_data(student_df, is_training=False)
        
        # Ensure same column order as training
        X_scaled = self.scaler.transform(student_processed)
        
        predicted_grade = self.best_model.predict(X_scaled)[0]
        
        print("\n" + "="*60)
        print("STUDENT GRADE PREDICTION")
        print("="*60)
        print(f"Predicted Final Grade: {predicted_grade:.2f}")
        
        # Risk assessment
        if predicted_grade < 50:
            risk_level = "HIGH RISK - Needs immediate intervention"
            color = "🔴"
        elif predicted_grade < 65:
            risk_level = "MODERATE RISK - Needs extra support"
            color = "🟡"
        else:
            risk_level = "LOW RISK - On track"
            color = "🟢"
        
        print(f"Risk Assessment: {color} {risk_level}")
        
        return predicted_grade

def main():
    print("="*60)
    print("STUDENT GRADE PREDICTION SYSTEM")
    print("="*60)
    
    # Initialize predictor
    predictor = StudentGradePredictor()
    
    # Generate sample data
    print("\n1. Generating sample student data...")
    df = predictor.generate_sample_data(n_samples=1000)
    print(f"   ✓ Generated {len(df)} student records")
    print(f"   ✓ Features: {len(df.columns)-1}")
    
    # Display sample data
    print("\nSample Data (first 5 rows):")
    print(df.head())
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    df_processed = predictor.preprocess_data(df, is_training=True)
    
    # Split features and target
    X = df_processed.drop('final_grade', axis=1)
    y = df_processed['final_grade']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    predictor.scaler.fit(X_train)
    X_train_scaled = predictor.scaler.transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    # Train models
    print("\n3. Training models...")
    train_results = predictor.train_models(X_train_scaled, y_train)
    
    # Evaluate models
    print("\n4. Evaluating models...")
    results_df = predictor.evaluate_models(X_test_scaled, y_test)
    
    # Visualize results
    print("\n5. Generating visualizations...")
    predictor.plot_results(X_test_scaled, y_test, results_df)
    
    # Feature importance
    print("\n6. Analyzing feature importance...")
    feature_importance_df = predictor.feature_importance_analysis(
        X_train_scaled, X.columns.tolist()
    )
    
    # Example prediction
    print("\n7. Example: Predicting grade for a sample student...")
    sample_student = {
        'age': 17,
        'gender': 'F',
        'urban_rural': 'Urban',
        'parent_education': 'High',
        'family_income': 'Medium',
        'family_size': 4,
        'parent_support': 4,
        'study_time_hours': 5.0,
        'attendance_percentage': 85.0,
        'previous_grade': 75.0,
        'homework_completion': 80.0,
        'class_participation': 4,
        'extracurricular_activities': 2,
        'social_activities_hours': 5.0,
        'romantic_relationship': 'No',
        'internet_access': 'Yes',
        'health_status': 4,
        'sleep_hours': 7.5,
        'travel_time_mins': 30
    }
    
    predicted_grade = predictor.predict_single_student(sample_student)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n✓ Best Model: {predictor.best_model_name}")
    print(f"✓ Best Model R² Score: {results_df[results_df['Model'] == predictor.best_model_name]['R² Score'].values[0]:.4f}")
    print(f"✓ Visualizations saved in current directory")
    print("\nThe system is ready to identify at-risk students!")

if __name__ == "__main__":
    main()
