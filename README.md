# Mobile-Tower-energy-Consumption
optimize energy consumption patterns and reduce system downtime for mobile tower infrastructure
**Objective**: Optimize energy consumption patterns and reduce system downtime for mobile tower infrastructure

#### Phase 1: Data Collection & Preparation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('telecom_energy_data.csv')
print(f"Dataset shape: {df.shape}")
print(df.info())
```

#### Phase 2: Exploratory Data Analysis
```python
# Basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize energy consumption patterns
plt.figure(figsize=(15, 10))

# Energy consumption over time
plt.subplot(2, 2, 1)
plt.plot(df['timestamp'], df['energy_consumption'])
plt.title('Energy Consumption Over Time')
plt.xticks(rotation=45)

# Distribution of downtime events
plt.subplot(2, 2, 2)
plt.hist(df['downtime_minutes'], bins=30, alpha=0.7)
plt.title('Distribution of Downtime Events')
plt.xlabel('Downtime (minutes)')

# Correlation heatmap
plt.subplot(2, 2, 3)
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')

# Tower performance by region
plt.subplot(2, 2, 4)
df.groupby('region')['downtime_minutes'].mean().plot(kind='bar')
plt.title('Average Downtime by Region')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('energy_eda.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Phase 3: Time Series Forecasting Model
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Feature engineering
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['month'] = pd.to_datetime(df['timestamp']).dt.month

# Create lag features
for lag in [1, 2, 3, 24, 168]:  # 1h, 2h, 3h, 1day, 1week
    df[f'energy_lag_{lag}'] = df['energy_consumption'].shift(lag)

# Remove rows with NaN values
df = df.dropna()

# Define features and target
features = ['hour', 'day_of_week', 'month', 'temperature', 'humidity', 
           'energy_lag_1', 'energy_lag_2', 'energy_lag_3', 'energy_lag_24', 'energy_lag_168']
target = 'energy_consumption'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='importance', y='feature')
plt.title('Feature Importance - Energy Consumption Prediction')
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Phase 4: Downtime Prediction & Optimization
```python
# Classify high-risk periods (downtime > threshold)
threshold = df['downtime_minutes'].quantile(0.8)
df['high_risk'] = (df['downtime_minutes'] > threshold).astype(int)

# Build classification model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Features for downtime prediction
downtime_features = ['energy_consumption', 'temperature', 'humidity', 'hour', 'day_of_week']
X_downtime = df[downtime_features].fillna(0)
y_downtime = df['high_risk']

X_dt_train, X_dt_test, y_dt_train, y_dt_test = train_test_split(
    X_downtime, y_downtime, test_size=0.2, random_state=42
)

# Train classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_dt_train, y_dt_train)

# Predictions
y_dt_pred = gb_classifier.predict(X_dt_test)

print("Downtime Classification Report:")
print(classification_report(y_dt_test, y_dt_pred))

# Visualization of results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('Actual vs Predicted Energy Consumption')

plt.subplot(2, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(2, 2, 3)
cm = confusion_matrix(y_dt_test, y_dt_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Downtime Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(2, 2, 4)
# Time series of actual vs predicted
sample_size = min(100, len(y_test))
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
plt.plot(y_test.iloc[sample_idx].values, label='Actual', alpha=0.7)
plt.plot(y_pred[sample_idx], label='Predicted', alpha=0.7)
plt.title('Sample: Actual vs Predicted Energy Consumption')
plt.legend()

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Phase 5: Business Impact Analysis
```python
# Calculate cost savings
baseline_downtime = df['downtime_minutes'].mean()
optimized_downtime = baseline_downtime * 0.25  # 75% reduction (40% to 10%)

cost_per_minute = 100  # Assume $100 per minute of downtime
monthly_towers = 1000  # Number of towers

monthly_savings = (baseline_downtime - optimized_downtime) * cost_per_minute * monthly_towers * 30
annual_savings = monthly_savings * 12

print(f"Business Impact Analysis:")
print(f"Baseline average downtime: {baseline_downtime:.2f} minutes")
print(f"Optimized downtime: {optimized_downtime:.2f} minutes")
print(f"Reduction: {((baseline_downtime - optimized_downtime) / baseline_downtime * 100):.1f}%")
print(f"Monthly cost savings: ${monthly_savings:,.2f}")
print(f"Annual cost savings: ${annual_savings:,.2f}")

# Create executive summary dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# KPI Cards
ax1.text(0.5, 0.7, f'{((baseline_downtime - optimized_downtime) / baseline_downtime * 100):.1f}%', 
         ha='center', va='center', fontsize=36, weight='bold', color='green')
ax1.text(0.5, 0.3, 'Downtime Reduction', ha='center', va='center', fontsize=14)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title('Key Performance Indicator', fontsize=16, weight='bold')

# Cost savings
ax2.text(0.5, 0.7, f'${annual_savings/1000000:.1f}M', 
         ha='center', va='center', fontsize=36, weight='bold', color='blue')
ax2.text(0.5, 0.3, 'Annual Savings', ha='center', va='center', fontsize=14)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Financial Impact', fontsize=16, weight='bold')

# Model accuracy
ax3.text(0.5, 0.7, f'{r2:.1%}', 
         ha='center', va='center', fontsize=36, weight='bold', color='orange')
ax3.text(0.5, 0.3, 'Prediction Accuracy', ha='center', va='center', fontsize=14)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Model Performance', fontsize=16, weight='bold')

# ROI calculation
implementation_cost = 500000  # $500k implementation cost
roi = (annual_savings - implementation_cost) / implementation_cost * 100
ax4.text(0.5, 0.7, f'{roi:.0f}%', 
         ha='center', va='center', fontsize=36, weight='bold', color='purple')
ax4.text(0.5, 0.3, 'Return on Investment', ha='center', va='center', fontsize=14)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.set_title('Business ROI', fontsize=16, weight='bold')

plt.suptitle('Mobile Tower Energy Optimization - Executive Summary', fontsize=20, weight='bold')
plt.tight_layout()
plt.savefig('executive_summary.png', dpi=300, bbox_inches='tight')
plt.show()
```

### GitHub Repository Structure:
```
mobile-tower-energy-optimization/
│
├── README.md
├── requirements.txt
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_business_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── visualization.py
├── models/
│   ├── energy_consumption_model.pkl
│   └── downtime_classifier.pkl
├── visualizations/
│   ├── energy_eda.png
│   ├── feature_importance.png
│   ├── model_results.png
│   └── executive_summary.png
└── reports/
    ├── technical_report.md
    └── executive_summary.pdf
```

---

## Project 2: Market Segmentation Analysis for Digital Payments

### Dataset (Kaggle):
- [Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- [Digital Payments India Dataset](https://www.kaggle.com/datasets/digital-payments-india)

### Implementation Steps:

#### Phase 1: Customer Segmentation Analysis
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load and explore data
df = pd.read_csv('credit_card_customers.csv')

# Feature selection for segmentation
segmentation_features = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

# Preprocessing
X = df[segmentation_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal clusters using elbow method
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png', dpi=300)
plt.show()

# Apply K-means clustering
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Segment'] = clusters

# Segment analysis
segment_summary = df.groupby('Segment')[segmentation_features].mean()
print("Segment Characteristics:")
print(segment_summary)
```

#### Phase 2: Segment Profiling & Visualization
```python
# Create comprehensive segment profiles
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Age distribution by segment
sns.boxplot(data=df, x='Segment', y='Customer_Age', ax=axes[0,0])
axes[0,0].set_title('Age Distribution by Segment')

# Credit limit by segment
sns.boxplot(data=df, x='Segment', y='Credit_Limit', ax=axes[0,1])
axes[0,1].set_title('Credit Limit by Segment')

# Transaction amount by segment
sns.boxplot(data=df, x='Segment', y='Total_Trans_Amt', ax=axes[0,2])
axes[0,2].set_title('Transaction Amount by Segment')

# Transaction count by segment
sns.boxplot(data=df, x='Segment', y='Total_Trans_Ct', ax=axes[1,0])
axes[1,0].set_title('Transaction Count by Segment')

# Utilization ratio by segment
sns.boxplot(data=df, x='Segment', y='Avg_Utilization_Ratio', ax=axes[1,1])
axes[1,1].set_title('Utilization Ratio by Segment')

# Segment sizes
segment_counts = df['Segment'].value_counts().sort_index()
axes[1,2].pie(segment_counts.values, labels=[f'Segment {i}' for i in segment_counts.index], 
              autopct='%1.1f%%', startangle=90)
axes[1,2].set_title('Segment Distribution')

plt.tight_layout()
plt.savefig('segment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Phase 3: Growth Opportunity Analysis
```python
# Define segment characteristics and growth potential
segment_profiles = {
    0: {"name": "Conservative Savers", "growth_potential": "Medium", "strategy": "Savings Products"},
    1: {"name": "High-Value Transactors", "growth_potential": "High", "strategy": "Premium Services"},
    2: {"name": "Emerging Users", "growth_potential": "High", "strategy": "Digital Onboarding"},
    3: {"name": "Low-Activity Users", "growth_potential": "Low", "strategy": "Retention Programs"}
}

# Calculate segment metrics
segment_metrics = []
for segment in range(optimal_k):
    segment_data = df[df['Segment'] == segment]
    
    metrics = {
        'Segment': segment,
        'Name': segment_profiles[segment]['name'],
        'Size': len(segment_data),
        'Avg_Credit_Limit': segment_data['Credit_Limit'].mean(),
        'Avg_Transaction_Amount': segment_data['Total_Trans_Amt'].mean(),
        'Avg_Transaction_Count': segment_data['Total_Trans_Ct'].mean(),
        'Avg_Utilization': segment_data['Avg_Utilization_Ratio'].mean(),
        'Growth_Potential': segment_profiles[segment]['growth_potential'],
        'Strategy': segment_profiles[segment]['strategy']
    }
    segment_metrics.append(metrics)

segment_df = pd.DataFrame(segment_metrics)
print("Segment Growth Analysis:")
print(segment_df)

# Create growth opportunity matrix
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate segment value score
segment_df['Value_Score'] = (
    segment_df['Avg_Transaction_Amount'] * 0.4 +
    segment_df['Avg_Transaction_Count'] * 0.3 +
    segment_df['Avg_Credit_Limit'] * 0.2 +
    segment_df['Size'] * 0.1
) / 1000  # Normalize

# Plot segments on growth matrix
colors = ['red', 'green', 'blue', 'orange']
for i, row in segment_df.iterrows():
    growth_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    ax.scatter(row['Value_Score'], growth_mapping[row['Growth_Potential']], 
               s=row['Size']/5, c=colors[i], alpha=0.7, label=row['Name'])
    ax.annotate(row['Name'], (row['Value_Score'], growth_mapping[row['Growth_Potential']]),
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Segment Value Score')
ax.set_ylabel('Growth Potential')
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Low', 'Medium', 'High'])
ax.set_title('Segment Growth Opportunity Matrix')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('growth_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Project 3: International Education Predictive Modeling

### Dataset (Kaggle):
- [Graduate Admissions Dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)
- [University Rankings Dataset](https://www.kaggle.com/datasets/mylesoneill/world-university-rankings)

### Implementation Approach:

#### Phase 1: Admission Probability Prediction
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load admission data
df = pd.read_csv('graduate_admissions.csv')

# Feature engineering
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 
           'LOR', 'CGPA', 'Research']
target = 'Chance of Admit'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'predictions': y_pred
    }

# Compare model performance
results_df = pd.DataFrame(results).T
print("Model Comparison:")
print(results_df[['MAE', 'RMSE', 'R2']])
```

### Complete GitHub Setup Instructions:

#### 1. Repository Structure for Each Project:
```bash
# Create repository
git init project-name
cd project-name

# Create directory structure
mkdir -p {data,notebooks,src,models,visualizations,reports}

# Create initial files
touch README.md requirements.txt .gitignore
```

#### 2. Requirements.txt Template:
```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.15.0
jupyter==1.0.0
statsmodels==0.14.0
tensorflow==2.13.0
xgboost==1.7.6
```

#### 3. README Template for Projects:
```markdown
# Project Name

## Overview
Brief description of the project, business problem, and solution approach.

## Dataset
- **Source**: [Kaggle Dataset Link]
- **Size**: X rows, Y columns
- **Features**: List key features
- **Target**: Target variable description

## Business Impact
- **Problem**: Specific business challenge
- **Solution**: Technical approach taken
- **Results**: Quantified outcomes (e.g., 40% to 10% downtime reduction)
- **Value**: Financial or operational impact

## Technical Implementation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Usage
```python
# Quick start example
python src/main.py
```

### Project Structure
```
├── data/                 # Raw and processed data
├── notebooks/           # Jupyter notebooks
├── src/                # Source code
├── models/             # Trained models
├── visualizations/     # Charts and graphs
└── reports/            # Analysis reports
```

## Key Findings
- Finding 1: Description with supporting data
- Finding 2: Description with supporting data
- Finding 3: Description with supporting data

## Visualizations
![Model Performance](visualizations/model_results.png)
![Business Impact](visualizations/executive_summary.png)

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 85% |
| Precision | 0.87 |
| Recall | 0.83 |
| F1-Score | 0.85 |

## Future Enhancements
- Enhancement 1
- Enhancement 2
- Enhancement 3

## Contact
Sharmitha Vijayakumar - sharmithavijayakumarofficial@gmail.com
```

#### 4. Complete Implementation for Data Quality & Integration Project:

```python
# Data Quality Assessment and Improvement Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataQualityAssessment:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.quality_report = {}
    
    def assess_completeness(self):
        """Check for missing values"""
        missing_stats = self.df.isnull().sum()
        missing_percent = (missing_stats / len(self.df)) * 100
        
        completeness_df = pd.DataFrame({
            'Missing_Count': missing_stats,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        self.quality_report['completeness'] = completeness_df
        return completeness_df
    
    def assess_consistency(self):
        """Check for data consistency issues"""
        consistency_issues = {}
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        consistency_issues['duplicate_rows'] = duplicates
        
        # Check for inconsistent formatting in text columns
        text_columns = self.df.select_dtypes(include=['object']).columns
        for col in text_columns:
            # Check for mixed case
            if self.df[col].dtype == 'object':
                mixed_case = len(self.df[col].unique()) != len(self.df[col].str.lower().unique())
                consistency_issues[f'{col}_mixed_case'] = mixed_case
        
        self.quality_report['consistency'] = consistency_issues
        return consistency_issues
    
    def assess_validity(self):
        """Check for invalid values"""
        validity_issues = {}
        
        # Check numeric columns for outliers
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            validity_issues[f'{col}_outliers'] = outliers
        
        self.quality_report['validity'] = validity_issues
        return validity_issues
    
    def generate_quality_dashboard(self):
        """Create comprehensive data quality dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Missing values heatmap
        sns.heatmap(self.df.isnull(), cbar=True, ax=axes[0,0])
        axes[0,0].set_title('Missing Values Pattern')
        
        # Missing values bar chart
        missing_data = self.quality_report['completeness']
        top_missing = missing_data.head(10)
        top_missing['Missing_Percentage'].plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Top 10 Columns with Missing Data')
        axes[0,1].set_ylabel('Missing Percentage')
        
        # Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        axes[0,2].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0,2].set_title('Data Types Distribution')
        
        # Numeric columns distribution
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]
        for i, col in enumerate(numeric_cols):
            if i < 3:
                self.df[col].hist(bins=30, ax=axes[1,i], alpha=0.7)
                axes[1,i].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.savefig('data_quality_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def clean_data(self):
        """Apply data cleaning procedures"""
        cleaned_df = self.df.copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown', inplace=True)
        
        # Standardize text columns
        for col in categorical_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.title()
        
        return cleaned_df

# Example usage with sample data
# Load your dataset
df_sample = pd.read_csv('sample_dataset.csv')  # Replace with actual dataset

# Create quality assessment instance
qa = DataQualityAssessment(df_sample)

# Run assessments
completeness = qa.assess_completeness()
consistency = qa.assess_consistency()
validity = qa.assess_validity()

# Generate dashboard
qa.generate_quality_dashboard()

# Clean data
cleaned_data = qa.clean_data()

print("Data Quality Improvement Results:")
print(f"Original dataset shape: {df_sample.shape}")
print(f"Cleaned dataset shape: {cleaned_data.shape}")
print(f"Improvement in completeness: {(1 - cleaned_data.isnull().sum().sum() / df_sample.isnull().sum().sum()) * 100:.1f}%")
```

#### 5. Advanced B2B Integration Analytics Project:

```python
# B2B API Integration Performance Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json

class APIIntegrationAnalytics:
    def __init__(self):
        self.integration_data = None
        self.performance_metrics = {}
    
    def simulate_api_integration_data(self, num_clients=50, days=90):
        """Generate realistic API integration performance data"""
        np.random.seed(42)
        
        # Client information
        clients = [f"Client_{i:03d}" for i in range(1, num_clients + 1)]
        api_endpoints = ['auth', 'payment', 'user_data', 'transaction', 'reporting']
        
        data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for client in clients:
            client_tier = np.random.choice(['Enterprise', 'Standard', 'Basic'], 
                                         p=[0.2, 0.5, 0.3])
            
            for day in range(days):
                current_date = base_date + timedelta(days=day)
                
                # Simulate daily API calls for each endpoint
                for endpoint in api_endpoints:
                    # Base call volume depends on client tier
                    if client_tier == 'Enterprise':
                        base_calls = np.random.poisson(1000)
                    elif client_tier == 'Standard':
                        base_calls = np.random.poisson(500)
                    else:
                        base_calls = np.random.poisson(100)
                    
                    # Success rate varies by endpoint
                    success_rates = {
                        'auth': 0.99, 'payment': 0.97, 'user_data': 0.98,
                        'transaction': 0.96, 'reporting': 0.99
                    }
                    
                    successful_calls = np.random.binomial(base_calls, success_rates[endpoint])
                    failed_calls = base_calls - successful_calls
                    
                    # Response times (ms)
                    avg_response_time = np.random.normal(200, 50)
                    avg_response_time = max(50, avg_response_time)  # Min 50ms
                    
                    data.append({
                        'client_id': client,
                        'client_tier': client_tier,
                        'date': current_date,
                        'endpoint': endpoint,
                        'total_calls': base_calls,
                        'successful_calls': successful_calls,
                        'failed_calls': failed_calls,
                        'success_rate': successful_calls / base_calls if base_calls > 0 else 0,
                        'avg_response_time_ms': avg_response_time,
                        'uptime_percentage': np.random.uniform(95, 99.9)
                    })
        
        self.integration_data = pd.DataFrame(data)
        return self.integration_data
    
    def analyze_performance_trends(self):
        """Analyze API performance trends"""
        if self.integration_data is None:
            raise ValueError("No data available. Please load or simulate data first.")
        
        # Daily aggregations
        daily_stats = self.integration_data.groupby('date').agg({
            'total_calls': 'sum',
            'successful_calls': 'sum',
            'failed_calls': 'sum',
            'success_rate': 'mean',
            'avg_response_time_ms': 'mean',
            'uptime_percentage': 'mean'
        }).reset_index()
        
        # Client tier analysis
        tier_stats = self.integration_data.groupby('client_tier').agg({
            'total_calls': 'mean',
            'success_rate': 'mean',
            'avg_response_time_ms': 'mean'
        }).reset_index()
        
        # Endpoint performance
        endpoint_stats = self.integration_data.groupby('endpoint').agg({
            'total_calls': 'sum',
            'success_rate': 'mean',
            'avg_response_time_ms': 'mean',
            'failed_calls': 'sum'
        }).reset_index()
        
        self.performance_metrics = {
            'daily': daily_stats,
            'by_tier': tier_stats,
            'by_endpoint': endpoint_stats
        }
        
        return self.performance_metrics
    
    def create_executive_dashboard(self):
        """Create executive-level dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # KPI Cards (Top row)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        # Calculate KPIs
        total_api_calls = self.integration_data['total_calls'].sum()
        avg_success_rate = self.integration_data['success_rate'].mean()
        avg_response_time = self.integration_data['avg_response_time_ms'].mean()
        total_clients = self.integration_data['client_id'].nunique()
        
        # KPI Card 1: Total API Calls
        ax1.text(0.5, 0.7, f'{total_api_calls:,.0f}', ha='center', va='center', 
                fontsize=24, weight='bold', color='blue')
        ax1.text(0.5, 0.3, 'Total API Calls', ha='center', va='center', fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # KPI Card 2: Success Rate
        ax2.text(0.5, 0.7, f'{avg_success_rate:.1%}', ha='center', va='center', 
                fontsize=24, weight='bold', color='green')
        ax2.text(0.5, 0.3, 'Avg Success Rate', ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # KPI Card 3: Response Time
        ax3.text(0.5, 0.7, f'{avg_response_time:.0f}ms', ha='center', va='center', 
                fontsize=24, weight='bold', color='orange')
        ax3.text(0.5, 0.3, 'Avg Response Time', ha='center', va='center', fontsize=12)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # KPI Card 4: Active Clients
        ax4.text(0.5, 0.7, f'{total_clients}', ha='center', va='center', 
                fontsize=24, weight='bold', color='purple')
        ax4.text(0.5, 0.3, 'Active Clients', ha='center', va='center', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # Daily trends (Second row)
        ax5 = fig.add_subplot(gs[1, :2])
        daily_stats = self.performance_metrics['daily']
        ax5.plot(daily_stats['date'], daily_stats['total_calls'], linewidth=2, color='blue')
        ax5.set_title('Daily API Call Volume', fontsize=14, weight='bold')
        ax5.set_ylabel('Total Calls')
        ax5.tick_params(axis='x', rotation=45)
        
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.plot(daily_stats['date'], daily_stats['success_rate'] * 100, 
                linewidth=2, color='green')
        ax6.set_title('Daily Success Rate Trend', fontsize=14, weight='bold')
        ax6.set_ylabel('Success Rate (%)')
        ax6.tick_params(axis='x', rotation=45)
        
        # Client tier analysis (Third row)
        ax7 = fig.add_subplot(gs[2, :2])
        tier_stats = self.performance_metrics['by_tier']
        bars = ax7.bar(tier_stats['client_tier'], tier_stats['total_calls'])
        ax7.set_title('Average Daily Calls by Client Tier', fontsize=14, weight='bold')
        ax7.set_ylabel('Average Daily Calls')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Endpoint performance (Third row right)
        ax8 = fig.add_subplot(gs[2, 2:])
        endpoint_stats = self.performance_metrics['by_endpoint']
        ax8.scatter(endpoint_stats['avg_response_time_ms'], 
                   endpoint_stats['success_rate'] * 100,
                   s=endpoint_stats['total_calls'] / 1000,
                   alpha=0.6, c=range(len(endpoint_stats)), cmap='viridis')
        
        for i, endpoint in enumerate(endpoint_stats['endpoint']):
            ax8.annotate(endpoint, 
                        (endpoint_stats.iloc[i]['avg_response_time_ms'],
                         endpoint_stats.iloc[i]['success_rate'] * 100))
        
        ax8.set_xlabel('Avg Response Time (ms)')
        ax8.set_ylabel('Success Rate (%)')
        ax8.set_title('Endpoint Performance Matrix', fontsize=14, weight='bold')
        
        # Business impact summary (Bottom row)
        ax9 = fig.add_subplot(gs[3, :])
        
        # Calculate business metrics
        failed_calls_cost = self.integration_data['failed_calls'].sum() * 0.10  # $0.10 per failed call
        optimization_savings = failed_calls_cost * 0.30  # 30% reduction potential
        
        business_text = f"""
        Business Impact Analysis:
        • Total Failed Calls: {self.integration_data['failed_calls'].sum():,.0f}
        • Estimated Cost of Failures: ${failed_calls_cost:,.2f}
        • Potential Monthly Savings: ${optimization_savings:,.2f}
        • Client Satisfaction Score: {avg_success_rate:.1%}
        • Integration Efficiency: {100 - (avg_response_time/10):.1f}%
        """
        
        ax9.text(0.05, 0.5, business_text, fontsize=12, va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')
        ax9.set_title('Business Impact Summary', fontsize=16, weight='bold')
        
        plt.suptitle('B2B API Integration Performance Dashboard', 
                    fontsize=20, weight='bold', y=0.98)
        
        plt.savefig('b2b_integration_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
api_analytics = APIIntegrationAnalytics()
data = api_analytics.simulate_api_integration_data(num_clients=30, days=60)
metrics = api_analytics.analyze_performance_trends()
api_analytics.create_executive_dashboard()
```

#### 6. Deployment and Documentation Best Practices:

```python
# deployment_guide.py
"""
Complete deployment guide for data analytics projects
"""

class ProjectDeployment:
    def __init__(self, project_name):
        self.project_name = project_name
    
    def create_docker_setup(self):
        """Generate Docker configuration"""
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "src/main.py"]
"""
        
        docker_compose = """
version: '3.8'
services:
  analytics-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    volumes:
      - ./data:/app/data
      - ./models:/app/models
"""
        
        return {
            'Dockerfile': dockerfile_content,
            'docker-compose.yml': docker_compose
        }
    
    def generate_api_wrapper(self):
        """Create FastAPI wrapper for model serving"""
        api_code = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict

app = FastAPI(title="Analytics API", version="1.0.0")

# Load trained models
try:
    model = joblib.load('models/trained_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    model = None
    scaler = None

class PredictionRequest(BaseModel):
    features: List[float]
    
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.get("/")
async def root():
    return {"message": "Analytics API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        if scaler:
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Calculate confidence (simplified)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 0.95  # Default confidence
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        return api_code

# Generate complete project structure
def create_complete_project_structure(project_name):
    """Generate complete project with all necessary files"""
    
    structure = {
        f"{project_name}/": {
            "README.md": "# Project documentation here",
            "requirements.txt": "pandas==1.5.3\nnumpy==1.24.3\nscikit-learn==1.3.0",
            ".gitignore": "__pycache__/\n*.pyc\n.env\nmodels/*.pkl\ndata/raw/",
            "setup.py": "from setuptools import setup, find_packages",
            
            "data/": {
                "raw/": {"README.md": "Raw data files"},
                "processed/": {"README.md": "Processed data files"},
            },
            
            "notebooks/": {
                "01_data_exploration.ipynb": "# Data exploration notebook",
                "02_feature_engineering.ipynb": "# Feature engineering notebook",
                "03_model_development.ipynb": "# Model development notebook",
                "04_evaluation.ipynb": "# Model evaluation notebook"
            },
            
            "src/": {
                "__init__.py": "",
                "data_preprocessing.py": "# Data preprocessing functions",
                "feature_engineering.py": "# Feature engineering functions",
                "model_training.py": "# Model training functions",
                "evaluation.py": "# Model evaluation functions",
                "visualization.py": "# Visualization functions",
                "main.py": "# Main execution script"
            },
            
            "models/": {
                "README.md": "Trained models directory"
            },
            
            "tests/": {
                "__init__.py": "",
                "test_data_preprocessing.py": "# Data preprocessing tests",
                "test_model_training.py": "# Model training tests"
            },
            
            "config/": {
                "config.yaml": "# Configuration parameters",
                "logging.conf": "# Logging configuration"
            },
            
            "docs/": {
                "technical_documentation.md": "# Technical documentation",
                "user_guide.md": "# User guide",
                "api_reference.md": "# API reference"
            }
        }
    }
    
    return structure

print("Project structure and deployment guide generated successfully!")
```
