# %% 1. SETUP & IMPORTS
# Run this cell first to import all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directory
os.makedirs('eda_outputs', exist_ok=True)

print("✓ All libraries imported successfully!")
print(f"  - pandas version: {pd.__version__}")
print(f"  - numpy version: {np.__version__}")
print(f"  - Output directory: eda_outputs/")

# %% 2. DATA LOADING
DATA_PATH = "LCA_Disclosure_Data_FY2026_Q1.xlsx"

print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

try:
    df = pd.read_excel(DATA_PATH)
    print(f"✓ Data loaded successfully")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Total columns: {len(df.columns)}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
except FileNotFoundError:
    print(f"✗ ERROR: File not found at '{DATA_PATH}'")
    print("\nPlease:")
    print("1. Download the LCA disclosure data from DOL website")
    print("2. Update the DATA_PATH variable above")
    print("3. Re-run this cell")
    raise
except Exception as e:
    print(f"✗ Error loading data: {str(e)}")
    raise

# %% 3. INITIAL EXPLORATION
# Display basic information about the dataset

print("\n" + "=" * 80)
print("STEP 2: INITIAL DATA EXPLORATION")
print("=" * 80)

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")

print("\n--- Column Names ---")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

print("\n--- Data Types Summary ---")
print(df.dtypes.value_counts())

print("\n--- First Few Records ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

# %% 4. DATA QUALITY ASSESSMENT
# Check for missing values, duplicates, and data quality issues

print("\n" + "=" * 80)
print("STEP 3: DATA QUALITY ASSESSMENT")
print("=" * 80)

# Missing values analysis
print("\n--- Missing Values Analysis ---")
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
    'Missing_Percentage', ascending=False
)

if len(missing_data) > 0:
    print(missing_data.to_string(index=False))
else:
    print("✓ No missing values found!")

# Duplicate records
print("\n--- Duplicate Records ---")
duplicates = df.duplicated().sum()
print(f"Total duplicate records: {duplicates:,}")
if duplicates > 0:
    print(f"Percentage: {duplicates/len(df)*100:.2f}%")

# Case status distribution
print("\n--- Case Status Distribution ---")
if 'CASE_STATUS' in df.columns:
    print(df['CASE_STATUS'].value_counts())

# Visa class distribution
print("\n--- Visa Class Distribution ---")
if 'VISA_CLASS' in df.columns:
    print(df['VISA_CLASS'].value_counts())

# %% 5. TARGET VARIABLE CREATION
# Calculate processing time in days (DECISION_DATE - RECEIVED_DATE)

print("\n" + "=" * 80)
print("STEP 4: TARGET VARIABLE CREATION")
print("=" * 80)

# Convert date columns to datetime
date_columns = ['RECEIVED_DATE', 'DECISION_DATE', 'BEGIN_DATE', 'END_DATE']

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"✓ Converted {col} to datetime")

# Calculate processing time
if 'RECEIVED_DATE' in df.columns and 'DECISION_DATE' in df.columns:
    df['PROCESSING_DAYS'] = (df['DECISION_DATE'] - df['RECEIVED_DATE']).dt.days
    
    print(f"\n✓ Processing time calculated successfully")
    
    # Statistics
    valid_processing = df['PROCESSING_DAYS'].dropna()
    print(f"\nProcessing Time Statistics:")
    print(f"  - Valid records: {len(valid_processing):,}")
    print(f"  - Mean: {valid_processing.mean():.2f} days")
    print(f"  - Median: {valid_processing.median():.2f} days")
    print(f"  - Std Dev: {valid_processing.std():.2f} days")
    print(f"  - Min: {valid_processing.min():.2f} days")
    print(f"  - Max: {valid_processing.max():.2f} days")
    
    # Identify outliers
    Q1 = valid_processing.quantile(0.25)
    Q3 = valid_processing.quantile(0.75)
    IQR = Q3 - Q1
    outliers = valid_processing[(valid_processing < Q1 - 1.5*IQR) | 
                               (valid_processing > Q3 + 1.5*IQR)]
    print(f"  - Outliers (IQR method): {len(outliers):,} ({len(outliers)/len(valid_processing)*100:.2f}%)")

# %% 6. DATA CLEANING
# Remove duplicates, invalid records, and clean data

print("\n" + "=" * 80)
print("STEP 5: DATA CLEANING")
print("=" * 80)

df_cleaned = df.copy()
initial_count = len(df_cleaned)

# 1. Remove duplicates
print("\n[1/6] Removing duplicate records...")
before_dup = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates()
after_dup = len(df_cleaned)
print(f"  Removed {before_dup - after_dup:,} duplicates")

# 2. Filter valid case statuses
print("\n[2/6] Filtering valid case statuses...")
if 'CASE_STATUS' in df_cleaned.columns:
    valid_statuses = ['Certified', 'Certified-Withdrawn', 'Denied', 'Withdrawn']
    before_status = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['CASE_STATUS'].isin(valid_statuses)]
    after_status = len(df_cleaned)
    print(f"  Kept records with valid status: {after_status:,}")
    print(f"  Removed {before_status - after_status:,} invalid status records")

# 3. Remove records with invalid processing time
print("\n[3/6] Cleaning processing time data...")
if 'PROCESSING_DAYS' in df_cleaned.columns:
    before_proc = len(df_cleaned)
    # Remove negative, zero, or null processing times
    df_cleaned = df_cleaned[
        (df_cleaned['PROCESSING_DAYS'] > 0) & 
        (df_cleaned['PROCESSING_DAYS'].notna())
    ]
    # Remove extreme outliers (> 365 days as unrealistic)
    df_cleaned = df_cleaned[df_cleaned['PROCESSING_DAYS'] <= 365]
    after_proc = len(df_cleaned)
    print(f"  Removed {before_proc - after_proc:,} records with invalid processing time")

# 4. Clean wage data
print("\n[4/6] Cleaning wage data...")
wage_cols = ['WAGE_RATE_OF_PAY_FROM', 'PREVAILING_WAGE']
for col in wage_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# 5. Standardize categorical variables
print("\n[5/6] Standardizing categorical variables...")
categorical_cols = ['EMPLOYER_STATE', 'WORKSITE_STATE', 'SOC_TITLE', 
                  'JOB_TITLE', 'FULL_TIME_POSITION']

for col in categorical_cols:
    if col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].str.strip().str.upper()

# 6. Handle missing values in critical columns
print("\n[6/6] Handling missing values...")
critical_cols = ['EMPLOYER_NAME', 'JOB_TITLE', 'SOC_CODE', 'WORKSITE_STATE']
before_missing = len(df_cleaned)
for col in critical_cols:
    if col in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned[col].notna()]
after_missing = len(df_cleaned)
print(f"  Removed {before_missing - after_missing:,} records with missing critical values")

# Summary
print("\n" + "-" * 80)
print(f"CLEANING SUMMARY:")
print(f"  Initial records: {initial_count:,}")
print(f"  Final records: {len(df_cleaned):,}")
print(f"  Records removed: {initial_count - len(df_cleaned):,} "
      f"({(initial_count - len(df_cleaned))/initial_count*100:.2f}%)")
print("-" * 80)

# %% 7. FEATURE ENGINEERING
# Create additional features for analysis and modeling

print("\n" + "=" * 80)
print("STEP 6: FEATURE ENGINEERING")
print("=" * 80)

# Temporal features from RECEIVED_DATE
if 'RECEIVED_DATE' in df_cleaned.columns:
    print("\n[1/4] Creating temporal features...")
    df_cleaned['RECEIVED_YEAR'] = df_cleaned['RECEIVED_DATE'].dt.year
    df_cleaned['RECEIVED_MONTH'] = df_cleaned['RECEIVED_DATE'].dt.month
    df_cleaned['RECEIVED_QUARTER'] = df_cleaned['RECEIVED_DATE'].dt.quarter
    df_cleaned['RECEIVED_DAY_OF_WEEK'] = df_cleaned['RECEIVED_DATE'].dt.dayofweek
    df_cleaned['RECEIVED_WEEK_OF_YEAR'] = df_cleaned['RECEIVED_DATE'].dt.isocalendar().week
    
    # Season
    df_cleaned['SEASON'] = df_cleaned['RECEIVED_MONTH'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    print("  ✓ Created: Year, Month, Quarter, Day of Week, Week of Year, Season")

# Employment duration
if 'BEGIN_DATE' in df_cleaned.columns and 'END_DATE' in df_cleaned.columns:
    print("\n[2/4] Calculating employment duration...")
    df_cleaned['EMPLOYMENT_DURATION_DAYS'] = (
        df_cleaned['END_DATE'] - df_cleaned['BEGIN_DATE']
    ).dt.days
    print("  ✓ Created: Employment Duration (days)")

# Wage features
print("\n[3/4] Engineering wage features...")
if 'WAGE_RATE_OF_PAY_FROM' in df_cleaned.columns:
    df_cleaned['ANNUAL_WAGE_ESTIMATE'] = df_cleaned['WAGE_RATE_OF_PAY_FROM']

# Application type indicators
print("\n[4/4] Creating application type indicators...")
app_type_cols = ['NEW_EMPLOYMENT', 'CONTINUED_EMPLOYMENT', 
                'CHANGE_PREVIOUS_EMPLOYMENT', 'CHANGE_EMPLOYER', 
                'AMENDED_PETITION', 'NEW_CONCURRENT_EMPLOYMENT']

for col in app_type_cols:
    if col in df_cleaned.columns:
        # Convert Y/N to 1/0 if not already numeric
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].map({'Y': 1, 'N': 0})

print("\n✓ Feature engineering completed!")

# %% 8. EXPLORATORY DATA ANALYSIS - Part 1: Processing Time Distribution
print("\n" + "=" * 80)
print("STEP 7: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\n[1/8] Analyzing processing time distribution...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(df_cleaned['PROCESSING_DAYS'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Processing Days')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Processing Times')
axes[0].axvline(df_cleaned['PROCESSING_DAYS'].mean(), 
               color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df_cleaned["PROCESSING_DAYS"].mean():.1f} days')
axes[0].axvline(df_cleaned['PROCESSING_DAYS'].median(), 
               color='green', linestyle='--', linewidth=2,
               label=f'Median: {df_cleaned["PROCESSING_DAYS"].median():.1f} days')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(df_cleaned['PROCESSING_DAYS'].dropna(), vert=True)
axes[1].set_ylabel('Processing Days')
axes[1].set_title('Processing Time Box Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_outputs/1_processing_time_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 1_processing_time_distribution.png")
plt.show()

# %% 9. EDA - Part 2: Case Status Analysis
if 'CASE_STATUS' in df_cleaned.columns:
    print("\n[2/8] Analyzing case status distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    status_counts = df_cleaned['CASE_STATUS'].value_counts()
    
    # Bar chart
    status_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Case Status')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Case Status Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add count labels
    for i, v in enumerate(status_counts):
        axes[0].text(i, v + max(status_counts)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Pie chart
    axes[1].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Case Status Proportion')
    
    plt.tight_layout()
    plt.savefig('eda_outputs/2_case_status_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 2_case_status_analysis.png")
    plt.show()

# %% 10. EDA - Part 3: Processing Time by Case Status
if 'CASE_STATUS' in df_cleaned.columns:
    print("\n[3/8] Analyzing processing time by case status...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_cleaned.boxplot(column='PROCESSING_DAYS', by='CASE_STATUS', ax=ax)
    ax.set_xlabel('Case Status')
    ax.set_ylabel('Processing Days')
    ax.set_title('Processing Time by Case Status')
    plt.suptitle('')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda_outputs/3_processing_time_by_status.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 3_processing_time_by_status.png")
    plt.show()

# %% 11. EDA - Part 4: Monthly Trends
if 'RECEIVED_MONTH' in df_cleaned.columns:
    print("\n[4/8] Analyzing monthly trends...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Applications per month
    monthly_apps = df_cleaned['RECEIVED_MONTH'].value_counts().sort_index()
    axes[0].bar(monthly_apps.index, monthly_apps.values, color='coral', edgecolor='black')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Number of Applications')
    axes[0].set_title('Applications Received by Month')
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Average processing time by month
    monthly_proc = df_cleaned.groupby('RECEIVED_MONTH')['PROCESSING_DAYS'].mean()
    axes[1].plot(monthly_proc.index, monthly_proc.values, marker='o', 
                linewidth=2, markersize=8, color='darkgreen')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Average Processing Days')
    axes[1].set_title('Average Processing Time by Month')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_outputs/4_monthly_trends.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 4_monthly_trends.png")
    plt.show()

# %% 12. EDA - Part 5: Top States Analysis
if 'WORKSITE_STATE' in df_cleaned.columns:
    print("\n[5/8] Analyzing top worksite states...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top 15 states by application count
    top_states = df_cleaned['WORKSITE_STATE'].value_counts().head(15)
    top_states.plot(kind='barh', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Number of Applications')
    axes[0].set_ylabel('State')
    axes[0].set_title('Top 15 States by Application Volume')
    axes[0].invert_yaxis()
    
    # Average processing time for top 10 states
    top_10_states = df_cleaned['WORKSITE_STATE'].value_counts().head(10).index
    state_proc_time = df_cleaned[
        df_cleaned['WORKSITE_STATE'].isin(top_10_states)
    ].groupby('WORKSITE_STATE')['PROCESSING_DAYS'].mean().sort_values()
    
    state_proc_time.plot(kind='barh', ax=axes[1], color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Average Processing Days')
    axes[1].set_ylabel('State')
    axes[1].set_title('Average Processing Time - Top 10 States')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('eda_outputs/5_state_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 5_state_analysis.png")
    plt.show()

# %% 13. EDA - Part 6: Visa Class Analysis
if 'VISA_CLASS' in df_cleaned.columns:
    print("\n[6/8] Analyzing visa classes...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    visa_counts = df_cleaned['VISA_CLASS'].value_counts()
    
    # Count by visa class
    visa_counts.plot(kind='bar', ax=axes[0], color='purple', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Visa Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Applications by Visa Class')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Processing time by visa class
    df_cleaned.boxplot(column='PROCESSING_DAYS', by='VISA_CLASS', ax=axes[1])
    axes[1].set_xlabel('Visa Class')
    axes[1].set_ylabel('Processing Days')
    axes[1].set_title('Processing Time by Visa Class')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('eda_outputs/6_visa_class_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 6_visa_class_analysis.png")
    plt.show()

# %% 14. EDA - Part 7: Full-time vs Part-time Analysis
if 'FULL_TIME_POSITION' in df_cleaned.columns:
    print("\n[7/8] Analyzing full-time vs part-time positions...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    ft_counts = df_cleaned['FULL_TIME_POSITION'].value_counts()
    
    # Count
    ft_counts.plot(kind='bar', ax=axes[0], color=['green', 'orange'], edgecolor='black')
    axes[0].set_xlabel('Position Type')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Full-time vs Part-time Positions')
    axes[0].set_xticklabels(['Full-time', 'Part-time'], rotation=0)
    
    # Processing time comparison
    df_cleaned.boxplot(column='PROCESSING_DAYS', by='FULL_TIME_POSITION', ax=axes[1])
    axes[1].set_xlabel('Position Type')
    axes[1].set_ylabel('Processing Days')
    axes[1].set_title('Processing Time: Full-time vs Part-time')
    axes[1].set_xticklabels(['Full-time', 'Part-time'])
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('eda_outputs/7_employment_type_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 7_employment_type_analysis.png")
    plt.show()

# %% 15. EDA - Part 8: Correlation Matrix
print("\n[8/8] Creating correlation heatmap...")

# Select numeric columns for correlation
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
# Remove ID or unnecessary columns
numeric_cols = [col for col in numeric_cols if not any(x in col.upper() 
                for x in ['FEIN', 'PHONE', 'POSTAL', 'PHONE_EXT'])]

if len(numeric_cols) > 2:
    # Limit to most relevant columns for readability
    relevant_cols = [col for col in numeric_cols if any(x in col for x in 
                    ['PROCESSING', 'WAGE', 'WORKER', 'DURATION', 'MONTH', 'QUARTER', 'YEAR'])]
    
    if len(relevant_cols) > 2:
        corr_matrix = df_cleaned[relevant_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8},
                   linewidths=0.5)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_outputs/8_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: 8_correlation_matrix.png")
        plt.show()

print("\n✓ All EDA visualizations completed!")
print(f"  Output directory: eda_outputs/")

# %% 16. SUMMARY REPORT GENERATION
print("\n" + "=" * 80)
print("STEP 8: GENERATING SUMMARY REPORT")
print("=" * 80)

report = []
report.append("=" * 80)
report.append("H-1B VISA PROCESSING TIME ANALYSIS - SUMMARY REPORT")
report.append("=" * 80)
report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"\n{'-' * 80}")

# Dataset Overview
report.append("\n1. DATASET OVERVIEW")
report.append(f"   - Total Records (Original): {len(df):,}")
report.append(f"   - Total Records (Cleaned): {len(df_cleaned):,}")
report.append(f"   - Data Retention Rate: {len(df_cleaned)/len(df)*100:.2f}%")
report.append(f"   - Total Features: {len(df_cleaned.columns)}")

# Processing Time Statistics
if 'PROCESSING_DAYS' in df_cleaned.columns:
    report.append("\n2. PROCESSING TIME STATISTICS")
    proc_stats = df_cleaned['PROCESSING_DAYS'].describe()
    report.append(f"   - Mean: {proc_stats['mean']:.2f} days")
    report.append(f"   - Median: {proc_stats['50%']:.2f} days")
    report.append(f"   - Std Dev: {proc_stats['std']:.2f} days")
    report.append(f"   - Min: {proc_stats['min']:.2f} days")
    report.append(f"   - 25th Percentile: {proc_stats['25%']:.2f} days")
    report.append(f"   - 75th Percentile: {proc_stats['75%']:.2f} days")
    report.append(f"   - Max: {proc_stats['max']:.2f} days")

# Case Status Breakdown
if 'CASE_STATUS' in df_cleaned.columns:
    report.append("\n3. CASE STATUS DISTRIBUTION")
    status_dist = df_cleaned['CASE_STATUS'].value_counts()
    for status, count in status_dist.items():
        pct = count / len(df_cleaned) * 100
        report.append(f"   - {status}: {count:,} ({pct:.2f}%)")

# Top States
if 'WORKSITE_STATE' in df_cleaned.columns:
    report.append("\n4. TOP 10 WORKSITE STATES")
    top_states = df_cleaned['WORKSITE_STATE'].value_counts().head(10)
    for i, (state, count) in enumerate(top_states.items(), 1):
        pct = count / len(df_cleaned) * 100
        report.append(f"   {i:2d}. {state}: {count:,} ({pct:.2f}%)")

# Temporal Insights
if 'RECEIVED_MONTH' in df_cleaned.columns:
    report.append("\n5. TEMPORAL INSIGHTS")
    monthly_avg = df_cleaned.groupby('RECEIVED_MONTH')['PROCESSING_DAYS'].mean()
    fastest_month = monthly_avg.idxmin()
    slowest_month = monthly_avg.idxmax()
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    report.append(f"   - Fastest Processing Month: {month_names.get(fastest_month, fastest_month)} ({monthly_avg[fastest_month]:.2f} days)")
    report.append(f"   - Slowest Processing Month: {month_names.get(slowest_month, slowest_month)} ({monthly_avg[slowest_month]:.2f} days)")

# Data Quality
report.append("\n6. DATA QUALITY METRICS")
missing_pct = (df_cleaned.isnull().sum().sum() / 
              (len(df_cleaned) * len(df_cleaned.columns)) * 100)
report.append(f"   - Overall Missing Data: {missing_pct:.2f}%")
report.append(f"   - Completeness Score: {100 - missing_pct:.2f}%")

report.append("\n" + "=" * 80)
report.append("END OF REPORT")
report.append("=" * 80)

# Print and save report
report_text = "\n".join(report)
print(report_text)

with open('eda_outputs/SUMMARY_REPORT.txt', 'w') as f:
    f.write(report_text)

print("\n✓ Summary report saved to: eda_outputs/SUMMARY_REPORT.txt")

# %% 17. SAVE CLEANED DATA
print("\n" + "=" * 80)
print("STEP 9: SAVING CLEANED DATA")
print("=" * 80)

# Save as CSV
output_csv = 'eda_outputs/cleaned_data.csv'
df_cleaned.to_csv(output_csv, index=False)
print(f"✓ Cleaned data saved to: {output_csv}")
print(f"  - Records: {len(df_cleaned):,}")
print(f"  - Features: {len(df_cleaned.columns)}")
print(f"  - File size: {os.path.getsize(output_csv) / 1024**2:.2f} MB")

# Save as pickle for faster loading
output_pkl = 'eda_outputs/cleaned_data.pkl'
df_cleaned.to_pickle(output_pkl)
print(f"✓ Also saved as pickle: {output_pkl}")

# %% 18. QUICK DATA PREVIEW
# Display a quick preview of the cleaned dataset

print("\n" + "=" * 80)
print("CLEANED DATA PREVIEW")
print("=" * 80)

print(f"\nDataset Shape: {df_cleaned.shape}")
print(f"  - Rows: {df_cleaned.shape[0]:,}")
print(f"  - Columns: {df_cleaned.shape[1]}")

print("\n--- First 5 Records ---")
print(df_cleaned.head())

print("\n--- Processing Time Distribution ---")
print(df_cleaned['PROCESSING_DAYS'].describe())

print("\n--- Feature List ---")
print("New features created:")
new_features = ['PROCESSING_DAYS', 'RECEIVED_YEAR', 'RECEIVED_MONTH', 
                'RECEIVED_QUARTER', 'SEASON', 'EMPLOYMENT_DURATION_DAYS']
for feat in new_features:
    if feat in df_cleaned.columns:
        print(f"  ✓ {feat}")

# %% 19. ADVANCED ANALYSIS - Statistical Tests (OPTIONAL)
print("\n" + "=" * 80)
print("STEP 10: ADVANCED STATISTICAL ANALYSIS (OPTIONAL)")
print("=" * 80)

# Test 1: Full-time vs Part-time processing time difference
if 'FULL_TIME_POSITION' in df_cleaned.columns and 'PROCESSING_DAYS' in df_cleaned.columns:
    print("\n[Test 1] Full-time vs Part-time Processing Time (T-Test)")
    ft_times = df_cleaned[df_cleaned['FULL_TIME_POSITION'] == 'Y']['PROCESSING_DAYS'].dropna()
    pt_times = df_cleaned[df_cleaned['FULL_TIME_POSITION'] == 'N']['PROCESSING_DAYS'].dropna()
    
    if len(ft_times) > 0 and len(pt_times) > 0:
        statistic, p_value = stats.ttest_ind(ft_times, pt_times)
        print(f"  Full-time mean: {ft_times.mean():.2f} days (n={len(ft_times):,})")
        print(f"  Part-time mean: {pt_times.mean():.2f} days (n={len(pt_times):,})")
        print(f"  T-statistic: {statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant? {'Yes - There IS a difference' if p_value < 0.05 else 'No - No significant difference'} (α=0.05)")

# Test 2: ANOVA - Processing time across visa classes
if 'VISA_CLASS' in df_cleaned.columns:
    print("\n[Test 2] Processing Time Across Visa Classes (ANOVA)")
    visa_groups = []
    visa_classes = df_cleaned['VISA_CLASS'].unique()
    
    for vc in visa_classes:
        group_data = df_cleaned[df_cleaned['VISA_CLASS'] == vc]['PROCESSING_DAYS'].dropna()
        if len(group_data) > 0:
            visa_groups.append(group_data)
            print(f"  {vc}: mean={group_data.mean():.2f} days (n={len(group_data):,})")
    
    if len(visa_groups) > 1:
        f_stat, p_value = stats.f_oneway(*visa_groups)
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant? {'Yes - Visa classes differ' if p_value < 0.05 else 'No - No significant difference'} (α=0.05)")

# Test 3: Correlation - Wage vs Processing Time
if 'WAGE_RATE_OF_PAY_FROM' in df_cleaned.columns:
    print("\n[Test 3] Correlation: Wage vs Processing Time")
    valid_data = df_cleaned[['WAGE_RATE_OF_PAY_FROM', 'PROCESSING_DAYS']].dropna()
    
    if len(valid_data) > 30:  # Need sufficient data
        correlation, p_value = stats.pearsonr(
            valid_data['WAGE_RATE_OF_PAY_FROM'],
            valid_data['PROCESSING_DAYS']
        )
        print(f"  Pearson correlation: {correlation:.4f}")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            if abs(correlation) < 0.1:
                interpretation = "Significant but very weak"
            elif abs(correlation) < 0.3:
                interpretation = "Significant and weak"
            elif abs(correlation) < 0.5:
                interpretation = "Significant and moderate"
            else:
                interpretation = "Significant and strong"
        else:
            interpretation = "Not significant"
        
        print(f"  Interpretation: {interpretation}")

# %% 20. TOP EMPLOYERS ANALYSIS (OPTIONAL)
print("\n" + "=" * 80)
print("TOP 20 EMPLOYERS ANALYSIS")
print("=" * 80)

if 'EMPLOYER_NAME' in df_cleaned.columns:
    top_employers = df_cleaned['EMPLOYER_NAME'].value_counts().head(20)
    
    employer_stats = []
    for employer in top_employers.index:
        employer_data = df_cleaned[df_cleaned['EMPLOYER_NAME'] == employer]
        
        stats_dict = {
            'Employer': employer[:40],  # Truncate long names
            'Applications': len(employer_data),
            'Avg_Processing': employer_data['PROCESSING_DAYS'].mean(),
            'Cert_Rate_%': (employer_data['CASE_STATUS'] == 'Certified').sum() / len(employer_data) * 100
        }
        employer_stats.append(stats_dict)
    
    employer_df = pd.DataFrame(employer_stats)
    print("\nTop 20 Employers by Application Volume:")
    print(employer_df.to_string(index=False))
    
    # Save to CSV
    employer_df.to_csv('eda_outputs/top_employers_analysis.csv', index=False)
    print("\n✓ Saved: eda_outputs/top_employers_analysis.csv")

# %% 21. TOP OCCUPATIONS ANALYSIS (OPTIONAL)
print("\n" + "=" * 80)
print("TOP 15 OCCUPATIONS ANALYSIS")
print("=" * 80)

if 'SOC_TITLE' in df_cleaned.columns:
    top_occupations = df_cleaned['SOC_TITLE'].value_counts().head(15)
    
    occ_stats = []
    for occupation in top_occupations.index:
        occ_data = df_cleaned[df_cleaned['SOC_TITLE'] == occupation]
        
        stats_dict = {
            'Occupation': occupation[:50],
            'Count': len(occ_data),
            'Avg_Processing': occ_data['PROCESSING_DAYS'].mean(),
            'Cert_Rate_%': (occ_data['CASE_STATUS'] == 'Certified').sum() / len(occ_data) * 100
        }
        
        if 'WAGE_RATE_OF_PAY_FROM' in df_cleaned.columns:
            stats_dict['Avg_Wage'] = occ_data['WAGE_RATE_OF_PAY_FROM'].mean()
        
        occ_stats.append(stats_dict)
    
    occ_df = pd.DataFrame(occ_stats)
    print("\nTop 15 Occupations:")
    print(occ_df.to_string(index=False))
    
    # Save to CSV
    occ_df.to_csv('eda_outputs/top_occupations_analysis.csv', index=False)
    print("\n✓ Saved: eda_outputs/top_occupations_analysis.csv")

