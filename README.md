# H-1B Visa Processing Time Analysis - Milestone 1

## AI-Enabled Visa Status Prediction and Processing Time Estimator

### Project Overview
This project aims to develop a predictive analytics system that estimates visa processing times based on historical H-1B LCA (Labor Condition Application) data from the U.S. Department of Labor.

---

## Milestone 1: Data Collection, Cleaning & EDA

### Objectives
- ✅ Load and explore H-1B LCA disclosure data
- ✅ Perform comprehensive data quality assessment
- ✅ Clean and preprocess the dataset
- ✅ Create target variable (processing time in days)
- ✅ Engineer relevant features
- ✅ Conduct exploratory data analysis with visualizations
- ✅ Generate summary insights and reports

---

## Setup Instructions

### 1. Prerequisites
```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn openpyxl
```

### 2. Download Data
1. Visit: https://www.dol.gov/agencies/eta/foreign-labor/performance (https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2026_Q1.xlsx)
2. Navigate to "OFLC Programs and Disclosures"
3. Download the LCA Disclosure Data (Excel format) for FY2026 Q1
4. Save as: `LCA_Disclosure_Data_FY2026_Q1.xlsx`

### 3. Update File Path
In `visa_data_analysis.py`, update the DATA_PATH:
```python
DATA_PATH = "path/to/your/LCA_Disclosure_Data_FY2026_Q1.xlsx"
```

---

## Running the Analysis

### Option 1: Run Complete Pipeline
```bash
python visa_data_analysis.py
```

### Option 2: Interactive Usage
```python
from visa_data_analysis import VisaDataProcessor

# Initialize
processor = VisaDataProcessor("LCA_Disclosure_Data_FY2026_Q1.xlsx")

# Run complete pipeline
cleaned_data = processor.run_complete_pipeline()

# Or run individual steps
processor.load_data()
processor.explore_basic_info()
processor.check_data_quality()
processor.create_target_variable()
processor.clean_data()
processor.feature_engineering()
processor.exploratory_analysis()
processor.generate_summary_report()
processor.save_cleaned_data()
```

---

## Output Files

### Generated Outputs (in `eda_outputs/` directory):

1. **Visualizations:**
   - `1_processing_time_distribution.png` - Distribution and box plot of processing times
   - `2_case_status_analysis.png` - Case status distribution (bar & pie charts)
   - `3_processing_time_by_status.png` - Processing time comparison by status
   - `4_monthly_trends.png` - Monthly application and processing time trends
   - `5_state_analysis.png` - Top states by volume and processing time
   - `6_visa_class_analysis.png` - Analysis by visa class (H-1B, E-3, H-1B1)
   - `7_employment_type_analysis.png` - Full-time vs part-time comparison
   - `8_correlation_matrix.png` - Feature correlation heatmap

2. **Data Files:**
   - `cleaned_data.csv` - Cleaned dataset ready for modeling
   - `cleaned_data.pkl` - Pickle format for faster loading
   - `SUMMARY_REPORT.txt` - Comprehensive summary report

---

## Pipeline Steps Explained

### Step 1: Data Loading
- Loads the Excel file containing LCA disclosure data
- Displays basic information (rows, columns, memory usage)

### Step 2: Initial Exploration
- Shows dataset shape and structure
- Lists all column names
- Displays data types
- Previews first few records

### Step 3: Data Quality Assessment
- **Missing Values Analysis:** Identifies columns with missing data
- **Duplicate Detection:** Finds and counts duplicate records
- **Distribution Analysis:** Shows case status and visa class distributions

### Step 4: Target Variable Creation
- Calculates `PROCESSING_DAYS = DECISION_DATE - RECEIVED_DATE`
- Converts date columns to datetime format
- Provides statistical summary of processing times
- Identifies outliers using IQR method

### Step 5: Data Cleaning
1. Remove duplicate records
2. Filter valid case statuses (Certified, Denied, Withdrawn, Certified-Withdrawn)
3. Remove invalid processing times (negative, zero, or > 365 days)
4. Clean wage data and remove unrealistic values
5. Standardize categorical variables (uppercase, trim whitespace)
6. Handle missing values in critical columns

### Step 6: Feature Engineering
Creates additional features:
- **Temporal Features:**
  - Year, Month, Quarter
  - Day of week, Week of year
  - Season (Spring, Summer, Fall, Winter)
- **Employment Duration:** Days between BEGIN_DATE and END_DATE
- **Wage Features:** Annualized wage estimates
- **Application Type Indicators:** Binary encoding (Y/N → 1/0)

### Step 7: Exploratory Data Analysis
Generates 8 comprehensive visualizations:
1. Processing time distribution (histogram + box plot)
2. Case status analysis (bar + pie chart)
3. Processing time by case status
4. Monthly trends (applications + processing time)
5. Top states analysis (volume + avg processing time)
6. Visa class comparison
7. Full-time vs part-time analysis
8. Feature correlation matrix

### Step 8: Summary Report
Generates a comprehensive text report with:
- Dataset overview and retention rate
- Processing time statistics (mean, median, percentiles)
- Case status distribution
- Top 10 worksite states
- Temporal insights (fastest/slowest months)
- Data quality metrics

### Step 9: Save Cleaned Data
- Saves cleaned dataset as CSV and pickle files
- Ready for use in Milestone 2 (Predictive Modeling)

---

## Key Insights to Look For

### 1. Processing Time Patterns
- Average processing time across all applications
- Variation by case status (Certified vs Denied)
- Seasonal trends (which months are faster/slower)

### 2. Geographic Patterns
- Which states have the most applications
- Which states have faster/slower processing
- Regional variations

### 3. Case Characteristics
- Certification vs denial rates
- Full-time vs part-time position differences
- Visa class variations (H-1B vs E-3 vs H-1B1)

### 4. Temporal Trends
- Monthly and quarterly patterns
- Day of week effects (if any)
- Seasonal variations

---

## Data Dictionary (Key Fields)

### Target Variable
- **PROCESSING_DAYS:** Time from submission to decision (in days)

### Key Features
- **CASE_STATUS:** Final decision (Certified, Denied, Withdrawn)
- **RECEIVED_DATE:** Application submission date
- **DECISION_DATE:** Final decision date
- **VISA_CLASS:** Type of visa (H-1B, E-3, H-1B1)
- **WORKSITE_STATE:** State where work will be performed
- **JOB_TITLE:** Position title
- **SOC_CODE:** Standard Occupational Classification code
- **FULL_TIME_POSITION:** Y/N indicator
- **WAGE_RATE_OF_PAY:** Offered wage
- **PREVAILING_WAGE:** Market rate for the position

### Engineered Features
- **RECEIVED_MONTH, RECEIVED_QUARTER, RECEIVED_YEAR**
- **SEASON:** Spring, Summer, Fall, Winter
- **EMPLOYMENT_DURATION_DAYS:** Contract length
- Application type indicators (NEW_EMPLOYMENT, CONTINUED_EMPLOYMENT, etc.)

---

## File Structure
```
project/
├── visa_data_analysis.py          # Main analysis script
├── README.md                       # This file
├── LCA_Disclosure_Data_FY2026_Q1.xlsx  # Input data (download separately)
└── eda_outputs/                   # Generated outputs
    ├── 1_processing_time_distribution.png
    ├── 2_case_status_analysis.png
    ├── 3_processing_time_by_status.png
    ├── 4_monthly_trends.png
    ├── 5_state_analysis.png
    ├── 6_visa_class_analysis.png
    ├── 7_employment_type_analysis.png
    ├── 8_correlation_matrix.png
    ├── cleaned_data.csv
    ├── cleaned_data.pkl
    └── SUMMARY_REPORT.txt
```
---

## Data Source
U.S. Department of Labor - Office of Foreign Labor Certification
https://www.dol.gov/agencies/eta/foreign-labor/performance
