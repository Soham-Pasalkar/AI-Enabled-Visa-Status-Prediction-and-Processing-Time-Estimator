# H-1B Visa Processing Time Analysis — Summary

## Overview

This analysis was performed on the H-1B LCA Disclosure Data (FY2026 Q1) to understand visa processing timelines and identify factors that influence processing duration.

The original dataset contained **83,120 records** and **98 features**. After cleaning and filtering invalid entries, **75,754 records** were retained, and **107 features** were available after feature engineering.

This cleaned dataset will be used for predictive modeling in the next phase.

---

## Processing Time Statistics

Key statistics:

- Mean processing time: **10.91 days**
- Median processing time: **8 days**
- Minimum: **1 day**
- Maximum: **40 days**
- Standard deviation: **9.29 days**

Most applications are processed within **1 to 2 weeks**, indicating relatively fast processing overall.

---

## Case Status Distribution

Breakdown:

- Certified: **98.56%**
- Withdrawn: **0.92%**
- Denied: **0.52%**

The extremely high certification rate shows that most applications are approved.

---

## Geographic Distribution

Top worksite states:

| Rank | State | Applications |
|-----|------|-------------|
| 1 | CA | 14,647 |
| 2 | TX | 11,625 |
| 3 | NY | 5,981 |
| 4 | WA | 5,431 |
| 5 | NJ | 3,553 |

These states represent major technology and employment hubs.

---

## Temporal Trends

Processing time varies depending on application month:

- Fastest month: **December (7.23 days)**
- Slowest month: **September (38.74 days)**

This suggests possible seasonal variation in processing workload.

---

## Employer Analysis

Top employers by application volume include:

- Amazon
- Microsoft
- Google
- Apple
- Meta
- Cognizant
- Infosys

These employers also had extremely high certification rates.

---

## Occupation Analysis

Most applications were concentrated in technology-related roles:

Top occupations:

- Software Developers
- Data Scientists
- IT Project Managers
- System Engineers

These positions also had higher average wages.

---

## Statistical Findings

### Full-Time vs Part-Time

There is a statistically significant difference in processing time:

- Full-time: **10.93 days**
- Part-time: **9.84 days**

However, the difference is relatively small.

---

### Visa Class Impact

Visa class affects processing time slightly, but differences are minor.

---

### Wage vs Processing Time

Correlation coefficient: **0.0104**

This indicates **almost no relationship** between wage and processing speed.

---

## Key Insights

Main conclusions:

- Most visas are processed quickly
- Certification rate is extremely high
- Processing time varies slightly based on:
  - State
  - Employer
  - Month
  - Occupation
- Wage has minimal impact on processing duration

---

## Dataset Readiness

The dataset is now:

- Cleaned
- Feature-engineered
- Ready for predictive modeling

Output files:

