# HARA_Automator

## Overview

Automated HARA (Hazard And Risk Assessment) system for automotive safety engineering. Automates Steps 3 and 4 of the ISO 26262 process using LLM analysis and rule-based ASIL calculations.

## What is HARA?

HARA systematically identifies hazards from item malfunctions and evaluates their risks. The ISO 26262 process has 5 steps:
1. Identify hazards
2. Identify operating scenarios  
3. **Determine hazardous events** *(automated)*
4. **Evaluate risk (ASIL)** *(automated)*
5. Determine safety goals

## Why Automation?

- Complete HARA involves thousands of scenario combinations
- Shortage of qualified safety engineers
- Ensures consistent, compliant analysis

## Design Rationale: Few-Shot Learning vs Fine-Tuning

**Why Few-Shot Learning?**

Given the limited training data (38 rows), few-shot in-context learning was chosen over model fine-tuning for several technical reasons:

- **Data Requirements**: Fine-tuning typically requires hundreds to thousands of examples for effective adaptation
- **Limited Dataset**: With only 38 validated HARA examples, fine-tuning would likely lead to overfitting
- **Domain Complexity**: HARA analysis requires understanding complex safety engineering concepts that benefit from explicit examples in context
- **Rapid Deployment**: Few-shot learning enables immediate application without extensive training infrastructure
- **Interpretability**: In-context examples provide transparency in how the model reasons about each scenario

**Technical Advantage**: Few-shot learning with curated examples leverages the LLM's pre-trained knowledge while providing domain-specific guidance through carefully selected examples.

## Two Implementation Approaches

### Approach 1: Hybrid HARA System (`hara_gpt5.py`)

**Architecture**: LLM Analysis + Rule-Based ASIL Calculation

**Key Features**:
- Uses GPT-5 for complex scenario analysis
- Rule-based ISO 26262 compliant ASIL determination
- Processes operating scenarios to generate hazardous events, severity ratings, and controllability assessments
- Guarantees standard compliance for risk calculations

**Workflow**:
1. **LLM Analysis**: Analyzes scenarios to determine hazardous events, people at risk, ΔV (impact velocity), severity (S), and controllability (C)
2. **Rule-Based ASIL**: Applies official ISO 26262 matrix for ASIL determination using S, E, C values

### Approach 2: RAG-Enhanced Hybrid System (`RAG_hara.py`)

**Architecture**: RAG Retrieval + LLM Analysis + Rule-Based ASIL Calculation

**Key Features**:
- **Semantic Similarity Search**: Uses sentence transformers for intelligent example selection
- **RAG (Retrieval-Augmented Generation)**: Retrieves most relevant training examples instead of using fixed examples
- **Enhanced Accuracy**: Better contextual understanding through similarity-based example selection
- **Configurable Retrieval**: Adjustable top-k parameter for example selection

**Workflow**:
1. **RAG Retrieval**: Finds most similar examples from training data using semantic embeddings
2. **LLM Analysis**: Uses retrieved examples for few-shot learning to analyze new scenarios
3. **Rule-Based ASIL**: Applies ISO 26262 matrix for compliant risk assessment

## Installation

### Prerequisites
```bash
pip install pandas openai scikit-learn numpy openpyxl

# For RAG approach, additionally install:
pip install sentence-transformers
```

### Setup
1. Obtain OpenAI API key from https://platform.openai.com/api-keys
2. Replace the API key in the code:
   ```python
   api_key = "your-openai-api-key-here"
   ```

## File Structure

### Input Files Required
- `examples.xlsx` / `data.xlsx`: Training examples with historical HARA analysis
- `your_data.xlsx`: New scenarios to be analyzed

### Input Data Format

**Training Data Columns**:
- Operating Scenario
- E (Exposure level: E1-E4)
- F/T (Failure/Tolerance)
- Hazard
- Hazardous Event
- Details of Hazardous event
- people at risk
- Δv (impact velocity)
- S (Severity: S0-S3)
- Severity Rational
- C (Controllability: C0-C3)
- Controllability Rational

**Test Data Columns**:
- Operating Scenario
- E (Exposure)
- F/T
- Hazard

## Usage

### Basic Hybrid Approach
```bash
python hara_gpt5.py
```

### RAG-Enhanced Approach
```bash
python RAG_hara.py
```

## Output
Both systems generate Excel files with complete HARA analysis:

* Basic Hybrid: [HARA_OUTPUT.xlsx](HARA_OUTPUT.xlsx)
* RAG-Enhanced: [RAG_HARA_OUTPUT.xlsx](RAG_HARA_OUTPUT.xlsx)

**Generated Columns**:
- Hazardous Event
- Details of Hazardous event
- people at risk
- Δv (impact velocity)
- S (Severity rating)
- Severity Rational
- C (Controllability rating)
- Controllability Rational
- **ASIL** (Risk level: QM, ASIL A/B/C/D)

## Technical Architecture

### Severity Classification (S0-S3)
- **S0**: No Injuries
- **S1**: Light to Moderate Injuries (Front <20 km/h, Side <15 km/h, Pedestrian <10 km/h)
- **S2**: Severe Injuries - Survival Probable (Front 20-40 km/h, Side 15-25 km/h, Pedestrian 10-30 km/h)
- **S3**: Life-Threatening Injuries - Survival Uncertain (Front >40 km/h, Side >25 km/h, Pedestrian >30 km/h)

### Controllability Classification (C0-C3)
- **C0**: Controllable in General (>99% can avoid harm)
- **C1**: Simply Controllable (>99% drivers OR >90% traffic participants)
- **C2**: Normally Controllable (90-99% can avoid harm)
- **C3**: Difficult to Control (<90% can avoid harm)

### ASIL Determination Matrix
Risk levels determined by S×E×C combination:
- **QM**: Quality Management (lowest risk)
- **ASIL A/B/C/D**: Increasing risk levels (ASIL D = highest risk)

## Key Differences

| Feature | Basic Hybrid | RAG-Enhanced |
|---------|-------------|-------------|
| Learning Type | Static Few-Shot In-Context Learning | Dynamic Few-Shot In-Context Learning |
| Example Selection | 5 curated, validated examples | Semantic similarity-based top-5 |
| Accuracy | Good baseline performance | Enhanced through contextually relevant examples |
| Processing | Standard few-shot prompting | RAG + few-shot combination |
| Embedding Model | None | SentenceTransformers |
| Similarity Search | None | Cosine similarity with embeddings |
| Context Relevance | Curated examples may not match scenario | Dynamically matched examples |

## Validation Features

- **Range Validation**: Ensures S (0-3), E (1-4), C (0-3) values are within valid ranges
- **Format Checking**: Validates Δv includes proper units (km/h)
- **Content Validation**: Ensures required text fields meet minimum length requirements
- **ISO 26262 Compliance**: Guarantees standard-compliant ASIL calculations

## Performance Metrics

The system tracks:
- Total scenarios processed
- Successful LLM analyses
- Rule-based ASIL calculations
- Overall success rate
- Processing time with rate limiting

## Safety and Compliance

- Rule-based ASIL calculation ensures ISO 26262 compliance
- Built-in validation and error handling
- Complete audit trail for analysis decisions

## License

For automotive safety engineering applications. Use in compliance with relevant industry standards.
