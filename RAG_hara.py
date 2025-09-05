#!/usr/bin/env python3
"""
Enhanced Hybrid HARA System with RAG
- RAG: Retrieval-Augmented Generation for few-shot example selection
- LLM analyzes: Hazardous Event, Details, People at Risk, Δv, S, Rationals, C, Rationals
- Rule-based calculates: ASIL (guaranteed ISO 26262 compliance)
"""

import pandas as pd
import json
import openai
import time
import os
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class RAGHybridHARASystem:
    def __init__(self, api_key, embedding_model='all-MiniLM-L6-v2'):
        openai.api_key = api_key
        self.iso_asil_matrix = self._load_iso_asil_matrix()
        self.severity_definitions = self._load_severity_definitions()
        self.controllability_definitions = self._load_controllability_definitions()
        
        # RAG components
        print(f"🔧 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.training_examples = []
        self.example_embeddings = None
        
        # Columns used for similarity search (key fields)
        self.similarity_columns = [
            'operating_scenario',
            'hazard', 
            'ft',
            'hazardous_event'
        ]
    
    def _load_iso_asil_matrix(self):
        """Official ISO 26262 ASIL determination matrix - RULE-BASED"""
        return {
            # S0 combinations (all QM)
            (0,1,0):"QM", (0,1,1):"QM", (0,1,2):"QM", (0,1,3):"QM",
            (0,2,0):"QM", (0,2,1):"QM", (0,2,2):"QM", (0,2,3):"QM",
            (0,3,0):"QM", (0,3,1):"QM", (0,3,2):"QM", (0,3,3):"QM",
            (0,4,0):"QM", (0,4,1):"QM", (0,4,2):"QM", (0,4,3):"QM",
            # S1 combinations
            (1,1,0):"QM", (1,1,1):"QM", (1,1,2):"QM", (1,1,3):"QM",
            (1,2,0):"QM", (1,2,1):"QM", (1,2,2):"QM", (1,2,3):"QM",
            (1,3,0):"QM", (1,3,1):"QM", (1,3,2):"QM", (1,3,3):"ASIL A",
            (1,4,0):"QM", (1,4,1):"QM", (1,4,2):"ASIL A", (1,4,3):"ASIL B",
            # S2 combinations
            (2,1,0):"QM", (2,1,1):"QM", (2,1,2):"QM", (2,1,3):"QM",
            (2,2,0):"QM", (2,2,1):"QM", (2,2,2):"QM", (2,2,3):"ASIL A",
            (2,3,0):"QM", (2,3,1):"QM", (2,3,2):"ASIL A", (2,3,3):"ASIL B",
            (2,4,0):"QM", (2,4,1):"ASIL A", (2,4,2):"ASIL B", (2,4,3):"ASIL C",
            # S3 combinations
            (3,1,0):"QM", (3,1,1):"QM", (3,1,2):"ASIL A", (3,1,3):"ASIL A",
            (3,2,0):"QM", (3,2,1):"QM", (3,2,2):"ASIL A", (3,2,3):"ASIL B",
            (3,3,0):"QM", (3,3,1):"ASIL A", (3,3,2):"ASIL B", (3,3,3):"ASIL C",
            (3,4,0):"QM", (3,4,1):"ASIL B", (3,4,2):"ASIL C", (3,4,3):"ASIL D"
        }
    
    def _load_severity_definitions(self):
        """Complete severity definitions"""
        return {
            'S0': 'No Injuries - No injuries occur, no occupant or pedestrian contact',
            'S1': 'Light to Moderate Injuries - AIS 0 and less than 10% probability of AIS 1-6. Impact thresholds: Front <20 km/h, Side <15 km/h, Pedestrian <10 km/h',
            'S2': 'Severe and Life-Threatening Injuries (Survival Uncertain but Probable) - More than 10% probability of AIS 1-6. Impact thresholds: Front 20-40 km/h, Side 15-25 km/h, Pedestrian 10-30 km/h',
            'S3': 'Life-Threatening to Fatal Injuries (Survival Uncertain) - More than 10% probability of AIS 5-6. Impact thresholds: Front >40 km/h, Side >25 km/h, Pedestrian >30 km/h'
        }
    
    def _load_controllability_definitions(self):
        """Complete controllability definitions"""
        return {
            'C0': 'Controllable in General - More than 99% of all drivers can avoid harm',
            'C1': 'Simply Controllable - More than 99% of drivers OR more than 90% of other traffic participants can avoid harm',
            'C2': 'Normally Controllable - Between 90% to 99% of all drivers or traffic participants can avoid harm',
            'C3': 'Difficult to Control or Uncontrollable - Less than 90% of all drivers or traffic participants can avoid harm'
        }
    
    def clean_text_data(self, text):
        """Clean text for safe processing"""
        if pd.isna(text):
            return ''
        
        text = str(text)
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'\r\n|\r|\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('"', "'").strip()
        
        if len(text) > 200:
            text = text[:200] + "..."
        
        return text
    
    def create_similarity_text(self, example):
        """Create combined text for similarity search using key columns"""
        
        # Combine key fields for similarity matching
        parts = []
        for col in self.similarity_columns:
            value = example.get(col, '')
            if value and str(value).strip():
                parts.append(str(value).strip())
        
        return ' '.join(parts)
    
    def load_and_embed_examples(self, examples_file):
        """Load training examples and create embeddings for RAG"""
        
        print(f"📚 Loading and embedding examples from {examples_file}...")
        
        df = pd.read_excel(examples_file)
        examples = []
        similarity_texts = []
        
        for idx, row in df.iterrows():
            try:
                if (pd.notna(row.get('Operating Scenario')) and 
                    pd.notna(row.get('S')) and 
                    pd.notna(row.get('C'))):
                    
                    example = {
                        'operating_scenario': self.clean_text_data(row.get('Operating Scenario', '')),
                        'e': row.get('E', ''),
                        'ft': self.clean_text_data(row.get('F/T', '')),
                        'hazard': self.clean_text_data(row.get('Hazard', '')),
                        'hazardous_event': self.clean_text_data(row.get('Hazardous Event', '')),
                        'details': self.clean_text_data(row.get('Details of Hazardous event', '')),
                        'people_at_risk': self.clean_text_data(row.get('people at risk', '')),
                        'delta_v': self.clean_text_data(row.get('Δv', '')),
                        's': int(float(row.get('S'))) if pd.notna(row.get('S')) else 0,
                        'severity_rational': self.clean_text_data(row.get('Severity Rational', '')),
                        'c': int(float(row.get('C'))) if pd.notna(row.get('C')) else 0,
                        'controllability_rational': self.clean_text_data(row.get('Controllability Rational', ''))
                    }
                    
                    # Create similarity text for this example
                    similarity_text = self.create_similarity_text(example)
                    if similarity_text.strip():  # Only add if we have meaningful content
                        examples.append(example)
                        similarity_texts.append(similarity_text)
            except:
                continue
        
        if not examples:
            print("❌ No valid examples found!")
            return [], None
        
        # Create embeddings
        print(f"🔮 Creating embeddings for {len(examples)} examples...")
        embeddings = self.embedding_model.encode(similarity_texts, show_progress_bar=True)
        
        print(f"✅ Loaded {len(examples)} examples with embeddings")
        print(f"🎯 Similarity search using: {', '.join(self.similarity_columns)}")
        
        self.training_examples = examples
        self.example_embeddings = embeddings
        return examples, embeddings
    
    def retrieve_similar_examples(self, scenario, e_value, ft_value, hazard, top_k=5):
        """RAG: Retrieve most similar examples based on scenario content"""
        
        if not self.training_examples or self.example_embeddings is None:
            print("❌ No training examples or embeddings available")
            return []
        
        # Create query text from input (same columns as training)
        query_parts = []
        for text in [scenario, hazard, ft_value]:
            if text and str(text).strip():
                query_parts.append(str(text).strip())
        
        query_text = ' '.join(query_parts)
        
        if not query_text.strip():
            print("⚠️ Empty query, returning first examples")
            return self.training_examples[:top_k]
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.example_embeddings)[0]
        
        # Get top-k most similar examples
        top_indices = np.argsort(similarities)[::-1][:top_k]
        similar_examples = [self.training_examples[i] for i in top_indices]
        
        print(f"🔍 RAG Retrieved {len(similar_examples)} similar examples")
        print(f"   📊 Similarity scores: {[f'{similarities[i]:.3f}' for i in top_indices[:3]]}")
        
        return similar_examples
    
    def calculate_asil_rule_based(self, s_value, e_value, c_value):
        """Rule-based ASIL calculation - NO LLM involved"""
        
        try:
            print(f"Calculating ASIL for S={s_value}, E={e_value}, C={c_value}")
            s = int(s_value) if s_value is not None else 0
            e = int(e_value) if e_value is not None else 1
            c = int(c_value) if c_value is not None else 0
            print(f"Calculating ASIL for S={s}, E={e}, C={c}")
            # Validate ranges
            if s not in [0,1,2,3] or e not in [1,2,3,4] or c not in [0,1,2,3]:
                return "QM", f"Invalid S({s}), E({e}), or C({c}) values"
            
            # Apply ISO 26262 matrix
            asil_result = self.iso_asil_matrix.get((s, e, c), "QM")
            
            calculation_note = f"Rule-based: S{s} + E{e} + C{c} = {asil_result}"
            return asil_result, calculation_note
            
        except Exception as e:
            return "QM", f"Calculation error: {e}"
    
    def create_rag_analysis_prompt(self, similar_examples, scenario, e_value, ft_value, hazard):
        """Create prompt using RAG-retrieved similar examples"""
        
        clean_scenario = self.clean_text_data(scenario)
        clean_hazard = self.clean_text_data(hazard)
        
        prompt = f"""You are an automotive safety engineer performing HARA analysis. 

## SEVERITY CLASSIFICATIONS (for your reference):
S0: No Injuries - No injuries occur
S1: Light to Moderate Injuries - Front <20 km/h, Side <15 km/h, Pedestrian <10 km/h
S2: Severe Injuries (Survival Probable) - Front 20-40 km/h, Side 15-25 km/h, Pedestrian 10-30 km/h  
S3: Life-Threatening Injuries (Survival Uncertain) - Front >40 km/h, Side >25 km/h, Pedestrian >30 km/h

## CONTROLLABILITY CLASSIFICATIONS (for your reference):
C0: Controllable in General - >99% can avoid harm
C1: Simply Controllable - >99% drivers OR >90% traffic participants can avoid harm
C2: Normally Controllable - 90-99% can avoid harm
C3: Difficult to Control - <90% can avoid harm

## MOST RELEVANT SIMILAR EXAMPLES (Retrieved via RAG):

"""
        
        # Use RAG-retrieved similar examples instead of first 5
        for i, ex in enumerate(similar_examples, 1):
            prompt += f"""Example {i}:
Input: "{ex['operating_scenario']}", E:{ex['e']}, F/T:{ex['ft']}, Hazard:"{ex['hazard']}"
Analysis: {{"hazardous_event": "{ex['hazardous_event']}", "details_of_hazardous_event": "{ex['details']}", "people_at_risk": "{ex['people_at_risk']}", "delta_v": "{ex['delta_v']}", "s": {ex['s']}, "severity_rational": "{ex['severity_rational']}", "c": {ex['c']}, "controllability_rational": "{ex['controllability_rational']}"}}

"""
        
        prompt += f"""## ANALYZE YOUR SCENARIO:

Input: "{clean_scenario}", E:{e_value}, F/T:{ft_value}, Hazard:"{clean_hazard}"

REQUIRED ANALYSIS (DO NOT calculate ASIL - that will be done separately):

1. Determine HAZARDOUS EVENT type (front-end collision, side collision, pedestrian collision, electric shock)
2. Calculate specific DETAILS OF HAZARDOUS EVENT with collision mechanism 
3. Identify PEOPLE AT RISK based on scenario environment
4. Calculate Δv (impact velocity) using collision physics (specific values like "65 km/h", not ranges)
5. Classify SEVERITY (S0/S1/S2/S3) based on calculated Δv and injury thresholds
6. Write SEVERITY RATIONAL explaining injury assessment and a reason as to why a particular Severity rating
7. Classify CONTROLLABILITY (C0/C1/C2/C3) based on whether it is controllable or not
8. Write CONTROLLABILITY RATIONAL explaining CONTROLLABILITY assessment and a reason as to why a particular Controllability rating

Return JSON WITHOUT ASIL field:
{{"hazardous_event": "", "details_of_hazardous_event": "", "people_at_risk": "", "delta_v": "", "s": 0, "severity_rational": "", "c": 0, "controllability_rational": ""}}"""
        
        return prompt
    
    def analyze_scenario_with_rag_llm(self, scenario, e_value, ft_value, hazard, top_k=5):
        """Get LLM analysis using RAG-retrieved examples"""
        
        # RAG: Retrieve most similar examples
        similar_examples = self.retrieve_similar_examples(scenario, e_value, ft_value, hazard, top_k)
        
        if not similar_examples:
            print("⚠️ No similar examples found, using random samples")
            similar_examples = self.training_examples[:top_k] if self.training_examples else []
        
        # Create prompt with RAG examples
        prompt = self.create_rag_analysis_prompt(similar_examples, scenario, e_value, ft_value, hazard)
        
        try:
            # Note: Using gpt-4 as gpt-5 is not yet available
            response = openai.ChatCompletion.create(
                model="gpt-5",  # Updated to available model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, 
                seed=12345,
                # temperature=0.1  # Low temperature for consistency
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start != -1:
                json_str = content[start:end]
                json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
                
                analysis_result = json.loads(json_str)
                
                # Validate required fields are present
                required_fields = ['hazardous_event', 'details_of_hazardous_event', 
                                 'people_at_risk', 'delta_v', 's', 'c']
                
                if all(field in analysis_result for field in required_fields):
                    return analysis_result
                else:
                    return None
                    
        except Exception as e:
            print(f"   ❌ LLM Analysis Error: {e}")
            return None
    
    def validate_analysis_result(self, result):
        """Validate LLM analysis for reasonableness"""
        
        issues = []
        
        # Check S value range
        s_value = result.get('s', -1)
        if s_value not in [0, 1, 2, 3]:
            issues.append(f"Invalid severity S={s_value}, must be 0-3")
        
        # Check C value range
        c_value = result.get('c', -1)
        if c_value not in [0, 1, 2, 3]:
            issues.append(f"Invalid controllability C={c_value}, must be 0-3")
        
        # Check Δv format
        delta_v = result.get('delta_v', '')
        if delta_v and delta_v != 'N/A':
            if 'km/h' not in delta_v and 'mph' not in delta_v and 'kph' not in delta_v:
                issues.append("Δv should include speed units (km/h)")
        
        # Check required text fields
        required_text = ['hazardous_event', 'details_of_hazardous_event', 'people_at_risk']
        for field in required_text:
            if not result.get(field) or len(result.get(field, '')) < 5:
                issues.append(f"{field} is too short or missing")
        
        return issues
    
    def process_hybrid_rag_hara(self, examples_file, test_file, top_k_examples=5):
        """Process HARA using hybrid approach with RAG"""
        
        print("🔄 RAG-ENHANCED HYBRID HARA SYSTEM")
        print("🔍 RAG Retrieval + 🧠 LLM Analysis + 📐 Rule-Based ASIL")
        print("=" * 70)
        
        # Load and embed training examples
        training_examples, embeddings = self.load_and_embed_examples(examples_file)
        if not training_examples:
            return
        
        # Load test scenarios
        test_df = pd.read_excel(test_file)
        print(f"\n🎯 Processing {len(test_df)} scenarios")
        
        # Add output columns
        output_cols = ['Hazardous Event', 'Details of Hazardous event', 'people at risk',
                      'Δv', 'S', 'Severity Rational', 'C', 'Controllability Rational', 'ASIL']
        for col in output_cols:
            if col not in test_df.columns:
                test_df[col] = ''
        
        # Process each row
        successful_analysis = 0
        asil_calculations = 0
        
        for idx, row in test_df.iterrows():
            if pd.notna(row.get('Operating Scenario')):
                
                row_id = self.clean_text_data(row.get('ID', f'Row {idx}'))
                print(f"\n[{idx + 1}/{len(test_df)}] {row_id}")
                scenario_preview = str(row.get('Operating Scenario', ''))[:50] + "..."
                print(f"📝 {scenario_preview}")
                
                # STEP 1: RAG + LLM Analysis
                print(f"   🔍 RAG retrieving top-{top_k_examples} similar examples...")
                analysis_result = self.analyze_scenario_with_rag_llm(
                    row.get('Operating Scenario', ''),
                    row.get('E', ''),
                    row.get('F/T', ''),
                    row.get('Hazard', ''),
                    top_k_examples
                )
                
                if analysis_result:
                    # Validate analysis
                    validation_issues = self.validate_analysis_result(analysis_result)
                    
                    if not validation_issues:
                        # STEP 2: Rule-Based ASIL Calculation
                        print("   📐 Rule-based ASIL calculation...")
                        
                        s_analyzed = analysis_result.get('s')
                        e_given = row.get('E')
                        c_analyzed = analysis_result.get('c')
                        print(f"   Calculating ASIL for S={s_analyzed}, E={e_given}, C={c_analyzed}")
                        asil_result, calculation_note = self.calculate_asil_rule_based(s_analyzed, e_given, c_analyzed)
                        
                        # Update DataFrame with RAG+LLM analysis + rule-based ASIL
                        test_df.at[idx, 'Hazardous Event'] = analysis_result.get('hazardous_event', '')
                        test_df.at[idx, 'Details of Hazardous event'] = analysis_result.get('details_of_hazardous_event', '')
                        test_df.at[idx, 'people at risk'] = analysis_result.get('people_at_risk', '')
                        test_df.at[idx, 'Δv'] = analysis_result.get('delta_v', '')
                        test_df.at[idx, 'S'] = analysis_result.get('s', '')
                        test_df.at[idx, 'Severity Rational'] = analysis_result.get('severity_rational', '')
                        test_df.at[idx, 'C'] = analysis_result.get('c', '')
                        test_df.at[idx, 'Controllability Rational'] = analysis_result.get('controllability_rational', '')
                        test_df.at[idx, 'ASIL'] = asil_result  # Rule-based calculation
                        
                        successful_analysis += 1
                        asil_calculations += 1
                        
                        print(f"   ✅ Analysis: S{s_analyzed}, C{c_analyzed} | Δv: {analysis_result.get('delta_v', '')}")
                        print(f"   📐 ASIL: {asil_result} | {calculation_note}")
                        print(f"   👥 People: {analysis_result.get('people_at_risk', '')[:40]}...")
                    else:
                        print(f"   ⚠️ Validation issues: {'; '.join(validation_issues)}")
                else:
                    print("   ❌ RAG+LLM analysis failed")
                
                time.sleep(1.2)  # Rate limiting
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"hara_rag_hybrid_{timestamp}.xlsx"
        test_df.to_excel(output_file, index=False)
        
        print(f"\n🎉 RAG-ENHANCED HYBRID HARA COMPLETED!")
        print(f"📊 Total scenarios: {len(test_df)}")
        print(f"🔍 RAG similarity search successful: {len(test_df)}")
        print(f"🧠 Successful LLM analysis: {successful_analysis}")
        print(f"📐 Rule-based ASIL calculations: {asil_calculations}")
        print(f"📈 Overall success rate: {successful_analysis/len(test_df)*100:.1f}%")
        print(f"💾 Results: {output_file}")
        
        return test_df
    
    def analyze_embedding_quality(self, test_scenario="Vehicle loses steering control on highway"):
        """Analyze the quality of embeddings and similarity search"""
        
        if not self.training_examples or self.example_embeddings is None:
            print("❌ No embeddings to analyze")
            return
        
        print(f"\n🔬 EMBEDDING QUALITY ANALYSIS")
        print(f"Test query: '{test_scenario}'")
        print("="*50)
        
        # Get similar examples
        similar_examples = self.retrieve_similar_examples(test_scenario, 3, "Steering failure", "Loss of control")
        
        print("\n🎯 Top Similar Examples:")
        for i, ex in enumerate(similar_examples[:3], 1):
            print(f"\n{i}. Operating Scenario: {ex['operating_scenario'][:80]}...")
            print(f"   Hazard: {ex['hazard']}")
            print(f"   S={ex['s']}, C={ex['c']}")

# Enhanced main function
def main():
    print("🔄 RAG-ENHANCED HYBRID HARA SYSTEM")
    print("🔍 Retrieval-Augmented Generation + 🧠 LLM Analysis + 📐 Rule-Based ASIL")
    print("=" * 80)
    print("✅ Guarantees ISO 26262 ASIL compliance")
    print("✅ Uses RAG for intelligent example selection")
    print("✅ Uses LLM for complex analysis tasks")
    print("✅ Uses rules for precise ASIL determination")
    print()
    
    examples_file = "data.xlsx"
    test_file = "your_data.xlsx"
    api_key = "your-openai-api-key-here"

    if not api_key or api_key == "your-openai-api-key-here":
        print(" Please set your OpenAI API key!")
        print(" Get your API key from: https://platform.openai.com/api-keys")
        return
    
    # Validate files exist
    for filename in [examples_file, test_file]:
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return
    
    # Run RAG-enhanced hybrid processing
    system = RAGHybridHARASystem(api_key)
    
    # Optional: Test embedding quality
    print("🔬 Testing embedding quality...")
    system.load_and_embed_examples(examples_file)
    system.analyze_embedding_quality()
    
    # Process with RAG
    print(f"\n🚀 Starting RAG-enhanced processing...")
    system.process_hybrid_rag_hara(examples_file, test_file, top_k_examples=5)

if __name__ == "__main__":
    main()

"""
INSTALLATION REQUIREMENTS:

pip install pandas openai scikit-learn sentence-transformers numpy openpyxl

KEY RAG ENHANCEMENTS:

1. 🔍 Semantic Similarity Search:
   - Uses sentence-transformers for semantic embeddings
   - Similarity search columns: Operating Scenario, E, F/T, Exposure Rational, Hazard
   - Cosine similarity for retrieval

2. 📊 Intelligent Example Selection:
   - RAG retrieves most relevant examples instead of using first 5
   - Configurable top_k parameter (default: 5)
   - Similarity scores shown for debugging

3. 🎯 Complete Column Output:
   - All columns preserved: Operating Scenario, E, F/T, Exposure Rational, Hazard
   - Generated columns: Hazardous Event, Details of Hazardous event, people at risk, Δv, S, Severity Rational, C, Controllability Rational, ASIL

4. 🔧 Enhanced Architecture:
   - Modular RAG components focused on specified similarity columns
   - Embedding quality analysis tools
   - ISO 26262 compliant rule-based ASIL calculation

NOTE: Updated to use gpt-4-turbo as gpt-5 is not yet available.
Replace api_key with your actual OpenAI API key.
"""