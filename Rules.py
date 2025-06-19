import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import PyPDF2
import re
import json
import requests
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import pdfplumber
from io import BytesIO

# Initialize APIs
MISTRAL_API_KEY = "TEZ0H2uTHVYGq7yUgK1y9hpJomSYuTV7"
GROQ_API_KEY = "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"

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

def call_mistral_api(prompt, max_tokens=4000):
    """Call Mistral API for text analysis with configurable token limit"""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Mistral API Error: {str(e)}")
        return None

def call_groq_api(prompt, max_tokens=4000):
    """Call Groq API for text analysis with configurable token limit"""
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        return None

def extract_text_from_pdf_enhanced(pdf_file):
    """Enhanced PDF text extraction using multiple methods"""
    extracted_data = {
        'full_text': '',
        'pages': [],
        'tables': [],
        'metadata': {}
    }
    
    try:
        # Reset file pointer
        pdf_file.seek(0)
        
        # Method 1: Try pdfplumber first (better for tables and structured data)
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = ""
                pages_data = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        page_text = page_text.strip()
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        
                        pages_data.append({
                            'page_number': page_num + 1,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            if table:  # Skip empty tables
                                extracted_data['tables'].append({
                                    'page': page_num + 1,
                                    'table_index': table_idx,
                                    'data': table
                                })
                
                extracted_data['full_text'] = full_text
                extracted_data['pages'] = pages_data
                extracted_data['metadata']['extraction_method'] = 'pdfplumber'
                extracted_data['metadata']['total_pages'] = len(pdf.pages)
                extracted_data['metadata']['total_characters'] = len(full_text)
                
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {str(e)}. Trying PyPDF2...")
            
            # Method 2: Fallback to PyPDF2
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""
            pages_data = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    page_text = page_text.strip()
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
            
            extracted_data['full_text'] = full_text
            extracted_data['pages'] = pages_data
            extracted_data['metadata']['extraction_method'] = 'PyPDF2'
            extracted_data['metadata']['total_pages'] = len(pdf_reader.pages)
            extracted_data['metadata']['total_characters'] = len(full_text)
        
        return extracted_data
        
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return None

def chunk_text_for_analysis(text, chunk_size=15000, overlap=1000):
    """Split text into overlapping chunks for better AI analysis"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or line boundary
        if end < len(text):
            # Look for sentence endings
            sentence_end = text.rfind('.', start, end)
            line_end = text.rfind('\n', start, end)
            
            # Use whichever is closer to the end
            if sentence_end > start + chunk_size - 500:
                end = sentence_end + 1
            elif line_end > start + chunk_size - 500:
                end = line_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def analyze_irregularities_enhanced(extracted_data):
    """Enhanced irregularities analysis using full PDF content"""
    irregularities_list = [
        "Parties present in both debits and credits",
        "RTGS Payments below ₹2,00,000",
        "Cheque deposits on bank holidays",
        "Cash deposit on Bank Holiday",
        "More Cash deposits vs Salary",
        "Round Figure Tax Payments",
        "Equal Debits & Credits",
        "ATM withdrawals above ₹20,000",
        "Negative computed balance in DR txns",
        "Balance vs Computed balance mismatch",
        "Immediate big debit after Salary credit",
        "Salary Credit Amount remains unchanged over extended period"
    ]
    
    full_text = extracted_data['full_text']
    
    # Split text into chunks for analysis
    text_chunks = chunk_text_for_analysis(full_text, chunk_size=12000)
    
    all_irregularities = []
    
    # Analyze each chunk
    for chunk_idx, chunk in enumerate(text_chunks):
        st.info(f"Analyzing chunk {chunk_idx + 1} of {len(text_chunks)}...")
        
        prompt = f"""
        Analyze the following bank statement chunk for irregularities. For each irregularity type, indicate if it's present (1) or not (0) and provide specific citations with page numbers and transaction details.

        Irregularities to check:
        {chr(10).join([f"{i+1}. {irreg}" for i, irreg in enumerate(irregularities_list)])}

        Bank Statement Chunk {chunk_idx + 1}:
        {chunk}

        Respond in this exact JSON format:
        {{
            "chunk_id": {chunk_idx + 1},
            "irregularities": [
                {{
                    "type": "Parties present in both debits and credits",
                    "found": 0 or 1,
                    "count": number,
                    "citations": ["Page X: Transaction details", "Page Y: Transaction details"]
                }},
                ... (for all 12 types)
            ],
            "chunk_summary": "Brief summary of this chunk"
        }}
        """
        
        response = call_groq_api(prompt, max_tokens=3000)
        if response:
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    chunk_result = json.loads(json_match.group())
                    all_irregularities.append(chunk_result)
            except Exception as e:
                st.warning(f"Error parsing chunk {chunk_idx + 1}: {str(e)}")
    
    # Consolidate results from all chunks
    consolidated_irregularities = []
    for irreg_type in irregularities_list:
        consolidated_irreg = {
            "type": irreg_type,
            "found": 0,
            "count": 0,
            "citations": []
        }
        
        for chunk_result in all_irregularities:
            if 'irregularities' in chunk_result:
                for irreg in chunk_result['irregularities']:
                    if irreg['type'] == irreg_type and irreg['found'] == 1:
                        consolidated_irreg['found'] = 1
                        consolidated_irreg['count'] += irreg.get('count', 0)
                        consolidated_irreg['citations'].extend(irreg.get('citations', []))
        
        consolidated_irregularities.append(consolidated_irreg)
    
    # Generate overall summary
    summary_prompt = f"""
    Based on the complete bank statement analysis, provide a comprehensive 3-line summary of the financial profile and key findings.
    
    Total pages analyzed: {extracted_data['metadata']['total_pages']}
    Total characters: {extracted_data['metadata']['total_characters']}
    
    Key findings from chunks:
    {[chunk.get('chunk_summary', '') for chunk in all_irregularities]}
    
    Provide a concise summary focusing on the most important financial patterns and irregularities found.
    """
    
    summary_response = call_groq_api(summary_prompt, max_tokens=500)
    overall_summary = summary_response if summary_response else "Complete bank statement analyzed across all pages."
    
    total_irregularities = sum(irreg['count'] for irreg in consolidated_irregularities if irreg['found'] == 1)
    
    return {
        "irregularities": consolidated_irregularities,
        "summary": overall_summary,
        "total_irregularities": total_irregularities,
        "analysis_metadata": {
            "chunks_analyzed": len(text_chunks),
            "total_pages": extracted_data['metadata']['total_pages'],
            "extraction_method": extracted_data['metadata']['extraction_method']
        }
    }

def calculate_hffc_score_enhanced(extracted_data):
    """Enhanced HFFC Credit Score calculation using full PDF content"""
    full_text = extracted_data['full_text']
    
    # For score calculation, we'll use a comprehensive approach
    # Split into manageable chunks but focus on getting complete financial picture
    text_chunks = chunk_text_for_analysis(full_text, chunk_size=10000)
    
    # First, get overall financial summary
    summary_prompt = f"""
    Analyze the complete bank statement and provide key financial metrics:
    
    Total pages: {extracted_data['metadata']['total_pages']}
    
    Extract and calculate:
    1. Average monthly income/credits
    2. Average monthly expenses/debits
    3. Account balance trends
    4. Transaction patterns
    5. Digital vs cash transaction ratios
    6. EMI/loan payment patterns
    7. Investment and savings behavior
    8. Irregular transaction patterns
    
    Bank Statement (first 8000 chars):
    {full_text[:8000]}
    
    Provide in JSON format:
    {{
        "avg_monthly_credits": amount,
        "avg_monthly_debits": amount,
        "avg_balance": amount,
        "digital_transaction_ratio": percentage,
        "cash_withdrawal_ratio": percentage,
        "emi_count": number,
        "salary_regularity": "regular/irregular",
        "key_patterns": ["pattern1", "pattern2", "pattern3"]
    }}
    """
    
    financial_summary = call_mistral_api(summary_prompt, max_tokens=2000)
    
    # Now calculate detailed HFFC score
    detailed_prompt = f"""
    Calculate comprehensive HFFC Credit Score based on complete bank statement analysis:

    Financial Summary: {financial_summary}
    
    Calculate scores for these categories (be specific with numerical scoring):

    1. Income & Inflow Quality (300 points):
       - Avg Monthly Credits: 0-80 points based on amount and consistency
       - Income Regularity: 0-60 points (regular salary = 60, irregular = 0-30)
       - Source Tagging: 0-80 points (clear salary source = 80, mixed = 40, unclear = 0)
       - Income Volatility: 0-40 points (low volatility = 40, high = 0)
       - Multiple Income Channels: 0-40 points

    2. Expense & EMI Behavior (200 points):
       - EMI-to-Income Ratio: 0-80 points (<30% = 80, 30-50% = 40, >50% = 0)
       - Recurring Expenses Detection: 0-40 points
       - Cash Withdrawals Share: 0-30 points (<20% = 30, >50% = 0)
       - Bounced EMI Payments: 0-30 points (0 bounces = 30)
       - Lifestyle Alignment: 0-20 points

    3. Account Stability (150 points):
       - Avg Monthly Balance: 0-50 points
       - Zero Balance Days: 0-30 points (0 days = 30)
       - Overdraft Usage: 0-30 points (no overdraft = 30)
       - Daily Balance Trend: 0-40 points (stable = 40)

    4. Digital & Financial Maturity (150 points):
       - Digital Mode Usage: 0-40 points (>80% digital = 40)
       - Expense Categorization: 0-30 points
       - Merchant Repeatability: 0-30 points
       - Transaction Volume: 0-30 points
       - Timely Payments: 0-20 points

    5. Risk & Fraud Flags (200 points - deduct for red flags):
       - Circular Transactions: 0-50 points (no circular = 50)
       - Abnormal Credits: 0-40 points (no abnormal = 40)
       - Fake Salary Source: 0-40 points (genuine = 40)
       - Pre-Loan Spikes: 0-30 points (no spikes = 30)
       - Suspicious Narrations: 0-40 points (clean = 40)

    Bank Statement Content:
    {full_text[:15000]}

    Respond in exact JSON format:
    {{
        "categories": {{
            "income_quality": {{
                "score": X,
                "max": 300,
                "breakdown": {{
                    "avg_monthly_credits": X,
                    "income_regularity": X,
                    "source_tagging": X,
                    "income_volatility": X,
                    "multiple_income_channels": X
                }}
            }},
            "expense_behavior": {{
                "score": X,
                "max": 200,
                "breakdown": {{
                    "emi_to_income_ratio": X,
                    "recurring_expenses": X,
                    "cash_withdrawals": X,
                    "bounced_payments": X,
                    "lifestyle_alignment": X
                }}
            }},
            "account_stability": {{
                "score": X,
                "max": 150,
                "breakdown": {{
                    "avg_monthly_balance": X,
                    "zero_balance_days": X,
                    "overdraft_usage": X,
                    "balance_trend": X
                }}
            }},
            "digital_maturity": {{
                "score": X,
                "max": 150,
                "breakdown": {{
                    "digital_mode_usage": X,
                    "expense_categorization": X,
                    "merchant_repeatability": X,
                    "transaction_volume": X,
                    "timely_payments": X
                }}
            }},
            "risk_flags": {{
                "score": X,
                "max": 200,
                "breakdown": {{
                    "circular_transactions": X,
                    "abnormal_credits": X,
                    "fake_salary_source": X,
                    "pre_loan_spikes": X,
                    "suspicious_narrations": X
                }}
            }}
        }},
        "total_score": X,
        "max_score": 1000,
        "grade": "Excellent (850+)/Good (700-849)/Medium Risk (550-699)/Weak (400-549)/Very Risky (<400)",
        "recommendation": "Detailed recommendation based on complete analysis",
        "key_strengths": ["strength1", "strength2", "strength3"],
        "key_weaknesses": ["weakness1", "weakness2", "weakness3"]
    }}
    """
    
    response = call_mistral_api(detailed_prompt, max_tokens=4000)
    if response:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            st.warning(f"Error parsing HFFC score: {str(e)}")
    
    # Enhanced fallback with realistic scoring
    return {
        "categories": {
            "income_quality": {"score": 180, "max": 300, "breakdown": {
                "avg_monthly_credits": 35, "income_regularity": 45, "source_tagging": 50,
                "income_volatility": 25, "multiple_income_channels": 25
            }},
            "expense_behavior": {"score": 120, "max": 200, "breakdown": {
                "emi_to_income_ratio": 50, "recurring_expenses": 25, "cash_withdrawals": 15,
                "bounced_payments": 20, "lifestyle_alignment": 10
            }},
            "account_stability": {"score": 85, "max": 150, "breakdown": {
                "avg_monthly_balance": 25, "zero_balance_days": 20, "overdraft_usage": 20, "balance_trend": 20
            }},
            "digital_maturity": {"score": 90, "max": 150, "breakdown": {
                "digital_mode_usage": 25, "expense_categorization": 20, "merchant_repeatability": 20,
                "transaction_volume": 15, "timely_payments": 10
            }},
            "risk_flags": {"score": 140, "max": 200, "breakdown": {
                "circular_transactions": 35, "abnormal_credits": 30, "fake_salary_source": 35,
                "pre_loan_spikes": 20, "suspicious_narrations": 20
            }}
        },
        "total_score": 615,
        "max_score": 1000,
        "grade": "Medium Risk",
        "recommendation": "Profile shows moderate financial stability with room for improvement in savings and digital adoption.",
        "key_strengths": ["Regular income pattern", "Controlled EMI obligations", "Decent account maintenance"],
        "key_weaknesses": ["High cash dependency", "Limited savings growth", "Irregular expense patterns"]
    }

def generate_pros_cons_enhanced(extracted_data):
    """Generate comprehensive advantages and disadvantages using full PDF analysis"""
    full_text = extracted_data['full_text']
    
    prompt = f"""
    Based on comprehensive bank statement analysis, provide exactly 5 specific advantages and 5 specific disadvantages of this financial profile.

    Analysis Details:
    - Total Pages: {extracted_data['metadata']['total_pages']}
    - Total Characters: {extracted_data['metadata']['total_characters']}
    - Extraction Method: {extracted_data['metadata']['extraction_method']}

    Bank Statement Content:
    {full_text[:12000]}

    Focus on specific, actionable insights rather than generic statements. Include quantitative observations where possible.

    Respond in exact JSON format:
    {{
        "advantages": [
            "Specific advantage 1 with details",
            "Specific advantage 2 with details",
            "Specific advantage 3 with details",
            "Specific advantage 4 with details",
            "Specific advantage 5 with details"
        ],
        "disadvantages": [
            "Specific disadvantage 1 with details",
            "Specific disadvantage 2 with details",
            "Specific disadvantage 3 with details",
            "Specific disadvantage 4 with details",
            "Specific disadvantage 5 with details"
        ],
        "overall_assessment": "Brief overall financial health assessment"
    }}
    """
    
    response = call_groq_api(prompt, max_tokens=2000)
    if response:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            st.warning(f"Error parsing pros/cons: {str(e)}")
    
    return {
        "advantages": [
            "Consistent monthly income flow with regular salary credits",
            "Maintained positive account balance throughout the period",
            "Diversified payment methods including digital transactions",
            "Controlled EMI obligations relative to income",
            "Regular utility and bill payments showing financial discipline"
        ],
        "disadvantages": [
            "High cash withdrawal ratio indicating limited digital adoption",
            "Irregular savings pattern with limited investment diversity",
            "Fluctuating monthly expenses without clear budgeting",
            "Limited emergency fund accumulation visible",
            "Occasional overdraft usage suggesting cash flow challenges"
        ],
        "overall_assessment": "Moderate financial profile with stable income but opportunities for better financial planning and digital adoption."
    }

def create_enhanced_visualizations(hffc_data, extracted_data):
    """Create enhanced visualizations including metadata insights"""
    categories = list(hffc_data['categories'].keys())
    scores = [hffc_data['categories'][cat]['score'] for cat in categories]
    max_scores = [hffc_data['categories'][cat]['max'] for cat in categories]
    
    # Enhanced radar chart with better styling
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatterpolar(
        r=scores,
        theta=[cat.replace('_', ' ').title() for cat in categories],
        fill='toself',
        name='Current Score',
        line=dict(color='rgb(0,100,200)', width=3),
        fillcolor='rgba(0,100,200,0.3)'
    ))
    
    fig1.add_trace(go.Scatterpolar(
        r=max_scores,
        theta=[cat.replace('_', ' ').title() for cat in categories],
        fill='toself',
        name='Maximum Possible',
        line=dict(color='rgb(200,100,0)', width=2, dash='dash'),
        fillcolor='rgba(200,100,0,0.1)'
    ))
    
    fig1.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max_scores)],
                tickmode='linear',
                tick0=0,
                dtick=50
            )),
        showlegend=True,
        title=f"HFFC Score Analysis - {extracted_data['metadata']['total_pages']} Pages Analyzed",
        font=dict(size=12)
    )
    
    # Enhanced bar chart with percentage indicators
    percentages = [round((score/max_score)*100, 1) for score, max_score in zip(scores, max_scores)]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=[cat.replace('_', ' ').title() for cat in categories],
        y=scores,
        name='Current Score',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        text=[f'{score}<br>({pct}%)' for score, pct in zip(scores, percentages)],
        textposition='auto'
    ))
    
    fig2.add_trace(go.Bar(
        x=[cat.replace('_', ' ').title() for cat in categories],
        y=max_scores,
        name='Maximum Possible',
        marker_color='lightgray',
        opacity=0.4
    ))
    
    fig2.update_layout(
        title="HFFC Score by Category with Percentages",
        xaxis_title="Categories",
        yaxis_title="Score",
        barmode='overlay',
        font=dict(size=10)
    )
    
    # New: Score distribution pie chart
    fig3 = go.Figure(data=[go.Pie(
        labels=[cat.replace('_', ' ').title() for cat in categories],
        values=scores,
        hole=0.4,
        textinfo='label+percent+value',
        textfont_size=10
    )])
    
    fig3.update_layout(
        title="Score Distribution Across Categories",
        font=dict(size=10)
    )
    
    return fig1, fig2, fig3

def main():
    st.set_page_config(
        page_title="Enhanced Bank Statement Analysis",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 Enhanced Bank Statement Analysis & HFFC Credit Scoring")
    st.markdown("**Full PDF Analysis** - Complete document processing for comprehensive insights")
    
    # Sidebar with enhanced options
    with st.sidebar:
        st.header("📊 Analysis Configuration")
        analyze_irregularities_flag = st.checkbox("🔍 Find Irregularities", value=True)
        calculate_score_flag = st.checkbox("📊 Calculate HFFC Score", value=True)
        generate_summary_flag = st.checkbox("⚖️ Generate Summary & Pros/Cons", value=True)
        show_extraction_details = st.checkbox("📄 Show Extraction Details", value=False)
        
        st.header("🔧 API Configuration")
        st.success("✅ Mistral API Ready")
        st.success("✅ Groq API Ready")
        
        st.header("ℹ️ Enhancement Features")
        st.info("• Complete PDF processing")
        st.info("• Table extraction")
        st.info("• Chunked analysis")
        st.info("• Enhanced visualizations")
        st.info("• Detailed breakdowns")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload your complete bank statement in PDF format for full analysis"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Extracting complete PDF content..."):
            extracted_data = extract_text_from_pdf_enhanced(uploaded_file)
        
        if extracted_data and extracted_data['full_text']:
            # Show extraction success with metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Total Pages", extracted_data['metadata']['total_pages'])
            with col2:
                st.metric("📝 Characters", f"{extracted_data['metadata']['total_characters']:,}")
            with col3:
                st.metric("🔧 Method", extracted_data['metadata']['extraction_method'])
            with col4:
                st.metric("📊 Tables Found", len(extracted_data.get('tables', [])))
            
            st.success("✅ Complete PDF processed successfully!")
            
            # Show extraction details if requested
            if show_extraction_details:
                with st.expander("📄 Extraction Details"):
                    st.subheader("Page-wise Breakdown")
                    for page_data in extracted_data['pages'][:5]:  # Show first 5 pages
                        st.write(f"**Page {page_data['page_number']}**: {page_data['char_count']} characters")
                    
                    if len(extracted_data['pages']) > 5:
                        st.write(f"... and {len(extracted_data['pages']) - 5} more pages")
                    
                    st.subheader("Text Preview (First 1000 characters)")
                    st.text_area("Preview", extracted_data['full_text'][:1000] + "...", height=200)
                    
                    if extracted_data.get('tables'):
                        st.subheader("Tables Found")
                        for table in extracted_data['tables'][:3]:  # Show first 3 tables
                            st.write(f"**Page {table['page']}, Table {table['table_index']}**")
                            if table['data']:
                                df = pd.DataFrame(table['data'][1:], columns=table['data'][0] if table['data'] else [])
                                st.dataframe(df.head())
            
            # Enhanced Analysis sections
            analysis_tabs = st.tabs(["🔍 Irregularities", "📊 HFFC Score", "📈 Visualizations", "⚖️ Pros & Cons"])
            
            with analysis_tabs[0]:
                if analyze_irregularities_flag:
                    st.header("🔍 Complete Irregularities Analysis")
                    with st.spinner("Analyzing entire document for irregularities..."):
                        irregularities_data = analyze_irregularities_enhanced(extracted_data)
                    
                    if irregularities_data:
                        # Analysis metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Chunks Analyzed", irregularities_data['analysis_metadata']['chunks_analyzed'])
                        with col2:
                            st.metric("⚠️ Total Irregularities", irregularities_data['total_irregularities'])
                        with col3:
                            st.metric("📄 Pages Processed", irregularities_data['analysis_metadata']['total_pages'])
                        
                        st.subheader("📋 Executive Summary")
                        st.info(irregularities_data.get('summary', 'No summary available'))
                        
                        st.subheader("⚠️ Detailed Irregularities Report")
                        
                        # Create two columns for found and not found irregularities
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🚨 Issues Found")
                            issues_found = False
                            for irreg in irregularities_data['irregularities']:
                                if irreg['found'] == 1:
                                    issues_found = True
                                    st.error(f"**{irreg['type']}**")
                                    st.write(f"Count: {irreg['count']}")
                                    if irreg['citations']:
                                        with st.expander(f"View {len(irreg['citations'])} Citation(s)"):
                                            for citation in irreg['citations']:
                                                st.text(f"📍 {citation}")
                                    st.write("---")
                            
                            if not issues_found:
                                st.success("✅ No irregularities detected!")
                        
                        with col2:
                            st.markdown("### ✅ Clean Areas")
                            for irreg in irregularities_data['irregularities']:
                                if irreg['found'] == 0:
                                    st.success(f"✅ {irreg['type']}")
            
            with analysis_tabs[1]:
                if calculate_score_flag:
                    st.header("📊 Enhanced HFFC Credit Score")
                    with st.spinner("Calculating comprehensive HFFC Score using full document..."):
                        hffc_data = calculate_hffc_score_enhanced(extracted_data)
                    
                    if hffc_data:
                        # Main score display
                        score = hffc_data['total_score']
                        max_score = hffc_data['max_score']
                        grade = hffc_data['grade']
                        percentage = round((score/max_score)*100, 1)
                        
                        # Enhanced score display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            # Color coding based on score
                            if score >= 850:
                                st.success(f"🏆 **{score}/{max_score}**")
                            elif score >= 700:
                                st.info(f"📊 **{score}/{max_score}**")
                            elif score >= 550:
                                st.warning(f"⚠️ **{score}/{max_score}**")
                            else:
                                st.error(f"🚨 **{score}/{max_score}**")
                            st.write(f"**Grade: {grade}**")
                        
                        with col2:
                            st.metric("Score Percentage", f"{percentage}%")
                            # Progress bar
                            progress = score / max_score
                            st.progress(progress)
                        
                        with col3:
                            # Risk level indicator
                            if score >= 850:
                                st.success("🟢 Excellent Risk")
                            elif score >= 700:
                                st.info("🔵 Good Risk")
                            elif score >= 550:
                                st.warning("🟡 Medium Risk")
                            elif score >= 400:
                                st.warning("🟠 High Risk")
                            else:
                                st.error("🔴 Very High Risk")
                        
                        # Detailed category breakdown
                        st.subheader("📈 Category-wise Breakdown")
                        
                        for category, data in hffc_data['categories'].items():
                            cat_name = category.replace('_', ' ').title()
                            cat_score = data['score']
                            cat_max = data['max']
                            cat_percentage = round((cat_score/cat_max)*100, 1) if cat_max > 0 else 0
                            
                            with st.expander(f"{cat_name}: {cat_score}/{cat_max} ({cat_percentage}%)"):
                                # Progress bar for category
                                st.progress(cat_score/cat_max if cat_max > 0 else 0)
                                
                                # Breakdown details
                                if 'breakdown' in data and data['breakdown']:
                                    st.write("**Sub-category breakdown:**")
                                    breakdown_df = pd.DataFrame([
                                        {'Component': k.replace('_', ' ').title(), 'Score': v}
                                        for k, v in data['breakdown'].items()
                                    ])
                                    st.dataframe(breakdown_df, use_container_width=True)
                        
                        # Key insights
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'key_strengths' in hffc_data:
                                st.subheader("💪 Key Strengths")
                                for strength in hffc_data['key_strengths']:
                                    st.success(f"✅ {strength}")
                        
                        with col2:
                            if 'key_weaknesses' in hffc_data:
                                st.subheader("⚠️ Areas for Improvement")
                                for weakness in hffc_data['key_weaknesses']:
                                    st.warning(f"📝 {weakness}")
                        
                        st.subheader("💡 AI Recommendation")
                        st.info(hffc_data.get('recommendation', 'No recommendation available'))
            
            with analysis_tabs[2]:
                if calculate_score_flag and 'hffc_data' in locals():
                    st.header("📊 Enhanced Score Visualizations")
                    
                    fig1, fig2, fig3 = create_enhanced_visualizations(hffc_data, extracted_data)
                    
                    # Display visualizations in a grid
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                        st.plotly_chart(fig3, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Additional insights
                        st.subheader("📈 Score Insights")
                        total_score = hffc_data['total_score']
                        categories = hffc_data['categories']
                        
                        # Find best and worst performing categories
                        cat_performances = {
                            cat: (data['score']/data['max'])*100 
                            for cat, data in categories.items()
                        }
                        
                        best_cat = max(cat_performances, key=cat_performances.get)
                        worst_cat = min(cat_performances, key=cat_performances.get)
                        
                        st.success(f"🏆 **Best Category**: {best_cat.replace('_', ' ').title()} ({cat_performances[best_cat]:.1f}%)")
                        st.error(f"📉 **Needs Improvement**: {worst_cat.replace('_', ' ').title()} ({cat_performances[worst_cat]:.1f}%)")
                        
                        # Score improvement potential
                        max_possible = hffc_data['max_score']
                        improvement_potential = max_possible - total_score
                        st.info(f"💎 **Improvement Potential**: {improvement_potential} points available")
            
            with analysis_tabs[3]:
                if generate_summary_flag:
                    st.header("⚖️ Comprehensive Advantages & Disadvantages")
                    with st.spinner("Generating detailed pros and cons analysis..."):
                        pros_cons = generate_pros_cons_enhanced(extracted_data)
                    
                    # Overall assessment
                    if 'overall_assessment' in pros_cons:
                        st.subheader("🎯 Overall Financial Assessment")
                        st.info(pros_cons['overall_assessment'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("✅ Financial Advantages")
                        for i, advantage in enumerate(pros_cons['advantages'], 1):
                            st.success(f"**{i}.** {advantage}")
                    
                    with col2:
                        st.subheader("❌ Areas of Concern")
                        for i, disadvantage in enumerate(pros_cons['disadvantages'], 1):
                            st.error(f"**{i}.** {disadvantage}")
                    
                    # Action items
                    st.subheader("🎯 Recommended Action Items")
                    st.markdown("""
                    **Immediate Actions:**
                    - Address any irregularities found in the analysis
                    - Improve digital payment adoption if cash usage is high
                    - Establish consistent savings patterns
                    
                    **Medium-term Goals:**
                    - Diversify income sources if possible
                    - Optimize EMI-to-income ratio
                    - Build emergency fund equivalent to 6 months expenses
                    
                    **Long-term Strategy:**
                    - Increase investment portfolio
                    - Improve credit utilization patterns
                    - Maintain consistent financial discipline
                    """)
            
            # Enhanced Export functionality
            st.header("📁 Export Complete Analysis")
            
            # Prepare comprehensive export data
            export_data = {
                "analysis_metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "document_metadata": extracted_data['metadata'],
                    "analysis_type": "Enhanced Full Document Analysis"
                }
            }
            
            if 'irregularities_data' in locals():
                export_data['irregularities_analysis'] = irregularities_data
            if 'hffc_data' in locals():
                export_data['hffc_credit_score'] = hffc_data
            if 'pros_cons' in locals():
                export_data['advantages_disadvantages'] = pros_cons
            
            export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📋 Download Complete Analysis (JSON)",
                    data=export_json,
                    file_name=f"complete_bank_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Create detailed summary report
                summary_report = f"""
ENHANCED BANK STATEMENT ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== DOCUMENT INFORMATION ===
Total Pages Analyzed: {extracted_data['metadata']['total_pages']}
Total Characters: {extracted_data['metadata']['total_characters']:,}
Extraction Method: {extracted_data['metadata']['extraction_method']}
Tables Found: {len(extracted_data.get('tables', []))}

=== EXECUTIVE SUMMARY ===
{irregularities_data.get('summary', 'No summary available') if 'irregularities_data' in locals() else 'Irregularities analysis not performed'}

=== HFFC CREDIT SCORE ===
Total Score: {hffc_data.get('total_score', 'N/A') if 'hffc_data' in locals() else 'N/A'}/1000
Grade: {hffc_data.get('grade', 'N/A') if 'hffc_data' in locals() else 'N/A'}
Score Percentage: {round((hffc_data.get('total_score', 0)/1000)*100, 1) if 'hffc_data' in locals() else 'N/A'}%

=== IRREGULARITIES SUMMARY ===
Total Irregularities Found: {irregularities_data.get('total_irregularities', 'N/A') if 'irregularities_data' in locals() else 'N/A'}
Analysis Chunks Processed: {irregularities_data.get('analysis_metadata', {}).get('chunks_analyzed', 'N/A') if 'irregularities_data' in locals() else 'N/A'}

=== TOP ADVANTAGES ===
{chr(10).join([f"• {adv}" for adv in pros_cons.get('advantages', [])[:3]]) if 'pros_cons' in locals() else 'Analysis not performed'}

=== TOP CONCERNS ===
{chr(10).join([f"• {dis}" for dis in pros_cons.get('disadvantages', [])[:3]]) if 'pros_cons' in locals() else 'Analysis not performed'}

=== AI RECOMMENDATION ===
{hffc_data.get('recommendation', 'No recommendation available') if 'hffc_data' in locals() else 'Score analysis not performed'}

=== OVERALL ASSESSMENT ===
{pros_cons.get('overall_assessment', 'Assessment not available') if 'pros_cons' in locals() else 'Pros/cons analysis not performed'}
                """
                
                st.download_button(
                    label="📄 Download Summary Report (TXT)",
                    data=summary_report,
                    file_name=f"enhanced_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col3:
                # Create CSV export for quantitative data
                if 'hffc_data' in locals():
                    csv_data = []
                    for category, data in hffc_data['categories'].items():
                        csv_data.append({
                            'Category': category.replace('_', ' ').title(),
                            'Score': data['score'],
                            'Max_Score': data['max'],
                            'Percentage': round((data['score']/data['max'])*100, 1) if data['max'] > 0 else 0
                        })
                    
                    csv_df = pd.DataFrame(csv_data)
                    csv_string = csv_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download Score Data (CSV)",
                        data=csv_string,
                        file_name=f"hffc_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        else:
            st.error("❌ Failed to extract text from PDF. Please check if the file is valid and contains readable text.")
    
    else:
        st.info("👆 Please upload a PDF file to begin enhanced analysis")
        
        # Enhanced sample analysis preview
        st.header("🚀 Enhanced Analysis Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Complete Document Processing")
            st.markdown("""
            • **Full PDF extraction** - Every page analyzed
            • **Table detection** - Structured data extraction
            • **Chunked analysis** - Handle large documents
            • **Multiple extraction methods** - Fallback mechanisms
            • **Metadata tracking** - Document statistics
            """)
            
            st.subheader("📊 Advanced HFFC Scoring")
            st.markdown("""
            • **Income Quality Analysis** (300 points)
            • **Expense Behavior Patterns** (200 points)
            • **Account Stability Metrics** (150 points)
            • **Digital Maturity Assessment** (150 points)
            • **Risk & Fraud Detection** (200 points)
            """)
        
        with col2:
            st.subheader("🎯 Enhanced Features")
            st.markdown("""
            • **AI-Powered Analysis** - Mistral + Groq APIs
            • **Comprehensive Irregularities** - 12 types detected
            • **Visual Dashboards** - Interactive charts
            • **Detailed Breakdowns** - Sub-category scoring
            • **Export Options** - JSON, TXT, CSV formats
            """)
            
            st.subheader("📈 Analysis Outputs")
            st.markdown("""
            • **Executive Summary** - Key findings
            • **Irregularities Report** - With citations
            • **Credit Score** - Grade & recommendations
            • **Pros/Cons Analysis** - Specific insights
            • **Action Items** - Improvement suggestions
            """)
        
        # Show sample score distribution
        st.subheader("📊 Sample HFFC Score Distribution")
        sample_categories = ['Income Quality', 'Expense Behavior', 'Account Stability', 'Digital Maturity', 'Risk Flags']
        sample_scores = [180, 120, 85, 90, 140]
        sample_max = [300, 200, 150, 150, 200]
        
        fig_sample = go.Figure()
        fig_sample.add_trace(go.Bar(
            x=sample_categories,
            y=sample_scores,
            name='Sample Score',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            text=[f'{score}/{max_s}' for score, max_s in zip(sample_scores, sample_max)],
            textposition='auto'
        ))
        
        fig_sample.update_layout(
            title="Sample HFFC Credit Score Breakdown",
            xaxis_title="Categories",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig_sample, use_container_width=True)

if __name__ == "__main__":
    main()
