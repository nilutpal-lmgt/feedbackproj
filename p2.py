import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
from openai import OpenAI

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AI Feedback Agent", layout="wide")
st.title("📊 Automated Customer Feedback & Response Agent")
st.subheader("Syllabus Integration: Data Analytics -> Classical ML -> GenAI Agents")

# 2. SIMULATED DATASET & TRAINED CLASSICAL ML MODEL (Syllabus Section 2 & 3)
# To save time, we train a lightweight Classical ML model right when the app starts
@st.cache_resource
def train_classical_ml_model():
    # Sample training data for sentiment
    data = {
        'text': [
            'I love this product, it is amazing!', 'Excellent customer service, very happy',
            'Great experience overall, highly recommend', 'This is the best purchase I have made',
            'Horrible quality, it broke on the first day', 'Terrible customer support, wasted my money',
            'I am very disappointed with the performance', 'Worst experience ever, do not buy'
        ],
        'sentiment': [1, 1, 1, 1, 0, 0, 0, 0] # 1 = Positive, 0 = Negative
    }
    df = pd.DataFrame(data)
    
    # Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X, y)
    return vectorizer, model

vectorizer, classical_model = train_classical_ml_model()

# 3. SIDEBAR CONTROLS
st.sidebar.header("🔑 API Configuration")
openai_key = st.sidebar.text_input("Enter OpenAI API Key", type="password", value="YOUR_OPENAI_API_KEY")

# 4. MAIN INTERFACE
st.write("### Step 1: Input Customer Feedback")
user_review = st.text_area(
    "Paste a customer review below to test the pipeline:",
    value="The product arrived late and the packaging was damaged. However, the customer support team was very helpful and offered a replacement quickly. I am mixed about this."
)

if st.button("🚀 Process Feedback"):
    if not user_review.strip():
        st.warning("Please enter some text first!")
    else:
        # ---- STAGE 1: DATA ANALYTICS & NLP TEXT EXTRACTION ----
        st.write("---")
        st.write("### 🔍 Stage 1: Data Processing & Text Analytics")
        
        char_count = len(user_review)
        word_count = len(user_review.split())
        
        col1, col2 = st.columns(2)
        col1.metric("Character Count", char_count)
        col2.metric("Word Count", word_count)
        
        # ---- STAGE 2: CLASSICAL MACHINE LEARNING (Sentiment Analysis) ----
        st.write("---")
        st.write("### 🤖 Stage 2: Classical ML Sentiment Classification")
        
        # Transform input text using the trained vectorizer
        transformed_review = vectorizer.transform([user_review])
        prediction = classical_model.predict(transformed_review)[0]
        probabilities = classical_model.predict_proba(transformed_review)[0]
        
        sentiment_label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        if sentiment_label == "POSITIVE":
            st.success(f"Predicted Sentiment: **{sentiment_label}** (Confidence: {confidence:.2%})")
        else:
            st.error(f"Predicted Sentiment: **{sentiment_label}** (Confidence: {confidence:.2%})")
            
        # ---- STAGE 3: GENERATIVE AI & AGENTIC RESPONSE ----
        st.write("---")
        st.write("### ✍️ Stage 3: GenAI Agentic Response Generation")
        
        # Define agent's behavior based on the classical ML sentiment outcome
        if sentiment_label == "POSITIVE":
            agent_instruction = "You are an AI Support Agent. Write a warm, grateful thank-you email acknowledging their positive review and offer a 10% discount code (THANKYOU10) for their next purchase."
        else:
            agent_instruction = "You are an AI Escalation Agent. Write an empathetic, professional apology email addressing their negative experience. Assure them a manager will review this, and ask for their order ID to process a full refund."

        st.info(f"**Agent Objective Auto-Assigned:** {agent_instruction}")
        
        # Call OpenAI API to generate the email response
        if openai_key and openai_key != "YOUR_OPENAI_API_KEY":
            try:
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": agent_instruction},
                        {"role": "user", "content": f"Customer Review: {user_review}"}
                    ],
                    max_tokens=250,
                    temperature=0.7
                )
                generated_email = response.choices[0].message.content
                st.write("#### Generated Customer Response Email:")
                st.text_area("Agent Output:", value=generated_email, height=250)
                
            except Exception as e:
                st.error(f"OpenAI API Error: {e}. Showing mock agent response instead.")
                # Fallback static text if API key fails
                st.text_area("Mock Agent Output (API Offline):", value=f"Dear Customer,\n\nThank you for reaching out. We noticed your review was flagged as {sentiment_label}. [Agent would dynamically draft unique response here]...", height=100)
        else:
            st.warning("⚠️ OpenAI API Key missing or invalid in the sidebar. Showing simulated local LLM framework generation below:")
            simulated_response = f"Subject: Regarding your recent experience\n\nDear Customer,\n\nWe have automatically processed your review through our internal triage agent. Because our ML module classified your experience as {sentiment_label}, our system is issuing a specialized follow-up workflow.\n\nThank you for helping us improve.\n\nBest regards,\nCustomer Operations AI"
            st.text_area("Simulated Agent Output:", value=simulated_response, height=150)