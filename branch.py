from openai import OpenAI
import streamlit as st
import pandas as pd

# Initialize the client once (make sure your API key is set in environment variables)
openai_key = st.secrets["api_keys"]["openai_key"]
client = OpenAI(api_key=openai_key)

def agentic_ai_branch_analyzer(df: pd.DataFrame, bank_summary: dict) -> str:
    """
    Analyze the bank reviews and return a summary on how to improve the branches.
    """
    # Limit dataframe size
    df_text = df.head(100).to_string()

    prompt = f"""
    You are a strategic financial analyst for BPI Bank.

    Here is a summary of the bank's branch performance in the area:
    {df_text}

    Here is a summary of competitor branches in the area.
    {bank_summary}

    Provide:
    - Key insights from the bank's branch performance data.
    - Recommendations for branch improvements based on customer feedback.
    - From the summary of competitor branches, recommend ways BPI can position itself to be the leading branch in the area.

    Format response in clear bullet points.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"