import os
import re
import pandas as pd
import duckdb
import json
import plotly.express as px
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)


SQL_PROMPT = """
You are an expert data analyst. You MUST consider conversation context.

Conversation History:
{history_text}

Convert the user's question into SQL.

Dataset table name is:

data

Available columns:

{columns}

Rules:

1. Return SQL ONLY.
2. No markdown.
3. No explanation.
4. Use DuckDB compatible SQL.
5 If user says "top 3", "only these", "compare them", use previous query context.

Question:

{question}
"""

def generate_sql(question, columns, chat_history=None):

    history_text = ""

    if chat_history:
        for msg in chat_history[-8:]:
            role = msg["role"]
            content = msg["content"]
            sql = msg.get("sql", "")
            if sql and role == "assistant":
                history_text += f"assistant (ran SQL): {sql}\n"
            else:
                history_text += f"{role}: {content}\n"

    prompt = SQL_PROMPT.format(columns=", ".join(columns), 
                               question=question, 
                               history_text=history_text or "None")

    response = model.invoke([HumanMessage(content=prompt)])

    sql = response.content.strip().removeprefix("```sql").removeprefix("```").removeprefix("```").strip()

    return sql


def execute_sql(df, sql_query):

    conn = duckdb.connect()

    conn.register("data", df)

    result = conn.execute(sql_query).df()

    conn.close()

    return result


def fix_sql(bad_sql, error_msg, columns):
    """Ask the model to self correct a broken SQL query"""
    prompt = SQL_PROMPT.format(
        sql=bad_sql,
        error=error_msg,
        columns=", ".join(columns)
    )

    response = model.invoke([HumanMessage(content=prompt)])

    fixed = response.content.strip().removeprefix("```sql").removeprefix("```").removeprefix("```").strip()

    return fixed



def execute_sql_with_retry(df, sql_query, columns, max_retries=2):
    """Try to run SQL; if it fails, ask the model to fix it and retry."""
    last_error = None
    current_sql = sql_query
 
    for attempt in range(max_retries + 1):
        try:
            result = execute_sql(df, current_sql)
            return result, current_sql
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                current_sql = fix_sql(current_sql, last_error, columns)
            else:
                raise RuntimeError(
                    f"SQL failed after {max_retries} fix attempts.\n\n"
                    f"Last SQL:\n{current_sql}\n\nError: {last_error}"
                )



def recommend_chart(df, question):

    columns = df.columns.tolist()
    sample = df.head(10).to_markdown()

    prompt = f"""
        You are a BI visualization expert.

        User question
        {question}

        Result Columns:
        {columns}

        Sample Data:
        {sample}

        Choose the most appropriate chart

        Return ONLY JSON

        Example:

        {{
            "chart_type": "bar",
            "x": "region",
            "y": "revenue" 
        }}

        Allowed chart types:

        - bar
        - line
        - pie
        - scatter
        - histogram

    """
   
    response = model.invoke([HumanMessage(content=prompt)])

    try:
        return json.loads(response.content)

    except Exception:

        return {
            "chart_type": "bar",
            "x": columns[0],
            "y": columns[1]
        }


def generate_kpis(df):

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.3)

    computed = {}

    numeric_columns = df.select_dtypes(include=["number"]).columns

    for col in numeric_columns:
        computed[col] = {
            "sum": round(float(df[col].sum()), 2),
            "mean": round(float(df[col].mean()), 2),
            "max": round(float(df[col].max()), 2),
            "min": round(float(df[col].min()), 2)
        }
    
    obj_columns = df.select_dtypes(include=["object"]).columns

    for col in obj_columns:
        computed[col] = {
            "unique_count": int(df[col].nunique()),
            "most_frequent": str(df[col].mode().iloc[0])
        }
    
    prompt = f"""
        You are a senior business analyst.

        Below are pre-computed statistics across all rows of the dataset:

        {computed}

        Your Task:
        - Select the most meaningful KPIs a business shareholder will care about
        - Give each a clear, human-readable label
        - Use ONLY the values provided. Do NOT recalculate or invent values

        Return ONLY a flat JSON object. No markdown. NO explanation

        Example of format:
        {{
            "Total Revenue": 150000,
            "Avg Quantity sold": 57,
            "Top Region": "East",
            "Unique products": 12
        }}
    """

    fallback_kpis = {f"Total {col.replace('_', ' ').title()}":computed[col]["sum"] for col in numeric_columns}

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()
        parsed_content = json.loads(content)

        if not isinstance(parsed_content, dict):
            return fallback_kpis
        else:
            return parsed_content

    except Exception:
        return fallback_kpis


def generate_questions(df):

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.3)

    prompt = f"""
    You are a senior business analyst. Dataset Information:

    Table:
    {df.head(5).to_dict()}

    Generate 5 useful business intelligence questions
    a user might want to ask about this dataset.

    Return ONLY a JSON array.

    Example:

    [
        "Which product generates the most revenue?",
        "What is the monthly revenue trend?",
        "Which region performs best?"
    ]
    """

    response = llm.invoke([prompt])

    try:
        questions = json.loads(response.content)

    except Exception:

        questions = [
            "Which category performs best?",
            "Show revenue trends.",
            "What are the top insights?"
        ]

    return questions


def generate_insights(question,result_df):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.2
    )

    sample_data = result_df.head(20).to_markdown()

    prompt = f"""
    You are a senior business intelligence analyst.

    User Question:
    {question}

    Query Result:
    {sample_data}

    Analyze the result and provide:
    1. Key findings
    2. Trends
    3. Notable observations
    4. Business implications

    Keep the response concise.

    Use bullet points.
    """

    response = llm.invoke([prompt])

    return response.content

