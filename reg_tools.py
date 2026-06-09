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

def load_data(uploaded_file):

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file format")


    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df


def generate_chart(df, chart_spec):

    chart_type = chart_spec["chart_type"]

    x = chart_spec["x"]
    y = chart_spec["y"]

    if chart_type == "bar":
        return px.bar(df, x, y, title=f"{y} per {x}")
    
    elif chart_type == "line":
        return px.line(df, x, y, title=f"{y} per {x}")
    
    elif chart_type == "pie":
        return px.pie(df, x, y)

    elif chart_type == "scatter":
        return px.scatter(df, x, y)
    
    elif chart_type == "histogram":
        px.histogram(df, x)
    
    return None


def profile_dataset(df):

    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "numeric_columns": [],
        "categorical_columns": [],
        "date_columns": []
    }

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            profile["numeric_columns"].append(col)

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            profile["date_columns"].append(col)

        else:
            profile["categorical_columns"].append(col)

    return profile


def data_quality_report(df):

    reports = {}

    reports["missing_values"] = (
        df.isnull()
        .sum()
        .to_dict()
    )

    reports["duplicates"] = (
        df.duplicated()
        .sum()
    )

    return reports



