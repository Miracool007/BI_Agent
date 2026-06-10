import streamlit as st
from reg_tools import *
from ai_tools import *

st.set_page_config(
    page_title="AI Business Intelligence Agent",
    layout="wide"
)

st.title("📊 AI Business Intelligence Agent")
st.markdown("#### **About This Project**")
st.write("**AI Business Intelligence Agent** is an AI-powered analytics assistant that enables users to upload CSV or Excel datasets and explore their data using natural language. Simply ask questions in plain English, and the system generates queries, visualizations, insights, and recommendations to help uncover meaningful business intelligence." \
" Built with Python, Streamlit, DuckDB, Plotly, and LLMs powered by Groq's Llama 3.3 70B model, this project showcases modern AI engineering techniques including natural language analytics, conversational memory, automated insight generation, and intelligent data exploration." \
" Designed and developed by **Miracle Aniobi** as a demonstration of practical AI solutions for business decision-making and data-driven insights.")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx", "xls"]
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""

if uploaded_file:
    try:
        df = load_data(uploaded_file)
    except UnicodeDecodeError:
        st.warning("⚠️ Could not read this file. It may have an unsupported encoding. " \
        "Try re-saving it as UTF-8 CSV and uploading again.")
        st.stop()
    except Exception:
        st.warning("⚠️ Something went wrong while loading your file. " \
        "Please check that it is a valid CSV or Excel file and try again.")
        st.stop()

    st.success("Dataset Loaded Successfully")

    profile = profile_dataset(df)

    st.subheader("Data Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", profile["rows"])
    
    with col2:
        st.metric("Columns", profile["columns"])
    
    kpis = generate_kpis(df)
    st.subheader("Key Metrics")
    cols = st.columns(min(len(kpis), 4))

    for i, (name, value) in enumerate(kpis.items()):
        if isinstance(value, (int, float)):
            display_value = f"{value:,.2f}"
        else:
            display_value = str(value)

        cols[i % 4].metric(label=name, value=display_value)
    

    quality = data_quality_report(df)
    st.subheader("Data Quality")
    st.write(f"Duplicate Rows: {quality['duplicates']}")
    st.write("Duplicated Columns:👇")
    st.dataframe(quality["missing_values"])

    st.subheader("Data Preview")
    st.dataframe(df.head())

    gen_questions = generate_questions(df)
    st.subheader("Suggested Questions")
    for q in gen_questions:
        st.info(q)

    st.subheader("Question & Answer Area")

    question = st.chat_input("Ask a question about your data...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        try:

            sql_query = generate_sql(question, df.columns.tolist(), st.session_state.chat_history)

            result, final_sql = execute_sql_with_retry(df, sql_query, df.columns.tolist())

            with st.expander("Generated SQL"):
                if final_sql != sql_query:
                    st.warning("Original SQL had an error - Auto-corrected:")
                    st.code(sql_query, language="sql")
                    st.success("Fixed SQL (what ran):")
                st.code(final_sql, language="sql")

            st.session_state.last_result = result
            st.session_state.last_question = question
            st.session_state.last_sql = sql_query

            with st.chat_message("assistant"):

                st.write("### Result")

                st.dataframe(result)

            chart_type = recommend_chart(result, question)

            fig = generate_chart(result, chart_type)

            if fig:
                with st.expander("AI Chart Decision"):
                    st.code(chart_type, "json")
                
                st.plotly_chart(fig, use_container_width=True)


            insights = generate_insights(question,result)

            st.subheader("AI Insights")

            st.markdown(insights)

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
                })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Analyzed: {question}",
                "sql": final_sql
                })
            
            st.subheader("Conversation History")

            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"🧑‍💻 **You:** {msg['content']}")
                else:
                    st.markdown(f"🤖 **AI:** {msg['content']}")

        except Exception as e:
            st.warning(f"Error in uploaded file: {e}")
    
