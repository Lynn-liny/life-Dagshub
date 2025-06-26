import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.header("📈 Interactive Data Visualization")

# Check for data
if st.session_state.df is None:
    st.warning("⚠️ Please load data from the 📊 Data Loading page first")
    st.stop()

df = st.session_state.df
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Visualization Controls
st.subheader("🎛️ Visualization Controls")
col1, col2, col3 = st.columns(3)
with col1:
    viz_type = st.selectbox("Select visualization type:", [
        "📊 Histogram", "📈 Scatter Plot", "📦 Box Plot"
    ], key="viz_type_select")

with col2:
    if viz_type == "📈 Scatter Plot":
        x_col = st.selectbox("X-axis:", numeric_cols + categorical_cols, index=0, key="x_axis_select")
        y_col = st.selectbox("Y-axis:", numeric_cols, index=0 if not numeric_cols else (numeric_cols.index("Life expectancy ") if "Life expectancy " in numeric_cols else 0), key="y_axis_select")
    else:
        selected_col = st.selectbox("Select column:", numeric_cols if viz_type != "📦 Box Plot" else df.columns, index=0 if not numeric_cols else (numeric_cols.index("Life expectancy ") if "Life expectancy " in numeric_cols else 0), key="selected_col_select")

with col3:
    color_options = ["None"] + categorical_cols if categorical_cols else ["None"]
    color_col = st.selectbox("Color by (optional):", color_options, index=0 if not categorical_cols else (categorical_cols.index("Status") + 1 if "Status" in categorical_cols else 0), key="color_col_select")
    color_col = None if color_col == "None" else color_col

# Visualization Output
st.subheader("📊 Visualization Output")
try:
    if viz_type == "📊 Histogram":
        fig = px.histogram(df, x=selected_col, color=color_col, title=f'Distribution of {selected_col}', marginal="box", nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "📈 Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f'{y_col} vs {x_col}', trendline="ols" if color_col is None else None)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "📦 Box Plot":
        if color_col:
            fig = px.box(df, y=selected_col, x=color_col, title=f'Box Plot of {selected_col} by {color_col}')
        else:
            fig = px.box(df, y=selected_col, title=f'Box Plot of {selected_col}')
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"❌ Error generating visualization: {str(e)}. Check column selections or data integrity.")
    st.write(f"Debug: viz_type={viz_type}, selected_col={selected_col if 'selected_col' in locals() else None}, x_col={x_col if 'x_col' in locals() else None}, y_col={y_col if 'y_col' in locals() else None}, color_col={color_col}")

# Additional Insights
st.subheader("📊 Additional Insights")
if categorical_cols and "Status" in categorical_cols:
    try:
        status_counts = df["Status"].value_counts()
        fig = px.bar(status_counts, x=status_counts.index, y=status_counts.values, title="Distribution of Status", text=status_counts.values, height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not generate Status distribution: {str(e)}. Ensure 'Status' is a valid categorical column.")
elif not categorical_cols:
    st.info("🔍 No categorical columns available for additional insights.")
else:
    st.info("🔍 'Status' column not found for additional insights.")