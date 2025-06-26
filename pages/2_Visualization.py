import streamlit as st
import pandas as pd
import plotly.express as px # Used for creating interactive plots
import numpy as np # Used for checking numeric data types

# --- Page Header ---
st.header("📈 Interactive Data Visualization")
st.markdown("---") # Add a separator for better visual structure

# --- Data Availability Check ---
# This is crucial for pages that depend on data loaded by the main app.
# If 'df' (DataFrame) is not found in Streamlit's session state,
# it means the data has not been loaded yet. A warning is displayed,
# and the script execution for this page is stopped until data is available.
if st.session_state.df is None:
    st.warning("⚠️ Please load data from the **📊 Data Loading** page first to enable visualizations.")
    st.stop() # Halts execution if no data is present

# Retrieve the DataFrame from session state
df = st.session_state.df

# --- Column Categorization ---
# Automatically identifies numeric and categorical columns in the loaded DataFrame.
# These lists are used to populate the select boxes for plot axis and color options.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# --- Visualization Controls ---
# This section provides interactive widgets for users to select plot types
# and specify which columns to use for X-axis, Y-axis, and color encoding.
st.subheader("🎛️ Visualization Controls")
col1, col2, col3 = st.columns(3) # Organizes widgets into three columns for better layout

with col1:
    # Dropdown to select the type of visualization
    viz_type = st.selectbox("Select visualization type:", [
        "📊 Histogram", "📈 Scatter Plot", "📦 Box Plot"
    ], key="viz_type_select")

with col2:
    # Dynamic selection of X/Y axis or a single column based on the chosen visualization type.
    # It attempts to pre-select 'Life expectancy ' if available and numeric.
    if viz_type == "📈 Scatter Plot":
        # For Scatter Plot, both X and Y axes are needed. X can be numeric or categorical.
        x_col = st.selectbox("X-axis:", numeric_cols + categorical_cols, 
                             index=0 if (not numeric_cols and not categorical_cols) else 
                                 (numeric_cols.index("Life expectancy ") if "Life expectancy " in numeric_cols else 
                                  (categorical_cols.index("Country") + len(numeric_cols) if "Country" in categorical_cols else 0) 
                                 ) if numeric_cols or categorical_cols else 0, # Handles cases where lists might be empty
                             key="x_axis_select")
        
        # Y-axis is typically numeric for regression plots like scatter plots.
        y_col = st.selectbox("Y-axis:", numeric_cols, 
                             index=0 if not numeric_cols else (numeric_cols.index("Life expectancy ") if "Life expectancy " in numeric_cols else 0), 
                             key="y_axis_select")
    else:
        # For Histogram and Box Plot, a single primary column is selected.
        # Box Plot can take any column, Histogram is usually numeric.
        cols_for_selection = numeric_cols if viz_type == "📊 Histogram" else df.columns.tolist()
        selected_col = st.selectbox("Select column:", cols_for_selection, 
                                    index=0 if not cols_for_selection else 
                                        (cols_for_selection.index("Life expectancy ") if "Life expectancy " in cols_for_selection else 0), 
                                    key="selected_col_select")

with col3:
    # Option to color the plot by a categorical column.
    color_options = ["None"] + categorical_cols # 'None' allows for no color differentiation
    color_col = st.selectbox("Color by (optional):", color_options, 
                             index=0 if not categorical_cols else (categorical_cols.index("Status") + 1 if "Status" in categorical_cols else 0), 
                             key="color_col_select")
    color_col = None if color_col == "None" else color_col # Set to None if 'None' is selected

# --- Visualization Output ---
# This section generates and displays the Plotly chart based on user selections.
# A try-except block is used to catch potential errors during plot generation.
st.subheader("📊 Visualization Output")
try:
    if viz_type == "📊 Histogram":
        # Histogram shows the distribution of a single variable.
        # 'marginal="box"' adds a box plot on top to show quartiles and outliers.
        fig = px.histogram(df, x=selected_col, color=color_col, 
                           title=f'Distribution of {selected_col}', 
                           marginal="box", nbins=30)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "📈 Scatter Plot":
        # Scatter Plot shows the relationship between two variables.
        # 'trendline="ols"' adds an Ordinary Least Squares regression line if no color is applied.
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                         title=f'{y_col} vs {x_col}', 
                         trendline="ols" if color_col is None else None)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "📦 Box Plot":
        # Box Plot visualizes the distribution of numerical data and can show
        # differences across categories if a 'color_col' (x-axis) is provided.
        if color_col:
            fig = px.box(df, y=selected_col, x=color_col, 
                         title=f'Box Plot of {selected_col} by {color_col}')
        else:
            fig = px.box(df, y=selected_col, 
                         title=f'Box Plot of {selected_col}')
        st.plotly_chart(fig, use_container_width=True)
        
except Exception as e:
    # Error message if plot generation fails (e.g., incompatible column types)
    st.error(f"❌ Error generating visualization: {str(e)}. "
             "Please check your column selections or data integrity.")
    # Debug information for troubleshooting
    st.write(f"Debug: viz_type={viz_type}, "
             f"selected_col={selected_col if 'selected_col' in locals() else 'N/A'}, "
             f"x_col={x_col if 'x_col' in locals() else 'N/A'}, "
             f"y_col={y_col if 'y_col' in locals() else 'N/A'}, "
             f"color_col={color_col}")

# --- Additional Insights ---
# This section provides an example of a specific, pre-defined visualization.
# It checks for the 'Status' column to display its distribution.
st.subheader("📊 Additional Insights")
if "Status" in categorical_cols: # Check if 'Status' column exists and is categorical
    try:
        status_counts = df["Status"].value_counts()
        fig = px.bar(status_counts, x=status_counts.index, y=status_counts.values, 
                     title="Distribution of Status", text_auto=True, height=400) # text_auto for bar values
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not generate 'Status' distribution: {str(e)}. "
                   "Ensure 'Status' is a valid categorical column with suitable data.")
elif not categorical_cols:
    st.info("🔍 No categorical columns available for additional insights (e.g., 'Status' distribution).")
else:
    st.info("🔍 The 'Status' column was not found for specific additional insights.")

# --- Debugging Confirmation ---
st.write("Debug: 2_Visualization.py page loaded and executed.")
