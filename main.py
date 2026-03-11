import streamlit as st
import pandas as pd
import os
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(page_title="SmartWin Predictor", layout="wide")
st.title("🏆 SmartWin Predictor")
st.markdown("This tool evaluates project win probability using historical data.")

# 2. Data Loading & Cleaning
@st.cache_data
def load_data():
    files = [
        '2312SalesReport(DistyDCincluded_TTLresidualcumulative).xlsx',
        '2412SalesReport(DistyDCincluded_TTLresidualcumulative).xlsx',
        '2512SalesReport(DistyDCincluded_TTLresidualcumulative).xlsx'
    ]
    df_list = []
    for f in files:
        if os.path.exists(f):
            df_temp = pd.read_excel(f)
            df_list.append(df_temp)
    
    if not df_list:
        st.error("Data files not found!")
        st.stop()
        
    full_df = pd.concat(df_list, ignore_index=True).fillna('Unknown')
    
    # Ép kiểu số: xử lý lỗi str vs int
    numeric_cols = ['DI_AMT', 'Chance_AMT']
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
    
    # Tính Is_Success (dựa trên cột Stage có sẵn)
    full_df['Is_Success'] = (full_df['Stage'] == 'Design Win').astype(int)
    return full_df

df = load_data()

# 3. Model Training
features = ['Customer Name', 'Application', 'Product Line', 'Region', 'Chance_AMT', 'DI_AMT']
X = df[features].copy()
y = df['Is_Success']

# Mã hóa dữ liệu chữ
encoders = {col: LabelEncoder().fit(list(X[col].astype(str).unique()) + ['Unknown']) 
            for col in ['Customer Name', 'Application', 'Product Line', 'Region']}

for col in encoders:
    X[col] = encoders[col].transform(X[col].astype(str))

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# 4. Visualization Dashboard
st.subheader("📊 Historical Data Insights")
col1, col2 = st.columns(2)
with col1:
    st.write("Win Rate by Region (%)")
    st.bar_chart(df.groupby('Region')['Is_Success'].mean() * 100)


with col2:
    st.write("Success Count by Product Line")
    st.bar_chart(df[df['Is_Success'] == 1].groupby('Product Line').size())


# 5. User Input
st.sidebar.header("📝 New Project Details")
input_customer = st.sidebar.text_input("Customer Name")
input_app = st.sidebar.selectbox("Application", options=sorted(df['Application'].unique()))
input_prod = st.sidebar.selectbox("Product Line", options=sorted(df['Product Line'].unique()))
input_reg = st.sidebar.selectbox("Region", options=sorted(df['Region'].unique()))
input_chance_amt = st.sidebar.number_input("Chance_AMT", value=0.0)
input_di_amt = st.sidebar.number_input("DI_AMT", value=0.0)

# 6. Prediction Logic
if st.sidebar.button("Analyze Probability"):
    input_data = pd.DataFrame([[input_customer if input_customer else "Unknown", input_app, input_prod, input_reg, input_chance_amt, input_di_amt]], columns=features)
    
    for col in encoders:
        input_data[col] = encoders[col].transform([val if val in encoders[col].classes_ else 'Unknown' for val in input_data[col].astype(str)])

    prob = model.predict_proba(input_data)[0][1]
    
    st.header(f"📈 Predicted Success Rate: {prob*100:.1f}%")
    st.progress(prob)
    
    st.write(f"- Historical Baseline: **{(df['Is_Success'].mean()*100):.1f}%**")
    
    if prob < 0.3:
        st.error("Strategy Suggestion: Low potential. Consider increasing scope (Chance_AMT) or selecting a different region.")
    else:
        st.success("Strategy Suggestion: Promising project! Focus on accelerating the sales cycle.")
        
    # Export Report
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        input_data.to_excel(writer, index=False, sheet_name='Report')
    st.download_button("📥 Download Report as Excel", data=output.getvalue(), file_name="Prediction_Report.xlsx", mime="application/vnd.ms-excel")