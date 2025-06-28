import pandas as pd
import pytesseract
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import spacy
from sklearn.ensemble import IsolationForest
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re

# Load SpaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()

# Load Google Sheets Data
@st.cache_data(ttl=3600)
def load_data():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account_key.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Medical Rep Expense Submission (Responses)").sheet1
        records = sheet.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {str(e)}")
        st.stop()

# Extract amount from OCR

def extract_amount(text):
    amounts = re.findall(r"(?:total|amount)[^\d]*(\d+[.,]?\d*)", text, flags=re.IGNORECASE)
    numbers = re.findall(r"\d+[.,]?\d*", text)
    candidates = amounts or numbers
    try:
        return float(max([float(a.replace(',', '')) for a in candidates]))
    except:
        return None

# OCR text extraction from receipt image

def process_receipt(url):
    try:
        if pd.isna(url) or url.strip() == "":
            return "No receipt URL provided", None
        if "drive.google.com" in url:
            if "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                file_id = url.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = requests.get(url, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('L')
        text = pytesseract.image_to_string(img, config='--psm 6')
        amount = extract_amount(text)
        return text.strip() or "No text could be extracted from receipt", amount
    except UnidentifiedImageError:
        return "Error: Unsupported image format", None
    except requests.RequestException as e:
        return f"Error downloading receipt: {str(e)}", None
    except Exception as e:
        return f"Error processing receipt: {str(e)}", None

# Parallel receipt processing

def process_receipts_parallel(urls):
    with st.spinner("Extracting text from receipts..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_receipt, urls))
    texts, amounts = zip(*results)
    return texts, amounts

# Apply expense policy rules

def check_policy_compliance(row):
    violations = []
    try:
        category = str(row["Expense Category"]).strip().lower()
        amount = float(row["Amount (EGP)"])
        extracted_amount = row.get("Extracted Amount")

        if category == "meals" and amount > 100:
            violations.append("Meal exceeds limit (100 EGP)")
        elif category == "hotel" and amount > 1000:
            violations.append("Hotel exceeds limit (1000 EGP)")
        elif category == "fuel" and amount > 500:
            violations.append("Fuel exceeds limit (500 EGP)")
        elif category == "transportation" and amount > 100:
            violations.append("Transportation exceeds limit (100 EGP)")

        if category == "hotel" and not str(row.get("Description", "")).strip():
            violations.append("Missing description for hotel expense")

        vendor = str(row.get("Vendor Name", "")).lower()
        if vendor and vendor not in str(row.get("Receipt Text", "")).lower():
            violations.append(f"Vendor '{vendor}' not found in receipt")

        if extracted_amount is None or abs(extracted_amount - amount) > 5:
            violations.append("Declared amount does not match receipt")

    except Exception as e:
        violations.append(f"Error during policy check: {e}")

    return violations if violations else ["Compliant"]

# Anomaly detection

def detect_anomalies(df):
    try:
        amounts = df["Amount (EGP)"].astype(float).values.reshape(-1, 1)
        model = IsolationForest(contamination=0.1, random_state=42)
        return model.fit_predict(amounts)
    except Exception as e:
        st.error(f"Anomaly detection failed: {str(e)}")
        return np.zeros(len(df))

# UI detail panel

def display_expense_details(expense):
    with st.expander("Expense Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Basic Information")
            st.write(f"**Employee:** {expense.get('Full Name', 'N/A')}")
            st.write(f"**Date:** {expense.get('Date', 'N/A')}")
            st.write(f"**Category:** {expense.get('Expense Category', 'N/A')}")
            st.write(f"**Amount:** {expense.get('Amount (EGP)', 'N/A')} EGP")
            st.write(f"**Vendor:** {expense.get('Vendor Name', 'N/A')}")
            st.write(f"**Extracted Total:** {expense.get('Extracted Amount', 'N/A')} EGP")

            st.markdown("### Policy Check")
            violations = expense.get("Policy Violations", [])
            if violations and violations != ["Compliant"]:
                for violation in violations:
                    st.error(violation)
            else:
                st.success("Compliant with all policies")

            st.markdown("### Anomaly Detection")
            st.write("🚩 Anomaly detected" if expense.get("Is Anomaly") else "✅ Normal expense")

        with col2:
            st.markdown("### Receipt Text")
            st.text_area("Extracted text", expense.get("Receipt Text", "No receipt text available"), height=200, label_visibility="collapsed")
            st.markdown("### Description")
            st.write(expense.get("Description", "No description provided"))

# Streamlit app entry

def main():
    st.set_page_config(page_title="AI Expense Auditor", layout="wide")
    st.title("📊 AI Expense Auditor")

    try:
        df = load_data()
        if df.empty:
            st.warning("No expense data found in the Google Sheet")
            return
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        return

    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Process All Receipts"):
            texts, extracted_amounts = process_receipts_parallel(df["Upload Receipt"])
            df["Receipt Text"] = texts
            df["Extracted Amount"] = extracted_amounts
            st.success("Receipt processing complete!")

        st.markdown("---")
        st.subheader("Filters")
        category_filter = st.selectbox("Expense Category", ["All"] + sorted(df["Expense Category"].unique().tolist()))
        anomaly_filter = st.checkbox("Show only anomalies", True)
        non_compliant_filter = st.checkbox("Show only policy violations", True)

    if "Receipt Text" not in df.columns or "Extracted Amount" not in df.columns:
        st.warning("Click 'Process All Receipts' to analyze receipts")
        st.dataframe(df.head(5))
        return

    df["Policy Violations"] = df.apply(check_policy_compliance, axis=1)
    df["Amount Matches Receipt"] = df["Policy Violations"].apply(lambda x: not any("amount" in v.lower() for v in x))
    df["Is Non-Compliant"] = df["Policy Violations"].apply(lambda x: x != ["Compliant"])
    df["Anomaly Score"] = detect_anomalies(df)
    df["Is Anomaly"] = df["Anomaly Score"] == -1

    filtered = df.copy()
    if category_filter != "All":
        filtered = filtered[filtered["Expense Category"] == category_filter]
    if anomaly_filter:
        filtered = filtered[filtered["Is Anomaly"]]
    if non_compliant_filter:
        filtered = filtered[filtered["Is Non-Compliant"]]

    st.subheader("Expense Report Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Expenses", len(df))
    col2.metric("Policy Violations", df["Is Non-Compliant"].sum())
    col3.metric("Anomalies Detected", df["Is Anomaly"].sum())

    if not filtered.empty:
        st.dataframe(filtered[["Full Name", "Expense Category", "Amount (EGP)", "Vendor Name", "Extracted Amount", "Policy Violations", "Is Anomaly"]], height=300, use_container_width=True)
        st.subheader("Detailed Analysis")
        filtered = filtered.reset_index(drop=True)
        selected = st.selectbox(
            "Choose expense",
            options=list(filtered.iterrows()),
            format_func=lambda x: f"{x[1].get('Full Name', 'Unknown')} - {x[1].get('Expense Category', 'Unknown')} - {x[1].get('Amount (EGP)', 'N/A')} EGP"
        )
        display_expense_details(selected[1])
    else:
        st.warning("No expenses match the current filters.")

if __name__ == "__main__":
    main()
