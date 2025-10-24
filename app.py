import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_text, load_reply_templates

# Load model + replies
model = joblib.load("comment_classifier.pkl")
reply_templates = load_reply_templates()

st.set_page_config(page_title="Comment Categorization Tool", layout="wide")
st.title("💬 Comment Categorization & Reply Assistant")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This tool classifies user comments into categories like Praise, Support, "
    "Constructive Criticism, Hate, Spam, etc. and suggests empathetic replies."
)

# Text input
user_input = st.text_area("✍️ Enter a comment:", height=100)

if st.button("Classify Comment"):
    if user_input.strip():
        cleaned = preprocess_text(user_input)
        category = model.predict([cleaned])[0]
        reply = reply_templates.get(category, "Thanks for your feedback!")

        st.subheader("🔎 Prediction Result")
        st.write(f"**Category:** {category}")
        st.write(f"**Suggested Reply:** {reply}")

# Batch upload
st.subheader("📂 Batch Comment Analysis")
uploaded_file = st.file_uploader("Upload CSV with 'text' column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must have a 'text' column.")
    else:
        df["cleaned_text"] = df["text"].apply(preprocess_text)
        df["Predicted_Category"] = model.predict(df["cleaned_text"])
        df["Suggested_Reply"] = df["Predicted_Category"].map(
            lambda c: reply_templates.get(c, "Thanks for your feedback!")
        )

        st.success("✅ Processed file")
        st.dataframe(df[["text", "Predicted_Category", "Suggested_Reply"]])

        # Visualization
        st.subheader("📊 Category Distribution")
        plt.figure(figsize=(8,5))
        sns.countplot(x=df["Predicted_Category"], order=df["Predicted_Category"].value_counts().index, palette="viridis")
        plt.xticks(rotation=30)
        st.pyplot(plt)

        # Download processed file
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Categorized Comments", csv, "categorized_comments.csv", "text/csv")
