import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data and models
data = pd.read_csv("data.csv")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance & Recommendation System")

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Select Dashboard",
    ["Study Dashboard", "Admin Dashboard", "Analytics & Performance"]
)

# ---------------- STUDY DASHBOARD ----------------
if menu == "Study Dashboard":
    st.header("📘 Study Dashboard")

    student_id = st.selectbox("Select Student ID", data.index)

    # Select numeric data only
    numeric_data = data.select_dtypes(include=["int64", "float64"])

    # Remove unwanted columns if present
    for col in ["Cluster", "cluster", "target", "label"]:
        if col in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=[col])

    # Select student row
    student_data = numeric_data.iloc[[student_id]]

    # IMPORTANT: convert to NumPy array
    scaled_data = scaler.transform(student_data.values)

    # Predict cluster
    cluster = kmeans.predict(scaled_data)[0]

    st.subheader("Student Details")
    st.write(student_data)

    st.subheader("Cluster Assigned")
    st.success(f"Cluster {cluster}")

    st.subheader("Personalized Feedback")
    if cluster == 0:
        st.warning("Needs Improvement. Focus on basics and extra practice.")
    elif cluster == 1:
        st.info("Average Performance. Maintain consistency.")
    else:
        st.success("Excellent Performance. Try advanced learning material.")


# ---------------- ADMIN DASHBOARD ----------------
elif menu == "Admin Dashboard":
    st.header("🧑‍💼 Admin Dashboard")

    data["Cluster"] = kmeans.labels_

    st.metric("Total Students", len(data))

    cluster_counts = data["Cluster"].value_counts()
    st.subheader("Cluster Distribution")
    st.bar_chart(cluster_counts)

# ---------------- ANALYTICS & PERFORMANCE ----------------
elif menu == "Analytics & Performance":
    st.header("📊 Analytics & Performance")

    data["Cluster"] = kmeans.labels_

    st.subheader("Cluster-wise Average Performance")
    st.write(data.groupby("Cluster").mean())

    st.subheader("Cluster Distribution Chart")
    fig, ax = plt.subplots()
    data["Cluster"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)





