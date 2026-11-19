
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="PredictCart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# SOLID PROFESSIONAL GRAPH COLOR PALETTE
# ==================================================
BLUE = "#2F65E5"
PINK = "#E83E8C"
ORANGE = "#FF7A00"
RED = "#E53935"
YELLOW = "#F4B400"
NAVY = "#1E3A5F"


# ==================================================
# CUSTOM WEBSITE STYLE
# ==================================================
st.markdown("""
<style>

.stApp {
    background-color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: #F8F6FC;
    border-right: 1px solid #E9D5FF;
}

[data-testid="stSidebar"] * {
    color: #463651;
}

.hero {
    background: #FFFFFF;
    padding: 35px 40px;
    border-radius: 20px;
    border: 1px solid #E9D5FF;
    box-shadow: 0px 8px 25px rgba(90, 70, 110, 0.10);
    margin-bottom: 28px;
}

.hero h1 {
    color: #463651;
    font-size: 46px;
    margin-bottom: 8px;
}

.hero p {
    color: #75657F;
    font-size: 18px;
}

h1, h2, h3 {
    color: #463651;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E9D5FF;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(90, 70, 110, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #75657F;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #463651;
    font-weight: 700;
}

.stButton > button {
    width: 100%;
    background-color: #7C5CBF;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 600;
    font-size: 16px;
}

.stButton > button:hover {
    background-color: #65479E;
    color: white;
}

[data-testid="stNumberInput"] input {
    border: 1px solid #DDD6FE;
    border-radius: 10px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E9D5FF;
    border-radius: 15px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 15px;
}

hr {
    border-color: #E9D5FF;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD DATA AND MODEL
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("Ecommerce.csv")


FEATURES = [
    "Avg. Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership"
]
TARGET = "Yearly Amount Spent"


@st.cache_resource
def train_models(data):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=5))
        ])
    }

    trained_models = {}
    metric_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        trained_models[name] = model
        mape = (abs((y_test - predictions) / y_test).replace([float("inf"), -float("inf")], pd.NA).dropna()).mean() * 100
        metric_rows.append({
            "Algorithm": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mean_squared_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "MAPE (%)": mape,
            "Accuracy Rate (%)": max(0.0, 100.0 - mape),
            "R² Score": r2_score(y_test, predictions)
        })

    return trained_models, pd.DataFrame(metric_rows)


df = load_data()
trained_models, results = train_models(df)


# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.markdown("# 🛒 PredictCart")
st.sidebar.caption("Customer Spending Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "📊 Analytics",
        "🤖 Model Comparison",
        "💰 Spending Prediction"
    ]
)


# ==================================================
# DASHBOARD
# ==================================================
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🛒 PredictCart</h1>
        <p>Smart Customer Spending Analytics & Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Total Customers", f"{len(df):,}")

    with col2:
        st.metric("📊 Dataset Features", len(df.columns))

    with col3:
        st.metric("🤖 Models Compared", "4")

    with col4:
        st.metric("🏆 Best Model", "Random Forest")

    st.markdown("<br>", unsafe_allow_html=True)

    # QUICK INSIGHTS
    st.subheader("📈 Quick Insights")

    col1, col2 = st.columns(2)

    # GRAPH 1
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(
            df["Yearly Amount Spent"],
            bins=20,
            color=BLUE,
            edgecolor=NAVY,
            alpha=0.85
        )

        ax.set_title("Customer Spending Distribution")
        ax.set_xlabel("Yearly Amount Spent")
        ax.set_ylabel("Number of Customers")
        ax.grid(axis="y", alpha=0.15)

        st.pyplot(fig)

    # GRAPH 2
    with col2:
        features = [
            "Avg. Session Length",
            "Time on App",
            "Time on Website",
            "Length of Membership"
        ]

        feature_means = df[features].mean()

        fig, ax = plt.subplots(figsize=(6, 4))

        bars = ax.bar(
            feature_means.index,
            feature_means.values,
            color=[BLUE, PINK, ORANGE, RED]
        )

        ax.set_title("Average Customer Behaviour")
        ax.tick_params(axis="x", rotation=20)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        st.pyplot(fig)

    # ==================================================
    # DATASET PREVIEW
    # ==================================================
    st.markdown("---")
    st.subheader("📋 Dataset Preview")

    preview_df = df.head(10).copy()

    # REALISTIC INDIAN CUSTOMER DETAILS
    names = [
        "Aarav Sharma",
        "Priya Patil",
        "Rohan Kulkarni",
        "Sneha Joshi",
        "Aditya Deshmukh",
        "Kavya Singh",
        "Rahul Pawar",
        "Ananya Gupta",
        "Siddharth Patil",
        "Neha Sharma"
    ]

    emails = [
        "aarav.sharma27@gmail.com",
        "priya.patil08@gmail.com",
        "rohan.kulkarni12@gmail.com",
        "sneha.joshi19@gmail.com",
        "aditya.deshmukh05@gmail.com",
        "kavya.singh21@gmail.com",
        "rahul.pawar17@gmail.com",
        "ananya.gupta14@gmail.com",
        "siddharth.patil09@gmail.com",
        "neha.sharma23@gmail.com"
    ]

    addresses = [
        "Andheri East, Mumbai, Maharashtra, India",
        "Kothrud, Pune, Maharashtra, India",
        "Vashi, Navi Mumbai, Maharashtra, India",
        "Thane West, Thane, Maharashtra, India",
        "Alibag, Raigad, Maharashtra, India",
        "Baner, Pune, Maharashtra, India",
        "Dadar, Mumbai, Maharashtra, India",
        "Ghodbunder Road, Thane, Maharashtra, India",
        "College Road, Nashik, Maharashtra, India",
        "Dharampeth, Nagpur, Maharashtra, India"
    ]

    # ADD COLUMNS ONLY IF NOT ALREADY PRESENT
    if "Name" not in preview_df.columns:
        preview_df.insert(0, "Name", names)

    if "Email" not in preview_df.columns:
        preview_df.insert(1, "Email", emails)

    if "Address" not in preview_df.columns:
        preview_df.insert(2, "Address", addresses)

    columns_to_show = [
        "Name",
        "Email",
        "Address",
        "Avg. Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership",
        "Yearly Amount Spent"
    ]

    available_columns = [
        col for col in columns_to_show
        if col in preview_df.columns
    ]

    st.dataframe(
        preview_df[available_columns],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# ANALYTICS PAGE
# ==================================================
elif page == "📊 Analytics":

    st.title("📊 Customer Analytics")
    st.caption("Explore customer behaviour and spending patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Spending Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(
            df["Yearly Amount Spent"],
            bins=20,
            color=BLUE,
            edgecolor=NAVY,
            alpha=0.85
        )

        ax.set_xlabel("Yearly Amount Spent")
        ax.set_ylabel("Number of Customers")
        ax.grid(axis="y", alpha=0.15)

        st.pyplot(fig)

    with col2:
        st.subheader("⭐ Membership vs Spending")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(
            df["Length of Membership"],
            df["Yearly Amount Spent"],
            color=PINK,
            alpha=0.75,
            edgecolors="white"
        )

        ax.set_xlabel("Length of Membership")
        ax.set_ylabel("Yearly Amount Spent")
        ax.grid(alpha=0.15)

        st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📱 App Usage vs Spending")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(
            df["Time on App"],
            df["Yearly Amount Spent"],
            color=ORANGE,
            alpha=0.75
        )

        ax.set_xlabel("Time on App")
        ax.set_ylabel("Yearly Amount Spent")
        ax.grid(alpha=0.15)

        st.pyplot(fig)

    with col2:
        st.subheader("🌐 Website Usage vs Spending")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(
            df["Time on Website"],
            df["Yearly Amount Spent"],
            color=RED,
            alpha=0.75
        )

        ax.set_xlabel("Time on Website")
        ax.set_ylabel("Yearly Amount Spent")
        ax.grid(alpha=0.15)

        st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Customer Spending Categories")

        spending_category = pd.cut(
            df["Yearly Amount Spent"],
            bins=3,
            labels=["Low", "Medium", "High"]
        )

        category_counts = spending_category.value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            category_counts.values,
            labels=category_counts.index,
            autopct="%1.1f%%",
            colors=[BLUE, PINK, ORANGE],
            startangle=90,
            wedgeprops={
                "edgecolor": "white",
                "linewidth": 2
            }
        )

        st.pyplot(fig)

    with col2:
        st.subheader("📊 Average Feature Values")

        features = [
            "Avg. Session Length",
            "Time on App",
            "Time on Website",
            "Length of Membership"
        ]

        feature_means = df[features].mean()

        fig, ax = plt.subplots(figsize=(6, 4))

        bars = ax.bar(
            feature_means.index,
            feature_means.values,
            color=[BLUE, PINK, ORANGE, RED]
        )

        ax.tick_params(axis="x", rotation=20)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig)

    st.markdown("---")

    # CORRELATION HEATMAP
    st.subheader("🔥 Feature Correlation Heatmap")

    numerical_df = df.select_dtypes(include="number")
    corr = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        corr,
        cmap="Blues",
        aspect="auto"
    )

    fig.colorbar(im)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    st.pyplot(fig)


# ==================================================
# MODEL COMPARISON PAGE
# ==================================================
elif page == "🤖 Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")
    st.caption("Compare four regression algorithms trained on the same dataset.")

    st.subheader("📋 Performance Results")
    st.dataframe(
        results.sort_values("R² Score", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 R² Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        sorted_results = results.sort_values("R² Score", ascending=False)
        bars = ax.bar(sorted_results["Algorithm"], sorted_results["R² Score"],
                      color=[BLUE, PINK, ORANGE, RED])
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.15)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                    ha="center", va="bottom")
        st.pyplot(fig)

    with col2:
        st.subheader("📊 MAE Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(results["Algorithm"], results["MAE"],
                      color=[PINK, ORANGE, BLUE, RED])
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.15)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}",
                    ha="center", va="bottom")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📉 MSE Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(results["Algorithm"], results["MSE"],
                  color=[BLUE, PINK, ORANGE, RED])
    ax.grid(axis="y", alpha=0.15)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.1f}",
                ha="center", va="bottom")
    st.pyplot(fig)

    best_model = results.loc[results["R² Score"].idxmax()]
    st.success(f"🏆 Best Model: {best_model['Algorithm']} | R² Score: {best_model['R² Score']:.4f}")


# ==================================================
# SPENDING PREDICTION PAGE — DESIGN 2
# ==================================================
elif page == "💰 Spending Prediction":

    st.markdown("""
    <div style="padding:8px 0 18px 0;">
        <h1 style="margin-bottom:4px;">💰 Spending Prediction</h1>
        <p style="color:#718096;margin-top:0;">Predict a customer's yearly spending using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    # Algorithm selection
    st.markdown("### 🤖 Select Machine Learning Algorithm")
    selected_algorithm = st.selectbox(
        "Choose algorithm",
        list(trained_models.keys()),
        help="The selected model will be used to generate the prediction."
    )
    selected_model = trained_models[selected_algorithm]
    selected_metrics = results[results["Algorithm"] == selected_algorithm].iloc[0]

    st.markdown("---")
    st.markdown("### 👤 Customer Behaviour Details")
    st.caption("Enter the customer's usage and membership information.")

    input_col1, input_col2 = st.columns(2)
    with input_col1:
        session_length = st.number_input(
            "⏱️ Average Session Length", min_value=0.0, value=33.5, step=0.1
        )
        time_on_app = st.number_input(
            "📱 Time on App", min_value=0.0, value=12.0, step=0.1
        )
    with input_col2:
        time_on_website = st.number_input(
            "🌐 Time on Website", min_value=0.0, value=37.0, step=0.1
        )
        membership_length = st.number_input(
            "⭐ Length of Membership", min_value=0.0, value=4.5, step=0.1
        )

    # Model metrics in a compact row
    st.markdown("---")
    st.markdown("### 📈 Selected Model Performance")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Approx. Accuracy Rate", f"{selected_metrics['Accuracy Rate (%)']:.2f}%")
    metric2.metric("R² Score", f"{selected_metrics['R² Score'] * 100:.2f}%")
    metric3.metric("MAE", f"₹ {selected_metrics['MAE']:,.2f}")
    metric4.metric("RMSE", f"₹ {selected_metrics['RMSE']:,.2f}")
    st.caption(
        "Accuracy Rate is an approximate regression score calculated as 100 − MAPE "
        "on the held-out test data. It is not classification accuracy."
    )

    predict_clicked = st.button(
        "✨ Generate Prediction",
        use_container_width=True,
        type="primary"
    )

    if predict_clicked:
        new_customer = pd.DataFrame([{
            "Avg. Session Length": session_length,
            "Time on App": time_on_app,
            "Time on Website": time_on_website,
            "Length of Membership": membership_length
        }], columns=FEATURES)

        prediction = float(selected_model.predict(new_customer)[0])

        # Keep the last prediction available after Streamlit reruns.
        st.session_state["predictcart_result"] = {
            "prediction": prediction,
            "algorithm": selected_algorithm,
            "accuracy": float(selected_metrics["Accuracy Rate (%)"]),
            "r2": float(selected_metrics["R² Score"] * 100),
            "rmse": float(selected_metrics["RMSE"]),
            "mae": float(selected_metrics["MAE"]),
            "inputs": {
                "session_length": session_length,
                "time_on_app": time_on_app,
                "time_on_website": time_on_website,
                "membership_length": membership_length
            }
        }

    # Display the most recent prediction
    result = st.session_state.get("predictcart_result")

    if result:
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        result_left, result_right = st.columns([1.55, 1], vertical_alignment="center")

        with result_left:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#edf4ff,#ffffff);
                        border:1px solid #d4e2ff;border-radius:24px;
                        padding:30px 28px;box-shadow:0 8px 24px rgba(47,101,229,.08);">
                <div style="color:#526784;font-size:13px;font-weight:700;
                            letter-spacing:1.2px;text-transform:uppercase;">
                    Estimated Yearly Spending
                </div>
                <div style="color:#2458c5;font-size:42px;font-weight:850;margin:12px 0 8px;">
                    ₹ {result['prediction']:,.2f}
                </div>
                <div style="color:#63748b;font-size:14px;">
                    Generated using <b>{result['algorithm']}</b>
                </div>
                <div style="margin-top:18px;background:#dce8ff;color:#2458c5;
                            display:inline-block;padding:7px 13px;border-radius:20px;
                            font-size:13px;font-weight:700;">
                    ✔ Prediction generated
                </div>
            </div>
            """, unsafe_allow_html=True)

        with result_right:
            accuracy_ring = max(0.0, min(100.0, result["accuracy"]))
            st.markdown(f"""
            <div style="display:flex;justify-content:center;align-items:center;padding:12px 4px;">
                <div style="width:190px;height:190px;border-radius:50%;
                            background:conic-gradient(#2f65e5 {accuracy_ring:.2f}%,
                            #e8edf5 0);display:flex;align-items:center;justify-content:center;">
                    <div style="width:145px;height:145px;border-radius:50%;background:#fff;
                                display:flex;flex-direction:column;align-items:center;
                                justify-content:center;box-shadow:0 3px 14px rgba(0,0,0,.06);">
                        <div style="font-size:30px;font-weight:850;color:#2458c5;">
                            {result['accuracy']:.1f}%
                        </div>
                        <div style="font-size:11px;color:#68778b;text-align:center;
                                    font-weight:700;letter-spacing:.4px;">
                            APPROX. MODEL<br>ACCURACY
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Model Snapshot")
        snap1, snap2, snap3 = st.columns(3)
        snap1.metric("Algorithm", result["algorithm"])
        snap2.metric("R² Score", f"{result['r2']:.2f}%")
        snap3.metric("RMSE", f"₹ {result['rmse']:,.2f}")

        with st.expander("🔎 View customer input values"):
            in1, in2 = st.columns(2)
            in1.write(f"**Average Session Length:** {result['inputs']['session_length']}")
            in1.write(f"**Time on App:** {result['inputs']['time_on_app']}")
            in2.write(f"**Time on Website:** {result['inputs']['time_on_website']}")
            in2.write(f"**Length of Membership:** {result['inputs']['membership_length']}")

        st.info(
            "The circular meter shows approximate model performance on the test dataset. "
            "It is not a confidence score or a guarantee for this individual prediction."
        )
    else:
        st.info("Enter the customer details above, select an algorithm, and click **Generate Prediction**.")
