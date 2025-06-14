import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import io
from PIL import Image
import os



# Set Streamlit page configuration
st.set_page_config(page_title="Customer Segmentation App", page_icon="📊", layout="wide")

st.markdown(
    """
    <div style="background-color:#f44b45; padding:0.5px; border-radius:10px; text-align:center">
        <h2 style="color:#f8f8fb;  font-weight: 700; margin-bottom:5px;">Customer Segmentation Dashboard</h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #eadede; 
    }
     [data-testid="stSidebar"] {
            background-color: #eadede;  /* Change this to your preferred color */
        }
    </style>
    """,
    unsafe_allow_html=True
)
plt.rcParams['axes.facecolor'] = '#eadede'     # Plot background
plt.rcParams['figure.facecolor'] = '#eadede'  
st.markdown("""
    <style>
        /* Style all download buttons */
        div[data-testid="stDownloadButton"] > button {
            background-color: #f44b45;
            color: white;
            border-radius: 6px;
            font-weight: 500;
            padding: 0.5em 1.5em;
            border: none;
            transition: background-color 0.3s ease;
        }

        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #f44b45;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# for styling of tables
st.markdown("""
    <style>
        table {
            background-color: #eadede ;
            color: black;
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            padding: 10px;
            text-align: center      ;
            border: 1px solid #040404;
        }
        th {
            background-color: #f98383;
        }
    </style>
""", unsafe_allow_html=True)


# Load saved model and dataset
try :
    model = joblib.load("models/kmeans_customer_segmentation.pkl")
except FileNotFoundError:
    st.warning('Model is not found, plz save the model first.')
df = pd.read_csv("data/Mall_Customers.csv")
try :
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    st.warning('Scalar object is not found, plz save the object first.')
scaled_features = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
df_scaled = pd.DataFrame(scaled_features, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
df['Label'] = model.labels_
cluster_labels = {
    0: "Moderate Income, Moderate Spending",
    1: "High Income, High Spending",
    2: "Young Customers, Moderate Spending",
    3: "High Income, Low Spending"
}
df['Segment'] = df['Label'].map(cluster_labels)

st.session_state['clustered_df'] = df  # 👈 this saves it for download later

# Sidebar navigation

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Dataset", "EDA", "Clustering", "Predict New Customer", "Download Results", "About"],
        icons=["house", "table", "bar-chart", "grid-3x3-gap", "person-plus", "cloud-download", "info-circle"],
        default_index=0,
        styles={
            "container": {
                "padding": "5   px",
                "background-color": "#dccac8",
                "border-radius": "10px",
                "box-shadow": "2px 2px 10px rgba(0, 0, 0, 0.05)",
                "border": "2px solid black"
            },
            "icon": {
                "color": "#ff5e5e",
                "font-size": "26px"
            },
            "nav-link": {
                "font-size": "15x",
                "font-weight": "500",
                "text-align": "left",
                "margin": "4px 0",
                "color": "#333",
                "--hover-color": "#ffe6e6",
                "border-radius": "8px",
            },
            "nav-link-selected": {
                "background-color": "#ff5e5e",
                "color": "#fff",
                "font-weight": "700",
                "font-size" : "20px"
            }
        }
    )


# Home Section
if selected == "Home":  
    st.image("images/banner.jpg", width=800)

    st.markdown("""
        <style>
            /* Custom fonts for header and body */
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Open+Sans&display=swap');

            h2, h3 {
                font-family: 'Montserrat', sans-serif;
                color: #2c3e50;
            }

            p, li {
                font-family: 'Open Sans', sans-serif;
                font-size: 16px;
                color: #333333;
                line-height: 1.6;
            }

            ul {
                margin-left: 20px;
            }

            h2 {
                font-size: 30px;
                margin-top: 35px;
            }

            h3 {
                font-size: 22px;
                margin-top: 25px;
            }
        </style>

        <h2>Customer Segmentation Dashboard</h2>

        <p>
            This interactive dashboard helps businesses better understand their customer base using
            <strong>K-Means Clustering</strong>. By analyzing data like age, income, and spending score,
            customers are grouped into insightful clusters that can guide personalized marketing and
            strategic planning.
        </p>

        <h3> Project Objective</h3>
        <ul>
            <li>Segment customers based on behavioral and demographic features.</li>
            <li>Visualize patterns and groupings to support data-driven decisions.</li>
            <li>Predict the segment of a new customer for personalized marketing.</li>
        </ul>

        <h3> Dataset Overview</h3>
        <ul>
            <li><strong>Gender</strong></li>
            <li><strong>Age</strong></li>
            <li><strong>Annual Income (k$)</strong></li>
            <li><strong>Spending Score (1–100)</strong></li>
        </ul>
        <p>This dataset includes 200 mall customers and is commonly used for clustering tasks.</p>

        <h3> How to Use the Dashboard</h3>
        <ul>
            <li>Use the sidebar to navigate through each section.</li>
            <li>Explore data visualizations and customer segments.</li>
            <li>Predict a new customer’s segment and receive marketing suggestions.</li>
        </ul>
    """, unsafe_allow_html=True)


# Dataset Section
if selected == "Dataset":
    st.title(" Dataset Overview")
    st.markdown(df.head().to_html(index=False, escape=False), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file to analyze your own customer data.", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset successfully uploaded!")
        st.markdown(df.head(10).to_html(index=False, escape=False), unsafe_allow_html=True)

        if all(col in df.columns for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
            if st.button("Cluster Uploaded Data"):
                uploaded_scaled = scaler.transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
                uploaded_labels = model.predict(uploaded_scaled)
                df['Label'] = uploaded_labels
                df['Segment'] = df['Label'].map(cluster_labels)
                st.session_state['clustered_df'] = df
                st.success("✅ Clustering completed on uploaded dataset!")
                # st.dataframe(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Label', 'Segment']])
                st.markdown(df.head().to_html(index=False, escape=False), unsafe_allow_html=True)
        else:
            st.error("Uploaded dataset must include: Age, Annual Income (k$), and Spending Score (1-100)")


    st.title(   "Basic Statistics")
    st.markdown(df.describe().to_html(index=False, escape=False), unsafe_allow_html=True)

    st.title("Customer Feature Distributions")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['Age'], kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title("Age Distribution")
    sns.histplot(df['Annual Income (k$)'], kde=True, ax=ax[1], color='lightgreen')
    ax[1].set_title("Income Distribution")
    sns.histplot(df['Spending Score (1-100)'], kde=True, ax=ax[2], color='salmon')
    ax[2].set_title("Spending Score Distribution")
    st.pyplot(fig)

# EDA Section
if selected == "EDA":
    st.title(" Exploratory Data Analysis")

    st.header("Correlation Heatmap")
    st.caption("Shows how customer attributes relate to one another. Darker cells indicate stronger relationships.")

    corr = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
    fig, ax = plt.subplots(figsize = (10,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.header("Age vs Income")
    st.caption("Explore how Age and Income are distributed across different customer segments.")

    fig1 = px.scatter(df, x='Age', y='Annual Income (k$)', color='Segment', width=800, height=400 )
    fig1.update_layout(
    paper_bgcolor="#eadede",   # Outer background (around chart)
    plot_bgcolor="#f9f9f9",    # Chart area background
)
    st.plotly_chart(fig1)

    st.header("Spending Score vs Income")
    st.caption("Helps to see how spending behavior relates to income across segments.")

    fig2 = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', color='Segment'
                        , width=800, height=400)
    fig2.update_layout(
    paper_bgcolor="#eadede",   # Outer background (around chart)
    plot_bgcolor="#f9f9f9",    # Chart area background
)
    st.plotly_chart(fig2)

    st.header("Customer Segment Distribution")
    st.caption("Shows the size of each cluster based on customer count.")

    pie_chart = px.pie(df, names='Segment', title='Cluster Distribution')
    pie_chart.update_layout(
    paper_bgcolor="#eadede",   # Outer background (around chart)
    plot_bgcolor="#f9f9f9",    # Chart area background
)
    st.plotly_chart(pie_chart)
    st.header(" Cluster Segment Descriptions")
    desc_df = pd.DataFrame.from_dict(cluster_labels, orient='index', columns=['Description']).reset_index()
    desc_df.columns = ['Cluster', 'Description']
    st.markdown(desc_df.to_html(index=False, escape=False), unsafe_allow_html=True)


# Clustering Section
if selected == "Clustering":
    st.title(" Clustering Visualizer")

    option = st.selectbox("Select Method for Dimensionality Reduction", ["PCA", "t-SNE"])

    if option == "PCA":
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(df_scaled)
    else:
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(df_scaled)

    df_viz = pd.DataFrame(reduced, columns=['Component 1', 'Component 2'])
    df_viz['Cluster'] = df['Label']

    st.subheader(f"2D Clustering using {option}")
    st.caption("This 2D plot simplifies customer data while preserving group similarity.")

    fig = px.scatter(df_viz, x='Component 1', y='Component 2', color=df_viz['Cluster'].astype(str),
                     title=f"Customer Clusters Visualized with {option}",
                     labels={'color': 'Cluster'})
    fig.update_layout(
    paper_bgcolor="#eadede",   # Outer background (around chart)
    plot_bgcolor="#f9f9f9",    # Chart area background
    )
    st.plotly_chart(fig)

    st.subheader("Manual Feature Pair Selection")
    st.caption("Visually compare clusters by choosing any two customer features.")

    feature_x = st.selectbox("Select X-axis feature", ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    feature_y = st.selectbox("Select Y-axis feature", ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

    fig2 = px.scatter(df, x=feature_x, y=feature_y, color=df['Label'].astype(str),
                      title=f"Clusters by {feature_x} vs {feature_y}")
    fig2.update_layout(
    paper_bgcolor="#eadede",   # Outer background (around chart)
    plot_bgcolor="#f9f9f9",    # Chart area background
)
    st.plotly_chart(fig2)
    
    st.header(" K-Means Clustering Summary")

    # Elbow Method Plot
    st.subheader(" Elbow Method for Optimal K")
    st.markdown("The Elbow Method helps to find the optimal number of clusters by observing the point where inertia decreases minimally.")
    try:
        elbow_img = Image.open("images/elbow_plot.png")
        st.image(elbow_img, caption="Elbow Plot - Optimal K", use_container_width=True)
    except FileNotFoundError:
        st.warning("Elbow plot image not found. Please upload or generate it.")

    # Inertia & Silhouette Score
    st.subheader(" Clustering Metrics")
    st.markdown(f"**Inertia:** {model.inertia_}")
    
    try:
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(df_scaled, model.labels_)
        st.markdown(f"**Silhouette Score:** {silhouette:.3f}")
    except:
        st.warning("Silhouette Score could not be calculated.")

    # Cluster Centroids
    st.subheader(" Cluster Centroids (Scaled)")
    st.caption("The average location of each cluster in the scaled feature space.")

    centroids_df = pd.DataFrame(model.cluster_centers_, columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    st.markdown(centroids_df.style.highlight_max(axis=0).to_html(index=False, escape=False), unsafe_allow_html=True)

    # Optional: Inverse scale centroids for interpretability
    st.subheader(" Cluster Centroids (Original Scale)")
    st.caption("Centroid positions converted back to the original units for interpretation.")

    centroids_original = scaler.inverse_transform(model.cluster_centers_)
    centroids_original_df = pd.DataFrame(centroids_original, columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    st.markdown(centroids_original_df.style.highlight_max(axis=0).to_html(index=False, escape=False), unsafe_allow_html=True)


elif selected == "Predict New Customer":
    st.title(" Predict Customer Segment")
    st.markdown("Enter the customer details below and click **Group** to see their segment.")

    # Input form
    with st.form(key='customer_form'):
        age = st.slider("Select Age", min_value=18, max_value=70, value=30)
        income = st.number_input("Enter Annual Income (k$)", min_value=10, max_value=150, value=50)
        score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
        submit_button = st.form_submit_button(label='Group')

    if submit_button:
        input_df = pd.DataFrame([[age, income, score]], columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Predict using trained model
        cluster = model.predict(scaled_input)[0]
        st.caption("The model predicts the nearest customer segment based on the provided input.")


        # Mapping clusters to labels
        cluster_labels = {
            0: "Moderate Income, Moderate Spending",
            1: "High Income, High Spending",
            2: "Young Customers, Moderate Spending",
            3: "High Income, Low Spending"
        }
        # Suggested strategies for each segment
        marketing_strategies = {
            0: "Focus on building trust through personalized customer service and loyalty programs.",
            1: "Promote high-end products with exclusive discounts, early access sales, and VIP events.",
            2: "Offer bundle deals, coupons, and referral bonuses to attract budget-conscious buyers.",
            3: "Highlight product quality and long-term value; consider elite membership perks."
        }
        suggestion = marketing_strategies.get(cluster, "No strategy found for this segment.")

        cluster_text = f"This customer belongs to <strong>Cluster {cluster}</strong>: <em>{cluster_labels.get(cluster, 'Unknown Segment')}</em>"


        st.markdown(f"""
                    <div style="background-color: #dccac8; color: black; padding: 15px; 
                        border-left: 5px solid #f44b45; border-radius: 8px; font-size: 16px;">
                <h5>{cluster_text}</h5>
            </div>
            <hr>
                    <div style="background-color: #dccac8; padding: 15px; border-left: 5px solid #f44b45; border-radius: 10px;">
                                    <h4 style="margin-bottom: 5px;"> Recommended Strategy</h4>
                        <h5>{suggestion}</h5>
                    </div>
""", unsafe_allow_html=True)
        # Show input
        st.subheader("Entered Data")
        st.markdown(input_df.to_html(index=False, escape=False), unsafe_allow_html=True)

        # Explainable AI (XAI) Section
        st.subheader(" Why This Cluster? (Explainable AI)")
        st.caption("Displays how far this customer is from each cluster center. Closest = predicted group.")


        # Compute distances to all centroids
        distances = np.linalg.norm(model.cluster_centers_ - scaled_input, axis=1)

        # Create a dataframe to display distances to each cluster center
        distance_df = pd.DataFrame({
            'Cluster': list(range(len(distances))),
            'Distance from Input': distances
        }).sort_values(by='Distance from Input')

        # Highlight the predicted cluster
        distance_df['Closest'] = ['✅' if i == cluster else '' for i in distance_df['Cluster']]

        # Show as table
        st.markdown(distance_df.style.highlight_min(subset=['Distance from Input'], color='lightgreen').to_html(index=False, escape=False), unsafe_allow_html=True)

        # Optional: Plot distances as bar chart
        fig_dist = px.bar(distance_df, x='Cluster', y='Distance from Input',
                        color=distance_df['Cluster'].astype(str),
                        title="Distance to Cluster Centers",
                        labels={'Distance from Input': 'Euclidean Distance'})
        fig_dist.update_layout(
    paper_bgcolor="#eadede",   # Outer background (around chart)
    plot_bgcolor="#f9f9f9",    # Chart area background
)
        st.plotly_chart(fig_dist)


elif selected == "Download Results":
    st.title(" Download Results")
    st.caption("Download clustered data and visual plots for reporting or further analysis.")


    # Check if clustered data exists
    if 'clustered_df' in st.session_state:
        st.subheader(" Download Clustered Dataset")
        csv = st.session_state['clustered_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name='clustered_customers.csv',
            mime='text/csv'
        )
    else:
        st.info("Upload and cluster a dataset first to enable download.")

    st.subheader(" Download Visualizations")
    st.markdown("Click below to download key plots:")

    # Download Elbow Plot
    with open("images/elbow_plot.png", "rb") as file:
        st.download_button(
            label="Download Elbow Plot (PNG)",
            data=file,
            file_name="elbow_plot.png",
            mime="image/png"
        )

    # Download Cluster Plot (2D/3D)
    if os.path.exists("images/cluster_plot.png"):
        with open("cluster_plot.png", "rb") as file:
            st.download_button(
                label="Download Cluster Visualization (PNG)",
                data=file,
                file_name="cluster_plot.png",
                mime="image/png"
            )
    else:
        st.info("Cluster visualization plot not available yet.")

    st.markdown("---")
    st.markdown("🔁 You can generate plots from the **EDA** or **Clustering** tabs before downloading.")

elif selected == "About":
    st.title(" About This Project")
    
    st.markdown("""
    Welcome to the **Mall Customer Segmentation App** – a data-driven tool designed to help businesses understand and classify their customers based on their:

    -  **Age**
    -  **Annual Income**
    -  **Spending Score**

    This application applies **K-Means Clustering** to identify groups of customers with similar characteristics, enabling more focused and effective business strategies.
    """)

    st.markdown("---")

    st.subheader(" Tools & Technologies Used")

    st.markdown(
        """



        <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" width="40" title="Python"/>
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" alt="Pandas" width="40" title="Pandas"/>
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" alt="NumPy" width="40" title="NumPy"/>
            <img src="https://matplotlib.org/stable/_static/logo2.svg" alt="Matplotlib" width="80" title="Matplotlib"/>
            <img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" alt="Seaborn" width="80" title="Seaborn"/>
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Plotly-logo.png" alt="Plotly" width="80" title="Plotly"/>
            <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-learn" width="40" title="Scikit-learn"/>
            <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit" width="100" title="Streamlit"/>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("---")

    st.subheader(" Developer")
    st.markdown("""
    Developed by **Muhammad Taha**, a Software Engineering student at **University of Engineering and Technology Taxila**.

    **Interests:**  
    - Data Science & Machine Learning  
    - Business Intelligence  
    - Web App Development
    ---
    ### Feel free to connect, give feedback, or explore the project further!
    """)

    st.markdown("""
        <div style="display: flex;  justify-content: center; gap: 60px; align-items: center;">
            <a href='https://github.com/MuhammadTaha1038' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg' alt='GitHub' width='100' height='100'>
            </a>
            <a href='https://linkedin.com/in/muhammad-taha-b88807248/' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' alt='LinkedIn' width='100' height='100'>
            </a>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 0px; border: 0; border-top: 1px solid #bbb;">
    <div style="text-align: center; font-size: 14px; color: gray;">
        <p>© 2025 Customer Segmentation Dashboard</p>
        <p>Developed by Muhammad Taha</p>
        <a href="https://github.com/MuhammadTaha1038" target="_blank" style="text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="24" style="margin-right:10px;" />
        </a>
        <a href="https://linkedin.com/in/muhammad-taha-b88807248/" target="_blank" style="text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="24" />
        </a>
    </div>
""", unsafe_allow_html=True)


