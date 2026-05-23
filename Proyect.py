import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Page config 
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background:#fff0f3; }
  div[data-testid="stMetric"] {
      background:#fff0f3; border-radius:10px;
      padding:.8rem 1rem; border:1px solid #f9a8c9;
  }
  div[data-testid="stMetric"] label { color:#c2185b !important; }
  div[data-testid="stMetric"] div   { color:#880e4f !important; font-size:1.6rem; }
  h1,h2,h3 { color:#880e4f; }
</style>
""", unsafe_allow_html=True)

# Load & train 
@st.cache_data
def load_and_train():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species_id"] = iris.target
    df["species"]    = [iris.target_names[i] for i in iris.target]

    X, y = iris.data, iris.target
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

    metrics = dict(
        accuracy  = accuracy_score(y_te, y_pred),
        precision = precision_score(y_te, y_pred, average="weighted"),
        recall    = recall_score(y_te, y_pred, average="weighted"),
        f1        = f1_score(y_te, y_pred, average="weighted"),
        cv_mean   = scores.mean(),
        cv_std    = scores.std(),
    )
    return df, model, scaler, iris, metrics, (X_te, y_te, y_pred), scores

df, model, scaler, iris, metrics, test_data, cv_scores = load_and_train()
FEAT  = iris.feature_names          # full names with "(cm)"
NAMES = iris.target_names.tolist()  # ['setosa','versicolor','virginica']
COLORS = {"setosa":"#e91e8c","versicolor":"#ff80ab","virginica":"#ad1457"}

# Sidebar – prediction inputs 
st.sidebar.title("🌸 Predict a Flower")
st.sidebar.markdown("Move the sliders and press **Classify**.")
sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sw = st.sidebar.slider("Sepal Width  (cm)", 2.0, 4.5, 3.0, 0.1)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
pw = st.sidebar.slider("Petal Width  (cm)", 0.1, 2.5, 1.2, 0.1)
classify_btn = st.sidebar.button("🌸 Classify Flower", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Sheila Daniela Hernandez Carrillo - 18038")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** Random Forest (200 trees)  \n"
    "**Scaling:** StandardScaler  \n"
    "**Split:** 75/25 + 5-fold CV"
)

# Header 
st.title("🌸 Iris Species Classification Dashboard")
st.markdown("*Data Mining Final Project · Universidad de la Costa · José Escorcia-Gutierrez, Ph.D.*")
st.markdown("---")

# TABS 
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Metrics", "🔍 Predict & Visualize",
    "📈 Data Exploration", "🌳 Feature Analysis"
])


# TAB 1 · MODEL METRICS
with tab1:
    st.subheader("Performance on test set (25 % held-out, stratified)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall",    f"{metrics['recall']:.4f}")
    c4.metric("F1-Score",  f"{metrics['f1']:.4f}")

    st.markdown("#### 5-Fold Cross-Validation")
    col_cv, col_cm = st.columns(2)

    with col_cv:
        fig_cv = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores,
            marker_color="#e91e8c",
            text=[f"{s:.4f}" for s in cv_scores],
            textposition="outside",
        ))
        fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash",
                         line_color="#880e4f",
                         annotation_text=f"Mean = {cv_scores.mean():.4f}")
        fig_cv.update_layout(
            title=f"CV Accuracy  (mean={metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f})",
            yaxis_range=[0.85, 1.02], height=350,
            plot_bgcolor="#fff0f3", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    with col_cm:
        X_te, y_te, y_pred = test_data
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=NAMES, y=NAMES,
            colorscale=[[0,"#fff0f3"],[0.5,"#f48fb1"],[1,"#880e4f"]],
            text=cm, texttemplate="%{text}",
            textfont={"size":18},
            showscale=False,
        ))
        fig_cm.update_layout(
            title="Confusion Matrix – test set",
            xaxis_title="Predicted", yaxis_title="Actual",
            height=350, plot_bgcolor="#fff0f3", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Per-class Report")
    report = classification_report(y_te, y_pred, target_names=NAMES, output_dict=True)
    rdf = pd.DataFrame(report).T.iloc[:3, :4].round(4)
    rdf.index.name = "Species"
    st.dataframe(
        rdf.style.background_gradient(cmap="RdPu", vmin=0.85, vmax=1.0)
               .format("{:.4f}"),
        use_container_width=True,
    )


# TAB 2 · PREDICT & VISUALIZE
with tab2:
    st.info("Use the **sidebar sliders** to set measurements, then press **Classify Flower**.")

    if classify_btn:
        sample        = np.array([[sl, sw, pl, pw]])
        sample_scaled = scaler.transform(sample)
        pred_id       = model.predict(sample_scaled)[0]
        pred_name     = NAMES[pred_id]
        proba         = model.predict_proba(sample_scaled)[0]

        st.success(f"### 🌸 Predicted species: *Iris {pred_name}*")

        p1, p2, p3 = st.columns(3)
        for col, sp, pr in zip([p1,p2,p3], NAMES, proba):
            col.metric(f"Iris {sp}", f"{pr:.1%}")

        # 3-D scatter
        st.markdown("#### 3D Scatter – Your sample vs the dataset")
        fig3d = px.scatter_3d(
            df, x=FEAT[0], y=FEAT[2], z=FEAT[3],
            color="species", color_discrete_map=COLORS,
            opacity=0.65, template="plotly_white",
        )
        fig3d.add_trace(go.Scatter3d(
            x=[sl], y=[pl], z=[pw],
            mode="markers+text",
            marker=dict(size=14, color="#FFD700",
                        line=dict(color="black", width=2)),
            text=[f"★ {pred_name}"], textposition="top center",
            name="Your sample",
        ))
        fig3d.update_layout(height=540,
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3d, use_container_width=True)

        # Radar
        st.markdown("#### Feature Profile vs Species Means")
        means = df.groupby("species")[list(FEAT)].mean()
        SHORT = ["Sepal Length","Sepal Width","Petal Length","Petal Width"]
        fig_r = go.Figure()
        for sp in NAMES:
            v = means.loc[sp].tolist()
            fig_r.add_trace(go.Scatterpolar(
                r=v+[v[0]], theta=SHORT+[SHORT[0]],
                fill="toself", name=f"Iris {sp}",
                line_color=COLORS[sp], opacity=0.45,
            ))
        fig_r.add_trace(go.Scatterpolar(
            r=[sl,sw,pl,pw,sl], theta=SHORT+[SHORT[0]],
            fill="toself", name="Your sample",
            line_color="#FFD700", line_width=3,
        ))
        fig_r.update_layout(
            polar=dict(bgcolor="#fff0f3"),
            paper_bgcolor="rgba(0,0,0,0)", height=420,
        )
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        # Default 3-D without highlight
        fig3d = px.scatter_3d(
            df, x=FEAT[0], y=FEAT[2], z=FEAT[3],
            color="species", color_discrete_map=COLORS,
            opacity=0.7, template="plotly_white",
            title="3D scatter of the Iris dataset",
        )
        fig3d.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("← Set measurements in the sidebar and press **Classify Flower**.")


# TAB 3 · DATA EXPLORATION
with tab3:
    st.subheader("Dataset overview")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Samples", 150)
    col_s2.metric("Features", 4)
    col_s3.metric("Classes", 3)

    st.dataframe(
        df[list(FEAT)].describe().round(3)
          .style.background_gradient(cmap="RdPu"),
        use_container_width=True,
    )

    st.markdown("#### Feature Distributions")
    feat_sel = st.selectbox("Select feature:", FEAT)
    fig_h = px.histogram(
        df, x=feat_sel, color="species",
        color_discrete_map=COLORS, barmode="overlay",
        nbins=25, opacity=0.75, template="plotly_white",
        title=f"Distribution of {feat_sel}",
    )
    fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("#### Scatter Matrix")
    fig_sm = px.scatter_matrix(
        df, dimensions=list(FEAT), color="species",
        color_discrete_map=COLORS, opacity=0.65,
        template="plotly_white",
    )
    fig_sm.update_traces(diagonal_visible=False, showupperhalf=False)
    fig_sm.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=620)
    st.plotly_chart(fig_sm, use_container_width=True)

    st.markdown("#### Box Plots – all features")
    fig_b = px.box(
        df.melt(id_vars="species", value_vars=list(FEAT),
                var_name="Feature", value_name="Value"),
        x="Feature", y="Value", color="species",
        color_discrete_map=COLORS, template="plotly_white",
    )
    fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420)
    st.plotly_chart(fig_b, use_container_width=True)


# TAB 4 · FEATURE ANALYSIS
with tab4:
    st.subheader("Feature Importance (Random Forest)")
    imp = model.feature_importances_
    SHORT4 = ["Sepal Length","Sepal Width","Petal Length","Petal Width"]
    fig_i = go.Figure(go.Bar(
        x=imp, y=SHORT4, orientation="h",
        marker_color=["#e91e8c","#f48fb1","#ad1457","#ff80ab"],
        text=[f"{v:.4f}" for v in imp], textposition="outside",
    ))
    fig_i.update_layout(
        yaxis={"categoryorder":"total ascending"},
        xaxis_title="Mean Decrease in Impurity",
        plot_bgcolor="#fff0f3", paper_bgcolor="rgba(0,0,0,0)", height=320,
    )
    st.plotly_chart(fig_i, use_container_width=True)

    st.subheader("PCA – 2D Projection")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaler.transform(df[list(FEAT)].values))
    df_pca = pd.DataFrame(coords, columns=["PC1","PC2"])
    df_pca["species"] = df["species"].values
    fig_pca = px.scatter(
        df_pca, x="PC1", y="PC2", color="species",
        color_discrete_map=COLORS, opacity=0.75,
        template="plotly_white",
        title=f"PCA 2D · explained variance: {pca.explained_variance_ratio_.sum():.1%}",
    )
    fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420)
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption(
        f"PC1 = {pca.explained_variance_ratio_[0]:.1%}   "
        f"PC2 = {pca.explained_variance_ratio_[1]:.1%}"
    )

    st.subheader("Feature Correlation Heatmap")
    corr = df[list(FEAT)].corr()
    fig_c = go.Figure(go.Heatmap(
        z=corr.values, x=SHORT4, y=SHORT4,
        colorscale="RdPu", zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
        textfont={"size":14},
    ))
    fig_c.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=420,
        title="Pearson correlation between features",
    )
    st.plotly_chart(fig_c, use_container_width=True)

# Footer 
st.markdown("---")
st.caption("Universidad de la Costa · Data Mining · Iris Species Classification · 2025")