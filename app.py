"""Interface Streamlit simple – Churn Telco : données, modèles, prédiction + explication SHAP."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ART = Path(__file__).resolve().parent / "artifacts"
st.set_page_config(page_title="Churn Telco – IA explicable", page_icon="📉", layout="wide")


# ------------------------------------------------------------------ chargement
def retrain(meta):
    """Réentraîne le modèle ici (évite les problèmes de version du fichier .joblib)."""
    df = pd.read_csv(ART / "telco_clean.csv")
    seed = meta.get("seed", 42)
    X, y = df[meta["columns"]], df["Churn"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=meta.get("test_size", 0.2),
                                        random_state=seed, stratify=y)
    p, name = meta["best_params"], meta["model_name"]
    if name == "Régression logistique":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", **p)
    elif name == "Arbre de décision":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **p)
    elif name == "Random Forest":
        clf = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **p)
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        clf = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=seed, n_jobs=-1, **p)
    else:
        raise ValueError(f"Modèle non pris en charge : {name}")
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         meta["num_cols"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                               sparse_output=False))]), meta["cat_cols"]),
    ], verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    return Pipeline([("prep", pre), ("clf", clf)]).fit(X_tr, y_tr)


@st.cache_resource
def load_model():
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    try:
        model = joblib.load(ART / "model.joblib")
    except Exception:  # fichier absent ou créé avec une autre version de scikit-learn
        if "best_params" not in meta:
            st.error("Impossible de charger model.joblib et `best_params` manquant dans meta.json. "
                     "Relance la cellule de mise à jour de meta.json dans le notebook.")
            st.stop()
        with st.spinner("Entraînement du modèle (quelques secondes)..."):
            model = retrain(meta)
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        bg = pd.read_csv(ART / "background.csv")[meta["columns"]]
        explainer = shap.LinearExplainer(clf, prep.transform(bg))
    else:  # arbre, forêt, XGBoost
        explainer = shap.TreeExplainer(clf)
    return model, meta, explainer


@st.cache_data
def load_data():
    return pd.read_csv(ART / "telco_clean.csv")


@st.cache_data
def load_results():
    return pd.read_csv(ART / "results.csv", index_col=0)


# ------------------------------------------------------------------ explications
def explain(model, explainer, meta, row):
    t = model.named_steps["prep"].transform(row)
    s = explainer(t)
    if s.values.ndim == 3:            # forêt / arbre : on garde la classe 1 (départ)
        s = s[:, :, 1]
    shown = t.copy()
    for c in meta["num_cols"]:        # valeurs d'origine (tenure = 3 et non -1.2)
        shown[c] = row[c].values
    return shap.Explanation(values=s.values[0],
                            base_values=float(np.ravel(s.base_values)[0]),
                            data=shown.values[0], feature_names=list(t.columns))


def group_shap(e, meta):
    """Somme des contributions des colonnes encodées, par variable d'origine."""
    v = pd.Series(e.values, index=e.feature_names)
    out = {}
    for c in meta["columns"]:
        cols = [c] if c in meta["num_cols"] else [n for n in v.index if n.startswith(c + "_")]
        out[c] = v[cols].sum()
    return pd.Series(out).sort_values(key=abs, ascending=False)


def synthese(p, thr, g, row):
    val = lambda c: f"{c} = {row[c].iloc[0]:g}" if c in row.select_dtypes("number") else f"{c} = {row[c].iloc[0]}"
    pos = " et ".join(val(c) for c in g[g > 0].head(2).index)
    neg = " et ".join(val(c) for c in g[g < 0].head(2).index)
    if p >= thr:
        s = f"Le risque de départ est **élevé** ({p:.0%}, seuil {thr:.0%}), principalement à cause de {pos}."
        return s + (f" Il est atténué par {neg}." if neg else "")
    s = f"Le risque de départ est **faible** ({p:.0%}, seuil {thr:.0%}), principalement grâce à {neg}."
    return s + (f" Il est augmenté par {pos}." if pos else "")


# ------------------------------------------------------------------ onglets
def tab_data():
    df = load_data()
    rate = df["Churn"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients", f"{len(df):,}")
    c2.metric("Variables", df.shape[1] - 1)
    c3.metric("Taux de départ", f"{rate:.1%}")

    left, right = st.columns(2)
    with left:
        counts = df["Churn"].value_counts().rename({0: "Reste", 1: "Part"})
        st.plotly_chart(px.pie(values=counts.values, names=counts.index, hole=0.4,
                               title="Répartition de la cible"), use_container_width=True)
    with right:
        cats = [c for c in df.select_dtypes(exclude="number").columns]
        var = st.selectbox("Taux de départ par…", cats, index=cats.index("Contract") if "Contract" in cats else 0)
        r = df.groupby(var)["Churn"].mean().reset_index()
        fig = px.bar(r, x=var, y="Churn", text=r["Churn"].map("{:.1%}".format))
        fig.add_hline(y=rate, line_dash="dash", annotation_text=f"Moyenne {rate:.1%}")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    nums = df.select_dtypes("number").columns.drop("Churn")
    v = st.selectbox("Distribution de…", nums)
    st.plotly_chart(px.histogram(df.assign(Churn=df["Churn"].astype(str)), x=v, color="Churn",
                                 barmode="overlay", opacity=0.65, marginal="box"),
                    use_container_width=True)
    with st.expander("Aperçu des données"):
        st.dataframe(df.head(100), use_container_width=True)


def tab_models():
    res = load_results()
    st.subheader("Comparaison des modèles (jeu de test)")
    st.dataframe(res.style.format("{:.3f}"), use_container_width=True)
    metrics = [c for c in ["AUC test", "PR-AUC", "Précision", "Rappel", "F1"] if c in res.columns]
    plot = res[metrics].reset_index().melt(id_vars=res.index.name or "index",
                                           var_name="Métrique", value_name="Valeur")
    st.plotly_chart(px.bar(plot, x=plot.columns[0], y="Valeur", color="Métrique", barmode="group"),
                    use_container_width=True)


def tab_predict(model, meta, explainer, thr):
    st.subheader("Saisir un client")
    with st.form("client"):
        cols = st.columns(3)
        values = {}
        for i, c in enumerate(meta["columns"]):
            with cols[i % 3]:
                if c in meta["num_cols"]:
                    s = meta["num_stats"][c]
                    if s["is_int"]:
                        values[c] = st.number_input(c, int(s["min"]), int(s["max"]), int(s["median"]))
                    else:
                        values[c] = st.number_input(c, float(s["min"]), float(s["max"]), float(s["median"]))
                else:
                    opts = meta["categories"][c]
                    values[c] = st.selectbox(c, opts, index=opts.index(meta["defaults_cat"][c]))
        if st.form_submit_button("Prédire", type="primary"):
            st.session_state["row"] = pd.DataFrame([values])[meta["columns"]]

    row = st.session_state.get("row")
    if row is None:
        st.info("Renseigne les caractéristiques du client puis clique sur « Prédire ».")
        return

    p = float(model.predict_proba(row)[0, 1])
    a, b = st.columns(2)
    a.metric("Probabilité de départ", f"{p:.1%}")
    b.metric("Décision", "⚠️ Risque élevé" if p >= thr else "✅ Risque faible", f"seuil = {thr:.2f}", delta_color="off")
    st.progress(min(max(p, 0.0), 1.0))

    e = explain(model, explainer, meta, row)
    st.markdown(synthese(p, thr, group_shap(e, meta), row))

    st.subheader("Pourquoi cette prédiction ? (SHAP waterfall)")
    shap.plots.waterfall(e, max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Barres rouges : la variable augmente le risque de départ ; barres bleues : elle le diminue. "
               "Les valeurs sont en log-odds pour la régression logistique et XGBoost. "
               "Ces explications décrivent le comportement du modèle, pas des relations de cause à effet.")


# ------------------------------------------------------------------ main
try:
    model, meta, explainer = load_model()
except FileNotFoundError:
    st.error("Dossier `artifacts/` introuvable : exporte le modèle depuis le notebook (cellule d'export).")
    st.stop()

st.title("📉 Prédiction du churn Telco – IA explicable")
st.sidebar.header("Paramètres")
st.sidebar.write(f"Modèle utilisé : **{meta['model_name']}**")
thr = st.sidebar.slider("Seuil de décision", 0.05, 0.95, float(meta["threshold"]), 0.01)

t1, t2, t3 = st.tabs(["📊 Données", "🏆 Modèles", "🔮 Prédiction & explication"])
with t1:
    tab_data()
with t2:
    tab_models()
with t3:
    tab_predict(model, meta, explainer, thr)
    
