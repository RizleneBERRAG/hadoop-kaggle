import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession

st.set_page_config(page_title="VTC Boston — Lakehouse", layout="wide")
st.title("Le prix des courses VTC à Boston : Uber vs Lyft")
st.caption(
    "Dataset Kaggle `ravi72munde/uber-lyft-cab-prices` (courses × météo horaire, 26 nov. – 18 déc. 2018). "
    "Ce tableau de bord lit **uniquement** les tables pré-calculées de la couche Gold (`/lakehouse/gold/`)."
)

GOLD = "/lakehouse/gold"


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("StreamlitGoldReader")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@st.cache_data(ttl=60)
def load_gold_table(hdfs_path: str):
    spark = get_spark()
    return spark.read.format("delta").load(hdfs_path).toPandas()


def explication(sql: str, justification: str):
    with st.expander("Requête SQL & Justification statistique"):
        st.code(sql.strip(), language="sql")
        st.markdown(justification)


try:
    resume = load_gold_table(f"{GOLD}/resume_distributions")
    pareto = load_gold_table(f"{GOLD}/pareto_trajets")
    jours = load_gold_table(f"{GOLD}/evolution_journaliere").sort_values("date_course")
    horaire = load_gold_table(f"{GOLD}/profil_horaire")
    meteo = load_gold_table(f"{GOLD}/effet_meteo")
    duel = load_gold_table(f"{GOLD}/offre_moins_chere_par_quartier")
    segments = load_gold_table(f"{GOLD}/segments_trajets")
    tests = load_gold_table(f"{GOLD}/tests_statistiques")
except Exception as exc:  # tables pas encore produites par les notebooks J3 / J4
    st.error(f"Tables Gold introuvables : exécutez d'abord les notebooks J3 et J4.\n\n`{exc}`")
    st.stop()

# ------------------------------------------------------------------ Filtres
st.sidebar.header("Filtres")
compagnies = st.sidebar.multiselect("Compagnie", ["Lyft", "Uber"], default=["Lyft", "Uber"])
if not compagnies:
    st.sidebar.warning("Choisissez au moins une compagnie.")
    st.stop()
top_n = st.sidebar.slider("Trajets affichés (Pareto)", min_value=10, max_value=len(pareto), value=min(30, len(pareto)), step=5)
gammes = sorted(duel["gamme"].unique())
gamme_choisie = st.sidebar.selectbox("Gamme (duel Uber / Lyft)", gammes, index=gammes.index("Standard") if "Standard" in gammes else 0)

# ------------------------------------------------------------------ Indicateurs clés
prix = resume.set_index("variable").loc["price"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Courses analysées", f"{int(pareto['nb_courses'].sum()):,}".replace(",", " "))
c2.metric("Prix médian", f"{prix['p50_mediane']:.2f} $")
c3.metric("Prix moyen", f"{prix['moyenne']:.2f} $", f"+{prix['ecart_moyenne_mediane']:.2f} $ vs médiane", delta_color="off")
c4.metric("Écart interquartile (IQR)", f"{prix['iqr']:.2f} $")
c5.metric("Asymétrie (skewness)", f"{prix['skewness']:.2f}")
explication(
    """
SELECT variable, COUNT(*) AS nb, AVG(valeur) AS moyenne, STDDEV(valeur) AS ecart_type,
       SKEWNESS(valeur) AS skewness, PERCENTILE_APPROX(valeur, 0.5) AS p50_mediane,
       PERCENTILE_APPROX(valeur, 0.75) - PERCENTILE_APPROX(valeur, 0.25) AS iqr
FROM (SELECT 'price' AS variable, price AS valeur FROM v_silver UNION ALL ...) GROUP BY variable
    """,
    "Le prix est **asymétrique à droite** (skewness > 0) : quelques courses premium très chères tirent la moyenne "
    "au-dessus de la médiane. Le tableau de bord affiche donc la **médiane** et l'**IQR**, robustes aux extrêmes, "
    "plutôt que la moyenne et l'écart-type.",
)

# ------------------------------------------------------------------ Onglets
t1, t2, t3, t4, t5 = st.tabs(["Trajets (Pareto)", "Heures & jours", "Météo", "Uber vs Lyft", "Segments & tests"])

with t1:
    st.subheader("Concentration du chiffre d'affaires simulé par trajet")
    p = pareto.sort_values("rang_ca").head(top_n)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=p["trajet"], y=p["ca_total"], name="CA du trajet ($)", marker_color="#3182bd")
    fig.add_scatter(x=p["trajet"], y=p["pareto_cumul_pct"], name="% cumulé du CA", mode="lines+markers",
                    marker_color="#de2d26", secondary_y=True)
    fig.add_hline(y=80, line_dash="dot", line_color="#de2d26", secondary_y=True)
    fig.update_layout(height=470, xaxis_tickangle=-60, legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title_text="CA ($)", secondary_y=False)
    fig.update_yaxes(title_text="% cumulé", range=[0, 105], secondary_y=True)
    st.plotly_chart(fig, width="stretch")
    seuil = pareto[pareto["pareto_cumul_pct"] >= 80].sort_values("rang_ca").iloc[0]
    st.markdown(f"**80 % du CA** est atteint avec **{int(seuil['rang_ca'])} trajets sur {len(pareto)}** "
                f"({seuil['pct_trajets_cumul']:.0f} % des trajets) : on est loin d'une loi 80/20.")
    explication(
        """
WITH par_trajet AS (
    SELECT trajet, COUNT(*) AS nb_courses, SUM(price) AS ca_total,
           PERCENTILE_APPROX(price, 0.5) AS prix_median,
           PERCENTILE_APPROX(price, 0.75) - PERCENTILE_APPROX(price, 0.25) AS iqr_prix
    FROM v_silver GROUP BY trajet)
SELECT *, DENSE_RANK() OVER (ORDER BY ca_total DESC) AS rang_ca,
       100 * SUM(ca_total) OVER (ORDER BY ca_total DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
           / SUM(ca_total) OVER () AS pareto_cumul_pct
FROM par_trajet
        """,
        "Courbe de **Pareto** : cumul des CA triés par ordre décroissant. Les données ont été collectées en interrogeant "
        "les API à intervalles réguliers sur tous les trajets, d'où une répartition très homogène : le CA dépend surtout "
        "de la **distance** du trajet, pas de sa popularité.",
    )

with t2:
    st.subheader("Prix médian selon l'heure de Boston")
    h = horaire[horaire["cab_type"].isin(compagnies)].sort_values("heure_boston")
    fig = px.line(h, x="heure_boston", y="prix_median", color="cab_type", markers=True,
                  labels={"heure_boston": "heure (Boston)", "prix_median": "prix médian ($)", "cab_type": "compagnie"},
                  color_discrete_map={"Lyft": "#e7298a", "Uber": "#252525"})
    fig.update_layout(height=380, xaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch")
    explication(
        """
SELECT heure_boston, cab_type, COUNT(*) AS nb_courses,
       PERCENTILE_APPROX(price, 0.5) AS prix_median, 100 * AVG(is_surge) AS pct_majorees
FROM v_silver GROUP BY heure_boston, cab_type
        """,
        "Heure **convertie en heure de Boston** (les horodatages du dataset sont en UTC, 5 h d'écart). "
        "La médiane bouge très peu d'une heure à l'autre : l'heure seule explique peu le prix (cf. test de Kruskal-Wallis).",
    )

    st.subheader("Évolution jour par jour")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=jours["date_course"], y=jours["prix_moyen"], name="prix moyen du jour", mode="markers+lines",
                    line=dict(color="#9ecae1"))
    fig.add_scatter(x=jours["date_course"], y=jours["prix_moyen_glissant_3j"], name="moyenne glissante 3 jours",
                    line=dict(color="#08519c", width=3))
    fig.add_bar(x=jours["date_course"], y=jours["pct_sous_pluie"], name="% de courses sous la pluie",
                marker_color="#bdbdbd", opacity=0.45, secondary_y=True)
    fig.update_layout(height=400, legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="prix ($)", secondary_y=False)
    fig.update_yaxes(title_text="% sous la pluie", secondary_y=True)
    st.plotly_chart(fig, width="stretch")
    trous = jours[jours["jours_depuis_precedent"] > 1]
    if len(trous):
        st.warning("Collecte interrompue avant : " + ", ".join(pd.to_datetime(trous["date_course"]).dt.strftime("%d/%m")))
    explication(
        """
WITH par_jour AS (SELECT date_course, COUNT(*) AS nb_courses, AVG(price) AS prix_moyen,
                         100 * AVG(is_pluie) AS pct_sous_pluie FROM v_silver GROUP BY date_course)
SELECT *, DATEDIFF(date_course, LAG(date_course, 1) OVER (ORDER BY date_course)) AS jours_depuis_precedent,
       AVG(prix_moyen) OVER (ORDER BY date_course ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS prix_moyen_glissant_3j
FROM par_jour
        """,
        "`LAG()` compare chaque jour au précédent ; `jours_depuis_precedent` révèle les **trous de collecte** "
        "(une variation calculée par-dessus un trou n'est pas une évolution d'un jour à l'autre). "
        "La moyenne glissante sur 3 jours lisse le bruit quotidien.",
    )

with t3:
    st.subheader("La pluie change-t-elle le prix ?")
    m = meteo[meteo["cab_type"].isin(compagnies)]
    fig = px.bar(m, x="gamme", y="prix_median", color="meteo", barmode="group", facet_col="cab_type",
                 color_discrete_map={"Sec": "#9ecae1", "Pluie": "#08519c"},
                 labels={"prix_median": "prix médian ($)", "gamme": ""}, hover_data=["nb_courses", "pct_majorees"])
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    explication(
        """
SELECT cab_type, gamme, CASE WHEN is_pluie = 1 THEN 'Pluie' ELSE 'Sec' END AS meteo,
       COUNT(*) AS nb_courses, PERCENTILE_APPROX(price, 0.5) AS prix_median, 100 * AVG(is_surge) AS pct_majorees
FROM v_silver GROUP BY cab_type, gamme, is_pluie
        """,
        "La météo vient de la **2e table** (`weather.csv`), jointe par **quartier + heure** après pré-agrégation "
        "(jusqu'à 15 relevés par heure, ce qui aurait dupliqué les courses). On compare par **gamme** pour ne pas confondre "
        "l'effet de la pluie avec un changement de mix de véhicules.",
    )

with t4:
    st.subheader(f"Qui est le moins cher au mile ? Gamme « {gamme_choisie} »")
    d = duel[duel["gamme"] == gamme_choisie].sort_values("quartier_depart")
    fig = px.bar(d, x="quartier_depart", y="prix_mile_median", color="compagnie_gagnante",
                 color_discrete_map={"Lyft": "#e7298a", "Uber": "#252525"}, hover_data=["offre_gagnante", "prix_median"],
                 labels={"quartier_depart": "quartier de départ", "prix_mile_median": "prix médian au mile ($)",
                         "compagnie_gagnante": "moins cher"})
    fig.update_layout(height=400, xaxis_tickangle=-35)
    st.plotly_chart(fig, width="stretch")
    victoires = duel[duel["nb_offres_en_lice"] > 1]["compagnie_gagnante"].value_counts()
    st.markdown("**Victoires toutes gammes confondues :** " + " · ".join(f"{k} {v}" for k, v in victoires.items()))
    explication(
        """
WITH offre_quartier AS (
    SELECT source, gamme, cab_type, name, PERCENTILE_APPROX(prix_par_mile, 0.5) AS prix_mile_median
    FROM v_silver GROUP BY source, gamme, cab_type, name)
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY source, gamme ORDER BY prix_mile_median) AS rn
    FROM offre_quartier) WHERE rn = 1
        """,
        "`ROW_NUMBER() OVER (PARTITION BY quartier, gamme ...)` garde l'offre la moins chère de chaque groupe. "
        "On compare le **prix médian au mile**, qui neutralise la longueur des trajets.",
    )

with t5:
    st.subheader("Segmentation K-Means des trajets")
    fig = px.scatter(segments, x="distance_mediane", y="prix_mile_median", color="segment", size="nb_courses",
                     hover_name="trajet", hover_data=["prix_median", "pct_majorees"],
                     color_discrete_sequence=["#e6550d", "#3182bd", "#31a354", "#756bb1", "#636363", "#e7298a"],
                     labels={"distance_mediane": "distance médiane (miles)", "prix_mile_median": "prix médian au mile ($)"})
    fig.update_layout(height=470, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(segments.groupby("segment")[["nb_courses", "distance_mediane", "prix_median", "prix_mile_median", "pct_majorees"]]
                 .median().round(2), width="stretch")
    explication(
        """
SELECT trajet, COUNT(*) AS nb_courses, PERCENTILE_APPROX(distance, 0.5) AS distance_mediane,
       PERCENTILE_APPROX(price, 0.5) AS prix_median, PERCENTILE_APPROX(prix_par_mile, 0.5) AS prix_mile_median,
       100 * AVG(is_surge) AS pct_majorees
FROM v_silver GROUP BY trajet   -- puis log1p + StandardScaler + KMeans (k choisi par silhouette)
        """,
        "Les 5 indicateurs sont passés en `log1p` (asymétrie) puis **centrés-réduits**, sinon le volume de courses "
        "écraserait les distances dans le calcul euclidien du K-Means. Les segments sont nommés à partir des **médianes** "
        "de chaque groupe.",
    )

    st.subheader("Tests d'hypothèses (échantillon de 10 %)")
    st.dataframe(tests[["test", "hypothese", "p_value", "decision_5pct", "mesure_effet", "taille_effet", "detail"]]
                 .round({"p_value": 6, "taille_effet": 4}), width="stretch", hide_index=True)
    explication(
        "-- échantillon : SELECT cab_type, price, distance, prix_par_mile, is_surge, is_pluie, tranche_horaire\n"
        "--               FROM v_silver TABLESAMPLE (10 PERCENT)",
        "Tests **non paramétriques** (Mann-Whitney, Kruskal-Wallis, Spearman) car les prix sont asymétriques, "
        "et **χ²** pour deux variables qualitatives. Avec des dizaines de milliers de courses, presque tout devient "
        "significatif : la **taille d'effet** dit si la différence compte vraiment.",
    )
