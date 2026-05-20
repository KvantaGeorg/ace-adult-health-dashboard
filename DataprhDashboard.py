import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

st.set_page_config(
    page_title="ACE and Adult Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #f7f5f0; color: #1a1a2e; }

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 6px;
    gap: 4px;
    border: 1px solid #e8e4dc;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: #8a8a9a;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1a1a2e !important;
    color: #ffffff !important;
}
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e8e4dc;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.9rem !important;
    color: #1a1a2e !important;
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9a9aaa !important;
}
.section-header {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #9a9aaa;
    font-weight: 600;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e8e4dc;
}
.dash-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.dash-sub {
    font-size: 0.85rem;
    color: #9a9aaa;
    margin-top: 0.3rem;
}
.insight-box {
    background: #1a1a2e;
    color: #f7f5f0;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    font-size: 0.82rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.insight-box b { color: #a78bfa; }
.coming-soon {
    background: #ffffff;
    border: 2px dashed #e8e4dc;
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
    color: #c0c0cc;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

BG     = "#ffffff"
BG2    = "#f7f5f0"
GRID   = "#e8e4dc"
TEXT   = "#1a1a2e"
DARK   = "#1a1a2e"
GREEN  = "#34d399"
RED    = "#f87171"
YELLOW = "#fbbf24"
ORANGE = "#fb923c"
PURPLE = "#a78bfa"
BLUE   = "#4C9BE8"
FONT   = dict(family="Sora, sans-serif", color=TEXT, size=12)

def gl(**kwargs):
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,

        font=dict(
            family="Sora, sans-serif",
            color=DARK,
            size=12
        ),

        margin=dict(l=16, r=16, t=40, b=16),

        xaxis=dict(
            gridcolor=GRID,
            linecolor=GRID,
            tickfont=dict(color=DARK),
            title_font=dict(color=DARK),
        ),

        yaxis=dict(
            gridcolor=GRID,
            linecolor=GRID,
            tickfont=dict(color=DARK),
            title_font=dict(color=DARK),
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=DARK),
            title_font=dict(color=DARK),
        ),
    )

    base.update(kwargs)
    return base


def ax(**kwargs):
    return dict(
        gridcolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=DARK, size=12),
        title_font=dict(color=DARK, size=12),
        **kwargs
    )
def force_dark_plot_text(fig):
    fig.update_layout(
        font=dict(family="Sora, sans-serif", color=DARK, size=12),
        title_font=dict(color=DARK),
        legend=dict(
            font=dict(color=DARK),
            title_font=dict(color=DARK)
        )
    )

    fig.update_xaxes(
        tickfont=dict(color=DARK),
        title_font=dict(color=DARK)
    )

    fig.update_yaxes(
        tickfont=dict(color=DARK),
        title_font=dict(color=DARK)
    )

    for ann in fig.layout.annotations:
        ann.font = dict(color=DARK, size=12)

    return fig


@st.cache_data
def load_data():
    df = pd.read_sas(
        "/Users/kvantageorg/Desktop/Data_prj_db/LLCP2024.XPT ",
        format="xport", encoding="latin-1"
    )

    ace_cols = ["ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
                "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX"]

    all_cols = list(dict.fromkeys(
        ace_cols +
        ["ACEADSAF", "ACEADNED", "_STATE", "_SEX", "_RACEGR3", "_EDUCAG", "_URBSTAT",
         "PHYSHLTH", "MENTHLTH", "POORHLTH", "GENHLTH", "_BMI5",
         "CVDINFR4", "CVDSTRK3", "DIABETE4", "CHCCOPD3", "HAVARTH4", "CHCKDNY2", "ASTHMA3", "ADDEPEV3",
         "DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE",
         "PERSDOC3", "PRIMINS2", "SDHFOOD1", "SDHBILLS", "SDHTRNSP", "MEDCOST1",
         "_INCOMG1", "_AGEG5YR", "EMPLOY1",
         "AVEDRNK4", "ALCDAY4", "MAXDRNKS", "LCSNUMCG", "MARIJAN1",
         "_SMOKER3", "_RFBING6", "_RFDRHV9", "ECIGNOW3", "SMOKE100",
         "LSATISFY", "EMTSUPRT", "SDLONELY"]
    ))
    df = df[all_cols].copy()

    def nan(df, cols, vals):
        df[cols] = df[cols].replace({v: np.nan for v in vals})
        return df

    nan(df, ["ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
             "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX",
             "ACEADSAF", "ACEADNED", "GENHLTH", "CVDINFR4", "CVDSTRK3", "DIABETE4",
             "CHCCOPD3", "HAVARTH4", "CHCKDNY2", "ASTHMA3", "ADDEPEV3",
             "DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE", "SDHBILLS", "SDHTRNSP",
             "MEDCOST1", "PERSDOC3", "SDHFOOD1", "ECIGNOW3", "SMOKE100",
             "LSATISFY", "EMTSUPRT", "SDLONELY"], [7.0, 9.0])
    nan(df, ["PHYSHLTH", "MENTHLTH", "POORHLTH", "AVEDRNK4", "MAXDRNKS", "MARIJAN1"], [77.0, 99.0])
    nan(df, ["ALCDAY4", "LCSNUMCG"], [777.0, 999.0])
    nan(df, ["PRIMINS2"], [77.0, 99.0])
    nan(df, ["MAXDRNKS"], [88.0])
    nan(df, ["_SMOKER3", "_RFBING6", "_RFDRHV9"], [9.0])
    nan(df, ["_INCOMG1", "_RACEGR3", "_EDUCAG"], [9.0])
    nan(df, ["_AGEG5YR"], [14.0])
    nan(df, ["EMPLOY1"], [9.0])

    df = df.dropna(subset=ace_cols).copy()

    binary_std = ["CVDINFR4", "CVDSTRK3", "CHCCOPD3", "HAVARTH4", "CHCKDNY2", "ASTHMA3", "ADDEPEV3",
                  "DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE", "SDHBILLS", "SDHTRNSP", "MEDCOST1", "SMOKE100"]
    df[binary_std] = df[binary_std].replace({1.0: 1, 2.0: 0})
    df[["_RFBING6", "_RFDRHV9"]] = df[["_RFBING6", "_RFDRHV9"]].replace({1.0: 0, 2.0: 1})
    df[["PHYSHLTH", "MENTHLTH", "POORHLTH"]] = df[["PHYSHLTH", "MENTHLTH", "POORHLTH"]].replace({88.0: 0})
    df["AVEDRNK4"] = df["AVEDRNK4"].replace({88.0: 0})
    df["MARIJAN1"] = df["MARIJAN1"].replace({88.0: 0})
    df["_SEX"]     = df["_SEX"].replace({1.0: 0, 2.0: 1})
    df["_URBSTAT"] = df["_URBSTAT"].replace({1.0: 1, 2.0: 0})
    df["_BMI5"]    = df["_BMI5"] / 100

    ace_binary = ["ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC"]
    df[ace_binary] = df[ace_binary].replace({1.0: 1, 2.0: 0, 8.0: 0})
    ace_freq = ["ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX"]
    df[ace_freq] = df[ace_freq].replace({1.0: 0, 2.0: 1, 3.0: 1})

    df["ACE_SCORE"] = df[ace_cols].sum(axis=1)
    df["ACE_GROUP"] = pd.cut(df["ACE_SCORE"], bins=[-1, 0, 2, 4, 11],
                             labels=["0", "1-2", "3-4", "5+"])

    ace_weights = {
        "ACEHVSEX": 2, "ACETOUCH": 2, "ACETTHEM": 2,
        "ACEPUNCH": 2, "ACEHURT1": 2,
        "ACESWEAR": 1, "ACEDEPRS": 1, "ACEDRINK": 1,
        "ACEDRUGS": 1, "ACEPRISN": 1, "ACEDIVRC": 1,
    }
    df["ACE_SCORE_W"] = sum(df[col] * w for col, w in ace_weights.items())
    ACE_W_ORDER = ["0", "1", "2-3", "4+"]

    df["ACE_GROUP_W"] = pd.cut(
    df["ACE_SCORE_W"],
    bins=[-1, 0, 1, 3, 16],
    labels=ACE_W_ORDER,
    include_lowest=True
)

    df["ACE_GROUP_W"] = pd.Categorical(
    df["ACE_GROUP_W"].astype(str),
    categories=ACE_W_ORDER,
    ordered=True
)

    df["MENTHLTH_BIN"] = (df["MENTHLTH"] >= 14).astype(float)
    df["EMTSUPRT_HIGH"] = df["EMTSUPRT"].apply(
        lambda x: 1 if x in [1.0, 2.0] else (0 if x in [3.0, 4.0, 5.0] else np.nan))
    df["LONELY_BIN"] = df["SDLONELY"].apply(
        lambda x: 1 if x in [1.0, 2.0] else (0 if x in [3.0, 4.0, 5.0] else np.nan))

    df["DIABETE4"] = df["DIABETE4"].replace({1: 1, 2: 0, 3: 0, 4: 0})

    disab_cols_list = ["DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE"]
    df["DISABILITY_SCORE"] = df[disab_cols_list].sum(axis=1, skipna=False)

    df["BMI"] = df["_BMI5"].copy()
    df.loc[df["BMI"] < 10, "BMI"] = np.nan
    df.loc[df["BMI"] > 80, "BMI"] = np.nan

    return df


with st.spinner("Loading BRFSS 2024 data..."):
    df = load_data()

st.markdown("""
<div style="padding: 1.5rem 0 1rem">
  <div class="dash-title">ACE and Adult Health</div>
  <div class="dash-sub">Adverse Childhood Experiences, BRFSS 2024, N = 457,670</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; justify-content:flex-end; margin-bottom:1rem;">
</div>
""", unsafe_allow_html=True)

if st.button("End of Semester", type="primary"):
    st.balloons()


tabs = st.tabs([
    "Overview",
    "Mental Health",
    "Physical Health",
    "Substance Use",
    "Healthcare Access",
    "Protective Factors",
    "Depression Risk",
    "AI Advisor",
])


# TAB 0 - ACE Prevalence (Angle 1)

with tabs[0]:
    st.markdown("""
    <div style="margin-bottom:1rem">
        <div class="dash-title" style="font-size:1.3rem">ACE Prevalence and Profile</div>
        <div class="dash-sub">
            ACE - Adverse Childhood Experiences
            Weighted ACE Score - a count (0-16) of how many distinct types of adverse childhood experiences a person experienced, with less severe ACEs contributing +1 and more severe ACEs contributing a +2 to the score
            ACEs are binary - each type is counted once regardless of frequency or duration
            Sample is filtered to only respondents who gave information on their ACEs (n=35,969)
        </div>
    </div>
    """, unsafe_allow_html=True)

    ace_readable = {
        "ACEDEPRS": "Parent depression",
        "ACEDRINK": "Parent alcohol problem",
        "ACEDRUGS": "Parent drug problem",
        "ACEPRISN": "Household member incarcerated",
        "ACEDIVRC": "Parents divorce",
        "ACEPUNCH": "Witness domestic violence",
        "ACEHURT1": "Physical abuse",
        "ACESWEAR": "Emotional abuse",
        "ACETOUCH": "Sexual abuse (unwanted touch)",
        "ACETTHEM": "Sexual abuse (forced touching)",
        "ACEHVSEX": "Sexual intercourse (forced)",
    }
    age_labels = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
        6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
        11: "70-74", 12: "75-79", 13: "80+"
    }

    ace_cols_list = ["ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
                     "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX"]

    prev = df[ace_cols_list].mean().mul(100).rename(ace_readable).sort_values()
    age_group = (
        df.assign(label=df["_AGEG5YR"].map(age_labels))
        .groupby("label", observed=True)["ACE_SCORE"]
        .mean()
        .reindex([age_labels[k] for k in sorted(age_labels)])
    )

    st.markdown('<div class="section-header">Q1: How prevalent are ACEs in this sample, and do reported ACE scores differ accross age groups?</div>', unsafe_allow_html=True)
    col_a1, col_a2 = st.columns([2, 1])

    with col_a1:
        fig_prev = go.Figure(go.Bar(
            x=prev.values,
            y=prev.index,
            orientation="h",
            marker_color=PURPLE,
            opacity=0.85,
            text=prev.values.round(1).astype(str) + "%",
            textposition="outside",
            textfont=dict(color=DARK, size=11),
        ))
        fig_prev.update_layout(**gl(
            title=dict(text="Prevalence of each ACE type (%)", font=dict(color=DARK, size=13)),
            xaxis=ax(ticksuffix="%", range=[0, prev.max() * 1.25]),
            height=420,
            margin=dict(l=220, r=60, t=40, b=16),
        ))
        force_dark_plot_text(fig_prev)
        st.plotly_chart(fig_prev, use_container_width=True)

    with col_a2:
        fig_age = go.Figure(go.Scatter(
            x=age_group.index,
            y=age_group.values,
            mode="lines+markers",
            line=dict(color=ORANGE, width=2.5),
            marker=dict(size=7, color=ORANGE),
        ))
        fig_age.update_layout(**gl(
            title=dict(text="Mean ACE score by age group", font=dict(color=DARK, size=13)),
            xaxis=ax(tickangle=45),
            yaxis=ax(title="Mean ACE score"),
            height=420,
        ))
        force_dark_plot_text(fig_age)
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown('<div class="section-header">Q2: What is the distribution of ACE score in the sample?</div>', unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        counts = df["ACE_SCORE"].value_counts().sort_index()
        pcts   = counts / counts.sum() * 100
        med    = df["ACE_SCORE"].median()
        mn     = df["ACE_SCORE"].mean()
        fig_dist1 = go.Figure(go.Bar(
            x=pcts.index, y=pcts.values,
            marker_color=BLUE, opacity=0.85,
            text=pcts.round(1).astype(str) + "%",
            textposition="outside",
            textfont=dict(color=DARK, size=10),
        ))
        fig_dist1.add_vline(x=med, line_dash="dash", line_color=DARK, line_width=1.5,
                            annotation_text=f"Median = {med:.0f}", annotation_font_color=DARK)
        fig_dist1.update_layout(**gl(
            title=dict(text="Unweighted ACE score (0-11)", font=dict(color=DARK, size=13)),
            xaxis=ax(title="Score", dtick=1),
            yaxis=ax(ticksuffix="%", range=[0, pcts.max() * 1.25]),
            height=340,
        ))
        force_dark_plot_text(fig_dist1)
        st.plotly_chart(fig_dist1, use_container_width=True)

    with col_b2:
        counts_w = df["ACE_SCORE_W"].value_counts().sort_index()
        pcts_w   = counts_w / counts_w.sum() * 100
        med_w    = df["ACE_SCORE_W"].median()
        fig_dist2 = go.Figure(go.Bar(
            x=pcts_w.index, y=pcts_w.values,
            marker_color=ORANGE, opacity=0.85,
            text=pcts_w.round(1).astype(str) + "%",
            textposition="outside",
            textfont=dict(color=DARK, size=10),
        ))
        fig_dist2.add_vline(x=med_w, line_dash="dash", line_color=DARK, line_width=1.5,
                            annotation_text=f"Median = {med_w:.0f}", annotation_font_color=DARK)
        fig_dist2.update_layout(**gl(
            title=dict(text="Weighted ACE score (0-16)", font=dict(color=DARK, size=13)),
            xaxis=ax(title="Score", dtick=1),
            yaxis=ax(ticksuffix="%", range=[0, pcts_w.max() * 1.25]),
            height=340,
        ))
        force_dark_plot_text(fig_dist2)
        st.plotly_chart(fig_dist2, use_container_width=True)

    st.markdown(
        f'<div style="text-align:center;color:#c0c0cc;font-size:0.7rem;'
        f'margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e8e4dc">'
        f'BRFSS 2024, Angle 1: ACE Prevalence, n = {len(df):,} respondents with complete ACE data'
        f'</div>', unsafe_allow_html=True)
    
st.markdown('<div class="section-header">Limitations of the Project</div>', unsafe_allow_html=True)
col_b1, col_b2 = st.columns(2)

with col_b1:
    st.markdown("""
    - Possible omitted ACEs  
    - The weighing of ACEs was subjective and based on our own perceived severity  
    - The survey is self-reported, ACEs most likely underreported  
    - Correlation between ACEs and bad health doesn't mean ACEs caused it, both may stem from the same underlying cause  
    """)

with col_b2:
    st.markdown("""
    - Hard to generalize as the sample represents four US states  
    - Only binary outcome for ACEs, ignores duration, severity, age at exposure, etc.  
    """)

st.markdown('<div class="section-header">Further Possible Analysis</div>', unsafe_allow_html=True)
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("""
    - Subgroup analysis by sex or age group  
    """)

with col_c2:
    st.markdown("""
    - Sensitivity analysis on ACE groupings and weights  
    """)

st.markdown('<div class="section-header">Additional Data That Would Have Been Nice to Have</div>', unsafe_allow_html=True)
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown("""
    - Age at first ACE exposure  
    - Continuous variables instead of binned into categories (for ex., income)  
    """)

with col_d2:
    st.markdown("""
    - Data on all states  
    - Data on mental health diagnoses
    """)


# TAB 1 - Mental Health (Angle 2)
with tabs[1]:
    st.markdown("""
    <div style="margin-bottom:1rem">
        <div class="dash-title" style="font-size:1.3rem">Mental Health</div>
        <div class="dash-sub">
            Mental Health examines how childhood adversity affects adult mental health, depression risk, loneliness, and emotional support outcomes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    a2 = df[["ACE_SCORE", "ACE_GROUP", "MENTHLTH", "MENTHLTH_BIN",
             "ADDEPEV3", "EMTSUPRT", "EMTSUPRT_HIGH", "SDLONELY", "LONELY_BIN",
             "_SEX", "_AGEG5YR", "_RACEGR3", "_INCOMG1", "_EDUCAG"]].copy()

    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        sex_map = {0.0: "Male", 1.0: "Female"}
        sex_opts = ["All"] + [sex_map[v] for v in sorted(a2["_SEX"].dropna().unique())]
        sel_sex = st.selectbox("Sex", sex_opts)

    with fc2:
        age_map = {
            1.0: "18-24", 2.0: "25-29", 3.0: "30-34", 4.0: "35-39",
            5.0: "40-44", 6.0: "45-49", 7.0: "50-54", 8.0: "55-59",
            9.0: "60-64", 10.0: "65-69", 11.0: "70-74", 12.0: "75-79", 13.0: "80+"
        }
        age_opts = ["All"] + [age_map[v] for v in sorted(a2["_AGEG5YR"].dropna().unique()) if v in age_map]
        sel_age = st.selectbox("Age group", age_opts)

    with fc3:
        income_map = {
            1.0: "<$15k", 2.0: "$15-25k", 3.0: "$25-35k",
            4.0: "$35-50k", 5.0: "$50-100k", 6.0: "$100-200k", 7.0: "$200k+"
        }
        inc_opts = ["All"] + [income_map[v] for v in sorted(a2["_INCOMG1"].dropna().unique()) if v in income_map]
        sel_inc = st.selectbox("Income", inc_opts)

    fdf = a2.copy()
    if sel_sex != "All":
        sex_rev = {"Male": 0.0, "Female": 1.0}
        fdf = fdf[fdf["_SEX"] == sex_rev[sel_sex]]
    if sel_age != "All":
        age_rev = {v: k for k, v in age_map.items()}
        fdf = fdf[fdf["_AGEG5YR"] == age_rev[sel_age]]
    if sel_inc != "All":
        inc_rev = {v: k for k, v in income_map.items()}
        fdf = fdf[fdf["_INCOMG1"] == inc_rev[sel_inc]]

    st.markdown(f'<div style="font-size:0.75rem;color:#9a9aaa;margin-bottom:1rem">Showing <b style="color:#1a1a2e">{len(fdf):,}</b> respondents</div>',
                unsafe_allow_html=True)

    lr_df = fdf[["ACE_SCORE", "ADDEPEV3"]].dropna()
    X_sm  = sm.add_constant(lr_df["ACE_SCORE"])
    logit = sm.Logit(lr_df["ADDEPEV3"], X_sm).fit(disp=0)
    OR    = np.exp(logit.params["ACE_SCORE"])
    CI_lo = np.exp(logit.conf_int().loc["ACE_SCORE", 0])
    CI_hi = np.exp(logit.conf_int().loc["ACE_SCORE", 1])

    dep_rate_0 = fdf[fdf["ACE_SCORE"] == 0]["ADDEPEV3"].mean() * 100
    dep_rate_5 = fdf[fdf["ACE_GROUP"] == "5+"]["ADDEPEV3"].mean() * 100
    lonely_dep = fdf[fdf["LONELY_BIN"] == 1]["ADDEPEV3"].mean() * 100
    mean_mh_5  = fdf[fdf["ACE_GROUP"] == "5+"]["MENTHLTH"].mean()

    st.markdown('<div class="section-header">Key findings</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Depression, 0 ACEs", f"{dep_rate_0:.1f}%")
    k2.metric("Depression, 5+ ACEs", f"{dep_rate_5:.1f}%")
    k3.metric("Odds Ratio per ACE", f"{OR:.3f}")
    k4.metric("Depression if lonely", f"{lonely_dep:.1f}%" if not np.isnan(lonely_dep) else "N/A")
    k5.metric("Mean bad days (5+)", f"{mean_mh_5:.1f} days")

    st.markdown('<div class="section-header">Insights</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f"""<div class="insight-box">
        Each additional ACE increases the odds of depression by <b>{(OR-1)*100:.0f}%</b>
        (OR = {OR:.3f}, 95% CI [{CI_lo:.3f}-{CI_hi:.3f}], p &lt; 0.0001).
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""<div class="insight-box">
        People with <b>5+ ACEs</b> are <b>{dep_rate_5/dep_rate_0:.1f}x</b> more likely
        to have a depression diagnosis than those with no ACEs
        ({dep_rate_5:.1f}% vs {dep_rate_0:.1f}%).
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""<div class="insight-box">
        <b>Loneliness</b> is a stronger predictor than ACE count alone.
        Lonely individuals show <b>{lonely_dep:.1f}%</b> depression rate
        vs 15.0% among non-lonely respondents.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Depression and mental health by ACE group</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    colors4 = [GREEN, YELLOW, ORANGE, RED]

    with col1:
        dep_grp = (
            fdf.dropna(subset=["ACE_GROUP", "ADDEPEV3"])
            .groupby("ACE_GROUP", observed=True)
            .agg(total=("ADDEPEV3", "count"), depressed=("ADDEPEV3", "sum"))
            .assign(dep_rate=lambda x: x["depressed"] / x["total"] * 100)
            .reset_index()
        )
        fig1 = go.Figure(go.Bar(
            x=dep_grp["ACE_GROUP"], y=dep_grp["dep_rate"],
            marker_color=colors4,
            text=dep_grp["dep_rate"].map("{:.1f}%".format),
            textposition="outside",
            textfont=dict(color=DARK, size=12, family="JetBrains Mono"),
        ))
        fig1.update_layout(**gl(
            title=dict(text="Depression rate by ACE score group", font=dict(color=DARK, size=13)),
            yaxis=ax(ticksuffix="%", range=[0, 65]),
            height=340,
        ))
        force_dark_plot_text(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        mh_grp = (
            fdf.dropna(subset=["ACE_GROUP", "MENTHLTH"])
            .groupby("ACE_GROUP", observed=True)["MENTHLTH"]
            .agg(["mean", "sem"]).reset_index()
        )
        fig2 = go.Figure(go.Bar(
            x=mh_grp["ACE_GROUP"], y=mh_grp["mean"],
            error_y=dict(type="data", array=mh_grp["sem"] * 1.96, color=TEXT),
            marker_color=colors4,
            text=mh_grp["mean"].map("{:.1f}".format),
            textposition="outside",
            textfont=dict(color=DARK, size=12, family="JetBrains Mono"),
        ))
        fig2.update_layout(**gl(
            title=dict(text="Mean poor mental health days (last 30 days)", font=dict(color=DARK, size=13)),
            yaxis=ax(range=[0, 15]),
            height=340,
        ))
        force_dark_plot_text(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Moderation and loneliness effects</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        mod_df = (
            fdf.dropna(subset=["ACE_GROUP", "ADDEPEV3", "EMTSUPRT_HIGH"])
            .groupby(["ACE_GROUP", "EMTSUPRT_HIGH"], observed=True)
            .agg(total=("ADDEPEV3", "count"), depressed=("ADDEPEV3", "sum"))
            .assign(dep_rate=lambda x: x["depressed"] / x["total"] * 100)
            .reset_index()
        )
        fig3 = go.Figure()
        for support, color, label in [(1.0, GREEN, "High support"), (0.0, RED, "Low support")]:
            sub = mod_df[mod_df["EMTSUPRT_HIGH"] == support]
            fig3.add_trace(go.Bar(
                name=label, x=sub["ACE_GROUP"], y=sub["dep_rate"],
                marker_color=color, opacity=0.85,
                text=sub["dep_rate"].map("{:.1f}%".format),
                textposition="outside",
                textfont=dict(color=DARK, size=11),
            ))
        fig3.update_layout(**gl(
            title=dict(text="Depression by ACE group, moderated by emotional support",
                       font=dict(color=DARK, size=13)),
            yaxis=ax(ticksuffix="%", range=[0, 75]),
            barmode="group", height=340,
        ))
        force_dark_plot_text(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        lonely_df = (
            fdf.dropna(subset=["LONELY_BIN", "ADDEPEV3", "ACE_GROUP"])
            .groupby(["ACE_GROUP", "LONELY_BIN"], observed=True)
            .agg(total=("ADDEPEV3", "count"), depressed=("ADDEPEV3", "sum"))
            .assign(dep_rate=lambda x: x["depressed"] / x["total"] * 100)
            .reset_index()
        )
        fig4 = go.Figure()
        for lonely, color, label in [(0.0, GREEN, "Not lonely"), (1.0, RED, "Lonely")]:
            sub = lonely_df[lonely_df["LONELY_BIN"] == lonely]
            fig4.add_trace(go.Bar(
                name=label, x=sub["ACE_GROUP"], y=sub["dep_rate"],
                marker_color=color, opacity=0.85,
                text=sub["dep_rate"].map("{:.1f}%".format),
                textposition="outside",
                textfont=dict(color=DARK, size=11),
            ))
        fig4.update_layout(**gl(
            title=dict(text="Depression by ACE group, lonely vs not lonely",
                       font=dict(color=DARK, size=13)),
            yaxis=ax(ticksuffix="%", range=[0, 75]),
            barmode="group", height=340,
        ))
        force_dark_plot_text(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">Logistic regression, odds ratio</div>', unsafe_allow_html=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=[OR], y=["ACE Score"],
        mode="markers",
        marker=dict(color=PURPLE, size=14, symbol="diamond"),
        error_x=dict(type="data", symmetric=False,
                     array=[CI_hi - OR], arrayminus=[OR - CI_lo],
                     color=PURPLE, thickness=2, width=8),
        hovertemplate=f"OR = {OR:.3f}<br>95% CI [{CI_lo:.3f}-{CI_hi:.3f}]<extra></extra>",
    ))
    fig5.add_vline(x=1.0, line_dash="dash", line_color=TEXT, line_width=1.5)
    fig5.add_annotation(
        x=OR, y=1.15, text=f"OR = {OR:.3f} [{CI_lo:.3f}-{CI_hi:.3f}]",
        showarrow=False, font=dict(color=PURPLE, size=12, family="JetBrains Mono"),
    )
    fig5.update_layout(**gl(
        title=dict(text="Odds Ratio: ACE Score to Depression (per 1-point increase)",
                   font=dict(color=DARK, size=13)),
        xaxis=ax(title="Odds Ratio", range=[1.1, 1.6]),
        height=220,
        margin=dict(l=120, r=40, t=40, b=40),
        yaxis=ax(showgrid=False),
    ))
    force_dark_plot_text(fig5)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        f'<div style="text-align:center;color:#c0c0cc;font-size:0.7rem;'
        f'margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e8e4dc">'
        f'BRFSS 2024, Angle 2: Mental Health, n = {len(fdf):,} filtered respondents'
        f'</div>', unsafe_allow_html=True)


# TAB  2 - Physical Health (Angle 3)

with tabs[2]:
    st.markdown("""
    <div style="margin-bottom:1rem">
        <div class="dash-title" style="font-size:1.3rem">Physical Health Outcomes</div>
        <div class="dash-sub">
            Do childhood adversity and current socioeconomic hardship predict worse physical health in adulthood?
        </div>
    </div>
    """, unsafe_allow_html=True)

    ACE_W_ORDER = ["0", "1", "2-3", "4+"]

    a3 = df[[
        "ACE_SCORE_W", "ACE_GROUP_W",
        "BMI", "PHYSHLTH",
        "CVDINFR4", "CVDSTRK3", "DIABETE4", "CHCCOPD3",
        "HAVARTH4", "CHCKDNY2", "ASTHMA3",
        "DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE",
        "DISABILITY_SCORE",
        "_SEX", "_AGEG5YR", "_RACEGR3", "_INCOMG1", "_EDUCAG", "_URBSTAT"
    ]].copy()

    # Rebuild ACE groups directly from ACE_SCORE_W to avoid cached or overwritten category bugs
    a3["ACE_GROUP_W"] = pd.cut(
        a3["ACE_SCORE_W"],
        bins=[-1, 0, 1, 3, 16],
        labels=ACE_W_ORDER,
        include_lowest=True
    )

    a3["ACE_GROUP_W"] = pd.Categorical(
        a3["ACE_GROUP_W"].astype(str),
        categories=ACE_W_ORDER,
        ordered=True
    )

    chronic_vars = {
        "CVDINFR4": "Heart Attack",
        "CVDSTRK3": "Stroke",
        "DIABETE4": "Diabetes",
        "CHCCOPD3": "COPD",
        "HAVARTH4": "Arthritis",
        "CHCKDNY2": "Kidney Disease",
        "ASTHMA3": "Asthma",
    }

    st.markdown('<div class="section-header">Key findings</div>', unsafe_allow_html=True)

    bmi_0 = a3[a3["ACE_GROUP_W"] == "0"]["BMI"].mean()
    bmi_4 = a3[a3["ACE_GROUP_W"] == "4+"]["BMI"].mean()

    phys_0 = a3[a3["ACE_GROUP_W"] == "0"]["PHYSHLTH"].mean()
    phys_4 = a3[a3["ACE_GROUP_W"] == "4+"]["PHYSHLTH"].mean()

    dis_0 = a3[a3["ACE_GROUP_W"] == "0"]["DISABILITY_SCORE"].mean()
    dis_4 = a3[a3["ACE_GROUP_W"] == "4+"]["DISABILITY_SCORE"].mean()

    asthma_4 = a3[a3["ACE_GROUP_W"] == "4+"]["ASTHMA3"].mean() * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("Bad days, 0 ACEs", f"{phys_0:.1f}")
    k2.metric("Bad days, 4+ ACEs", f"{phys_4:.1f}")
    k3.metric("Asthma, 4+ ACEs", f"{asthma_4:.1f}%")

    st.markdown('<div class="section-header">Insights</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(f"""
        <div class="insight-box">
        Respondents with <b>4+ weighted ACEs</b> have higher average BMI:
        <b>{bmi_4:.1f}</b> vs <b>{bmi_0:.1f}</b>.
        </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown(f"""
        <div class="insight-box">
        Poor physical health days rise from <b>{phys_0:.1f}</b> to
        <b>{phys_4:.1f}</b> days per month.
        </div>
        """, unsafe_allow_html=True)

    with i3:
        st.markdown(f"""
        <div class="insight-box">
        Functional disability burden is higher among high-ACE respondents:
        <b>{dis_4:.2f}</b> vs <b>{dis_0:.2f}</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-header">Poor physical health days by ACE group</div>',
        unsafe_allow_html=True
    )


    grp_phys = (
        a3.dropna(subset=["ACE_GROUP_W", "PHYSHLTH"])
        .groupby("ACE_GROUP_W", observed=False)["PHYSHLTH"]
        .mean()
        .reindex(ACE_W_ORDER)
        .reset_index()
    )

    fig_phys = go.Figure(go.Bar(
            x=grp_phys["ACE_GROUP_W"].astype(str),
            y=grp_phys["PHYSHLTH"].round(1),
            marker_color=ORANGE,
            opacity=0.85,
            text=grp_phys["PHYSHLTH"].round(1),
            textposition="outside",
            textfont=dict(color=DARK, size=11, family="JetBrains Mono"),
        ))

    fig_phys.update_layout(**gl(
            title=dict(text="Mean Poor Physical Health Days by Weighted ACE Score Group", font=dict(color=DARK, size=13)),
            xaxis=ax(title="Weighted ACE Score Group (0-16)", type="category"),
            yaxis=ax(title="Mean Days", range=[0, grp_phys["PHYSHLTH"].max() * 1.2]),
            height=380,
        ))
    force_dark_plot_text(fig_phys)
    st.plotly_chart(fig_phys, use_container_width=True)

    st.markdown(
        '<div class="section-header">Chronic disease prevalence by ACE group</div>',
        unsafe_allow_html=True
    )

    prev_data = {}

    for var, label in chronic_vars.items():
        prev = (
            a3.dropna(subset=["ACE_GROUP_W", var])
            .groupby("ACE_GROUP_W", observed=False)[var]
            .mean()
            .mul(100)
            .reindex(ACE_W_ORDER)
            .round(1)
        )
        prev_data[label] = prev

    prev_df = pd.DataFrame(prev_data).reindex(ACE_W_ORDER)
    line_colors = [BLUE, RED, GREEN, ORANGE, PURPLE, "#8B6914", "#E84C8B"]

    fig_chronic = go.Figure()

    for condition, color in zip(prev_df.columns, line_colors):
        fig_chronic.add_trace(go.Scatter(
            name=condition,
            x=prev_df.index.astype(str),
            y=prev_df[condition],
            mode="lines+markers",
            line=dict(color=color, width=2.3),
            marker=dict(size=8, color=color),
            hovertemplate="%{x}: %{y:.1f}%<extra>" + condition + "</extra>",
        ))

    fig_chronic.update_layout(**gl(
        title=dict(text="Chronic Disease Prevalence by Weighted ACE Score Group", font=dict(color=DARK, size=13)),
        xaxis=ax(title="Weighted ACE Score Group (0-16)", type="category"),
        yaxis=ax(title="Prevalence (%)", range=[0, prev_df.max().max() * 1.25]),
        height=480,
        legend=dict(title="Condition", bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    ))
    force_dark_plot_text(fig_chronic)
    st.plotly_chart(fig_chronic, use_container_width=True)

    st.markdown(
        '<div class="section-header">ACE score difference by chronic disease diagnosis</div>',
        unsafe_allow_html=True
    )

    means_list = []

    for var, label in chronic_vars.items():
        sub = a3[[var, "ACE_SCORE_W"]].dropna()
        means = sub.groupby(var)["ACE_SCORE_W"].mean()

        if 0 in means.index and 1 in means.index:
            means_list.append({
                "Condition": label,
                "Not diagnosed": means[0],
                "Diagnosed": means[1],
                "Difference": means[1] - means[0],
            })

    means_df = pd.DataFrame(means_list).sort_values("Difference", ascending=True)

    fig_diff = go.Figure(go.Bar(
        y=means_df["Condition"],
        x=means_df["Difference"],
        orientation="h",
        marker_color=[RED if d > 0 else BLUE for d in means_df["Difference"]],
        opacity=0.85,
        text=means_df["Difference"].apply(lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}"),
        textposition="outside",
        textfont=dict(color=DARK, size=11, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Difference: %{x:.2f}<extra></extra>",
    ))

    x_min = means_df["Difference"].min()
    x_max = means_df["Difference"].max()
    padding = max(abs(x_min), abs(x_max)) * 0.35

    fig_diff.add_vline(
        x=0,
        line_dash="dash",
        line_color=TEXT,
        line_width=1.4
    )

    fig_diff.update_layout(**gl(
        title=dict(
            text="Difference in Mean Weighted ACE Score, Diagnosed minus Not Diagnosed",
            font=dict(color=DARK, size=13)
        ),
        xaxis=ax(title="ACE Score Difference", range=[x_min - padding, x_max + padding]),
        yaxis=ax(),
        height=420,
        margin=dict(l=150, r=80, t=45, b=35),
        showlegend=False,
    ))
    force_dark_plot_text(fig_diff)
    st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown(
        '<div class="section-header">Functional disability burden by ACE group</div>',
        unsafe_allow_html=True
    )

    disab_grp = (
        a3.dropna(subset=["ACE_GROUP_W", "DISABILITY_SCORE"])
        .groupby("ACE_GROUP_W", observed=False)["DISABILITY_SCORE"]
        .mean()
        .reindex(ACE_W_ORDER)
        .reset_index()
    )

    fig_disab = go.Figure(go.Bar(
        x=disab_grp["ACE_GROUP_W"].astype(str),
        y=disab_grp["DISABILITY_SCORE"].round(2),
        marker_color=PURPLE,
        opacity=0.85,
        text=disab_grp["DISABILITY_SCORE"].round(2),
        textposition="outside",
        textfont=dict(color=DARK, size=11, family="JetBrains Mono"),
    ))

    fig_disab.update_layout(**gl(
        title=dict(text="Functional Disability Burden by Weighted ACE Score Group", font=dict(color=DARK, size=13)),
        xaxis=ax(title="Weighted ACE Score Group (0-16)", type="category"),
        yaxis=ax(
            title="Mean Disability Burden Score (0-4)",
            range=[0, disab_grp["DISABILITY_SCORE"].max() * 1.3]
        ),
        height=380,
    ))
    force_dark_plot_text(fig_disab)
    st.plotly_chart(fig_disab, use_container_width=True)

    st.markdown(
        f'<div style="text-align:center;color:#c0c0cc;font-size:0.7rem;'
        f'margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e8e4dc">'
        f'BRFSS 2024, Angle 3: Physical Health, n = {len(a3):,} respondents with complete ACE data'
        f'</div>',
        unsafe_allow_html=True
    )


# TAB 3 - Substance Use (Angle 4)
with tabs[3]:
    st.markdown("""
    <div style="margin-bottom:1rem">
        <div class="dash-title" style="font-size:1.3rem">Substance Use</div>
        <div class="dash-sub">
            Substance Use examines how childhood adversity relates to alcohol, smoking, binge drinking, and marijuana use patterns in adulthood.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-header">Substance use overview by ACE group</div>', unsafe_allow_html=True)

    sub_df = df.copy()
    sub_df["ALCDAY4"] = sub_df["ALCDAY4"].replace({888: 0})
    sub_df["ACE_GROUP_S"] = pd.cut(
        sub_df["ACE_SCORE"],
        bins=[-0.1, 0, 2, 4, 11],
        labels=["0 (None)", "1-2 (Low)", "3-4 (Moderate)", "5+ (High)"]
    )

    summary = sub_df.groupby("ACE_GROUP_S", observed=True).agg(
        N          =("ACE_SCORE",  "count"),
        avg_drinks =("AVEDRNK4",  "mean"),
        max_drinks =("MAXDRNKS",  "mean"),
        cigs_day   =("LCSNUMCG",  "mean"),
        marijuana  =("MARIJAN1",  "mean"),
        binge_pct  =("_RFBING6",  lambda x: x.mean() * 100),
        heavy_pct  =("_RFDRHV9",  lambda x: x.mean() * 100),
        smoke_pct  =("SMOKE100",  lambda x: x.mean() * 100),
    ).round(2)
    summary.columns = ["N", "Avg drinks/day", "Max drinks/occ",
                        "Cigs/day", "Marijuana days", "Binge %", "Heavy drinking %", "Ever smoked %"]

    plot_cols = ["Avg drinks/day", "Max drinks/occ", "Cigs/day",
                 "Marijuana days", "Binge %", "Heavy drinking %", "Ever smoked %"]
    hmap_raw = summary[plot_cols].copy()
    hmap_z   = hmap_raw.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    groups   = summary.index.tolist()
    annot_text = [[str(hmap_raw.loc[g, c]) for c in plot_cols] for g in groups]

    fig_hm = go.Figure(go.Heatmap(
        z=hmap_z.values.tolist(),
        x=plot_cols,
        y=groups,
        colorscale="RdYlGn",
        reversescale=True,
        showscale=True,
        text=annot_text,
        texttemplate="%{text}",
        textfont=dict(size=12, color=DARK, family="JetBrains Mono"),
        zmin=-1.5, zmax=1.5,
    ))
    fig_hm.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
        title=dict(text="Substance use by ACE score group (z-score intensity, real values annotated)",
                   font=dict(color=DARK, size=13), y=0.98, yanchor="top"),
        height=340,
        margin=dict(l=130, r=80, t=40, b=140),
        xaxis=dict(tickangle=-40, tickfont=dict(color=TEXT, size=11),
                   gridcolor=GRID, linecolor=GRID, side="bottom"),
        yaxis=dict(tickfont=dict(color=TEXT, size=11),
                   gridcolor=GRID, linecolor=GRID, autorange="reversed"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    force_dark_plot_text(fig_hm)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Key pattern:</b> Substance use rises with ACE score across every measure.
    Marijuana days jump from 1.0 to 5.5 (5x increase). Binge drinking doubles (8.4% to 17.0%).
    Cigarettes per day stays flat around 15-16. ACEs affect <b>whether</b> someone smokes, not <b>how much</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
    '<div class="section-header">Correlation between ACE items and substance use</div>',
    unsafe_allow_html=True
)

    ace_items = [
        "ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
        "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX"
    ]

    substance_vars = ["AVEDRNK4", "_RFBING6", "SMOKE100", "MARIJAN1"]

    ace_labels = {
        "ACEDEPRS": "Parent depressed",
        "ACEDRINK": "Parent alcoholic",
        "ACEDRUGS": "Parent drug user",
        "ACEPRISN": "Parent in prison",
        "ACEDIVRC": "Parents divorced",
        "ACEPUNCH": "Witnessed violence",
        "ACEHURT1": "Physically abused",
        "ACESWEAR": "Emotionally abused",
        "ACETOUCH": "Sexually touched",
        "ACETTHEM": "Sexual attempt",
        "ACEHVSEX": "Forced sex",
    }

    subst_labels = {
        "AVEDRNK4": "Avg drinks/day",
        "_RFBING6": "Binge drinking",
        "SMOKE100": "Ever smoked",
        "MARIJAN1": "Marijuana days",
    }

    corr_data = {}

    for ace_col in ace_items:
        corr_data[ace_labels[ace_col]] = {}
        for subst_col in substance_vars:
            sub = sub_df[[ace_col, subst_col]].dropna()
            r, p = scipy_stats.pearsonr(sub[ace_col], sub[subst_col])
            corr_data[ace_labels[ace_col]][subst_labels[subst_col]] = round(r, 3)

    corr_df = pd.DataFrame(corr_data).T

    fig_corr = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu",
        reversescale=True,
        zmid=0,
        text=corr_df.round(3).astype(str).values,
        texttemplate="%{text}",
        textfont=dict(color=DARK, size=12),
        xgap=1,
        ygap=1,
        colorbar=dict(
            title=dict(text="Pearson r", font=dict(color=DARK)),
            tickfont=dict(color=DARK),
            len=0.55
        ),
        hovertemplate="%{y}<br>%{x}<br>Pearson r: %{z:.3f}<extra></extra>",
    ))

    fig_corr.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Sora, sans-serif", color=DARK, size=12),
        title=dict(
            text="Correlation Between ACE Items and Substance Use",
            font=dict(color=DARK, size=16),
            x=0.5
        ),
        height=720,
        margin=dict(l=170, r=80, t=70, b=110),
        xaxis=dict(
            tickangle=-20,
            tickfont=dict(color=DARK, size=12),
            title=dict(text="", font=dict(color=DARK)),
            gridcolor="white",
            linecolor=GRID
        ),
        yaxis=dict(
            tickfont=dict(color=DARK, size=12),
            title=dict(text="", font=dict(color=DARK)),
            autorange="reversed",
            gridcolor="white",
            linecolor=GRID
        ),
    )

    force_dark_plot_text(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

# TAB 4 - Healthcare Access (Angle 5)
with tabs[4]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

    st.markdown("""
    <div style="margin-bottom:1rem">
        <div class="dash-title" style="font-size:1.3rem">Healthcare Access</div>
        <div class="dash-sub">
            Do ACEs predict worse healthcare access: cost barriers, no personal provider, no insurance?
            Does current socioeconomic hardship compound these disadvantages? And for cost barriers specifically,
            do these effects hold independently after controlling for income and age?
        </div>
    </div>
    """, unsafe_allow_html=True)

    def force_dark_plot_text(fig):
        fig.update_layout(
            font=dict(family="Sora, sans-serif", color=DARK, size=12),
            title_font=dict(color=DARK),
            legend=dict(font=dict(color=DARK), title_font=dict(color=DARK)),
        )
        fig.update_xaxes(
            color=DARK,
            title_font=dict(color=DARK),
            tickfont=dict(color=DARK),
            linecolor=DARK,
            tickcolor=DARK,
        )
        fig.update_yaxes(
            color=DARK,
            title_font=dict(color=DARK),
            tickfont=dict(color=DARK),
            linecolor=DARK,
            tickcolor=DARK,
        )
        return fig

    hc_cols_list = [
        "ACE_SCORE_W", "MEDCOST1", "PERSDOC3", "PRIMINS2",
        "SDHFOOD1", "SDHBILLS", "SDHTRNSP",
        "_INCOMG1", "_AGEG5YR", "_STATE"
    ]

    df5 = df[["ACE_SCORE_W", "MEDCOST1", "PERSDOC3", "PRIMINS2", "_INCOMG1", "_AGEG5YR", "_STATE"]].copy()

    df5["PERSDOC3"] = df5["PERSDOC3"].map({1.0: 1, 2.0: 1, 3.0: 0})
    df5["PRIMINS2"] = df5["PRIMINS2"].apply(lambda x: 0 if x == 88.0 else (1 if pd.notna(x) else np.nan))
    df5 = df5.dropna().copy()

    df5["MEDCOST1"] = df5["MEDCOST1"].astype(int)
    df5["PERSDOC3"] = df5["PERSDOC3"].astype(int)
    df5["PRIMINS2"] = df5["PRIMINS2"].astype(int)

    df5["ACE_GROUP"] = pd.cut(
        df5["ACE_SCORE_W"],
        bins=[-1, 0, 2, 5, 16],
        labels=["0", "1-2", "3-5", "6+"]
    )

    df5_sdh = df[[
        "ACE_SCORE_W", "MEDCOST1", "PERSDOC3", "PRIMINS2",
        "SDHFOOD1", "SDHBILLS", "SDHTRNSP", "_INCOMG1", "_AGEG5YR"
    ]].copy()

    df5_sdh["PERSDOC3"] = df5_sdh["PERSDOC3"].map({1.0: 1, 2.0: 1, 3.0: 0})
    df5_sdh["PRIMINS2"] = df5_sdh["PRIMINS2"].apply(lambda x: 0 if x == 88.0 else (1 if pd.notna(x) else np.nan))
    df5_sdh = df5_sdh.dropna().copy()

    df5_sdh["MEDCOST1"] = df5_sdh["MEDCOST1"].astype(int)
    df5_sdh["PERSDOC3"] = df5_sdh["PERSDOC3"].astype(int)
    df5_sdh["PRIMINS2"] = df5_sdh["PRIMINS2"].astype(int)

    colors4 = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    ace_groups = ["0", "1-2", "3-5", "6+"]

    st.markdown('<div class="section-header">Key findings</div>', unsafe_allow_html=True)

    medcost_by_ace = df5.groupby("ACE_GROUP", observed=True)["MEDCOST1"].mean().mul(100).reindex(ace_groups)
    persdoc_by_ace = df5.groupby("ACE_GROUP", observed=True)["PERSDOC3"].apply(lambda x: (x == 0).mean() * 100).reindex(ace_groups)
    primins_by_ace = df5.groupby("ACE_GROUP", observed=True)["PRIMINS2"].apply(lambda x: (x == 0).mean() * 100).reindex(ace_groups)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cost barrier, 0 ACEs", f"{medcost_by_ace.loc['0']:.1f}%")
    k2.metric("Cost barrier, 6+ ACEs", f"{medcost_by_ace.loc['6+']:.1f}%")
    k3.metric("No provider, 6+ ACEs", f"{persdoc_by_ace.loc['6+']:.1f}%")
    k4.metric("Uninsured, 6+ ACEs", f"{primins_by_ace.loc['6+']:.1f}%")

    def single_bar(series, title, ylabel):
        vals = series.reindex(ace_groups)

        fig = go.Figure(go.Bar(
            x=ace_groups,
            y=vals.values,
            marker_color=colors4,
            text=[f"{v:.1f}%" for v in vals.values],
            textposition="outside",
            textfont=dict(color=DARK, size=12, family="JetBrains Mono"),
        ))

        fig.update_layout(**gl(
            title=dict(text=title, font=dict(color=DARK, size=13)),
            xaxis=ax(title="ACE Score Group", type="category"),
            yaxis=ax(title=ylabel, ticksuffix="%", range=[0, vals.max() * 1.25]),
            height=380,
            margin=dict(l=50, r=30, t=55, b=50),
            showlegend=False,
        ))

        return force_dark_plot_text(fig)

    st.markdown(
        '<div class="section-header">Question 1: Do people with more childhood adversity face greater cost barriers to healthcare?</div>',
        unsafe_allow_html=True
    )
    st.plotly_chart(
        single_bar(medcost_by_ace, "Healthcare Cost Barrier by ACE Score Group", "% Could Not Afford Doctor"),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-header">Question 2: Are people with higher ACE scores less likely to have an established personal healthcare provider?</div>',
        unsafe_allow_html=True
    )
    st.plotly_chart(
        single_bar(persdoc_by_ace, "No Personal Healthcare Provider by ACE Score Group", "% With No Personal Provider"),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-header">Question 3: Does childhood adversity predict lower insurance coverage rates?</div>',
        unsafe_allow_html=True
    )
    st.plotly_chart(
        single_bar(primins_by_ace, "No Health Insurance by ACE Score Group", "% With No Insurance"),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-header">Question 4: Does the ACE effect on cost barriers persist within the same income bracket?</div>',
        unsafe_allow_html=True
    )

    df5["INCOME_GROUP"] = df5["_INCOMG1"].apply(
        lambda x: "Low Income (1-3)" if x <= 3 else ("High Income (5-7)" if x >= 5 else None)
    )
    df5_income = df5.dropna(subset=["INCOME_GROUP"])

    medcost_income = (
        df5_income
        .groupby(["ACE_GROUP", "INCOME_GROUP"], observed=True)["MEDCOST1"]
        .mean()
        .mul(100)
        .unstack()
        .reindex(ace_groups)
    )

    fig_inc = go.Figure()

    income_colors = {
        "Low Income (1-3)": "#3498db",
        "High Income (5-7)": "#e74c3c",
    }

    for inc_group in ["Low Income (1-3)", "High Income (5-7)"]:
        fig_inc.add_trace(go.Bar(
            name=inc_group,
            x=ace_groups,
            y=medcost_income[inc_group],
            marker_color=income_colors[inc_group],
            text=[f"{v:.1f}%" for v in medcost_income[inc_group]],
            textposition="outside",
            textfont=dict(color=DARK, size=11, family="JetBrains Mono"),
        ))

    fig_inc.update_layout(**gl(
        title=dict(text="Healthcare Cost Barrier by ACE Group and Income Level", font=dict(color=DARK, size=13)),
        xaxis=ax(title="ACE Score Group", type="category"),
        yaxis=ax(title="% Could Not Afford Doctor", ticksuffix="%", range=[0, medcost_income.max().max() * 1.25]),
        barmode="group",
        height=430,
        margin=dict(l=50, r=30, t=55, b=50),
        legend=dict(title="Income Group", font=dict(color=DARK), title_font=dict(color=DARK)),
    ))

    force_dark_plot_text(fig_inc)
    st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown(
        '<div class="section-header">Question 5: Beyond childhood adversity, does present-day socioeconomic hardship predict being unable to afford care?</div>',
        unsafe_allow_html=True
    )

    df5_sdh["FOOD_INSECURE"] = df5_sdh["SDHFOOD1"].apply(
        lambda x: "Food Insecure" if x <= 2 else ("Food Secure" if x == 5 else None)
    )
    df5_sdh["BILLS_LABEL"] = df5_sdh["SDHBILLS"].map({
        0.0: "Could Pay Bills",
        1.0: "Could Not Pay Bills"
    })
    df5_sdh["TRNSP_LABEL"] = df5_sdh["SDHTRNSP"].map({
        0.0: "No Transport Barrier",
        1.0: "Transport Barrier"
    })

    df5_food = df5_sdh.dropna(subset=["FOOD_INSECURE"])

    sdh_charts = [
        (
            df5_food.groupby("FOOD_INSECURE")["MEDCOST1"].mean().mul(100),
            ["Food Insecure", "Food Secure"],
            "Food Insecurity vs<br>Cost Barrier"
        ),
        (
            df5_sdh.groupby("BILLS_LABEL")["MEDCOST1"].mean().mul(100),
            ["Could Not Pay Bills", "Could Pay Bills"],
            "Bill Payment vs<br>Cost Barrier"
        ),
        (
            df5_sdh.groupby("TRNSP_LABEL")["MEDCOST1"].mean().mul(100),
            ["Transport Barrier", "No Transport Barrier"],
            "Transport Barrier vs<br>Cost Barrier"
        ),
    ]

    s1, s2, s3 = st.columns(3)

    for col_obj, (series, order, title) in zip([s1, s2, s3], sdh_charts):
        vals = series.reindex(order)

        fig_sdh = go.Figure(go.Bar(
            x=order,
            y=vals.values,
            marker_color=["#e74c3c", "#2ecc71"],
            text=[f"{v:.1f}%" for v in vals.values],
            textposition="outside",
            textfont=dict(color=DARK, size=11, family="JetBrains Mono"),
        ))

        fig_sdh.update_layout(**gl(
            title=dict(text=title, font=dict(color=DARK, size=12)),
            xaxis=ax(type="category"),
            yaxis=ax(title="% Could Not Afford Doctor", ticksuffix="%", range=[0, vals.max() * 1.25]),
            height=350,
            margin=dict(l=45, r=20, t=60, b=80),
            showlegend=False,
        ))

        force_dark_plot_text(fig_sdh)
        col_obj.plotly_chart(fig_sdh, use_container_width=True)

    st.markdown(
        '<div class="section-header">Logistic regression: Do ACEs and current hardship, with income and age controls, predict healthcare cost barriers?</div>',
        unsafe_allow_html=True
    )

    columns = ["ACE_SCORE_W", "SDHFOOD1", "SDHBILLS", "SDHTRNSP", "_INCOMG1", "_AGEG5YR"]
    X = df5_sdh[columns]
    y = df5_sdh["MEDCOST1"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    r1, r2, r3 = st.columns(3)
    r1.metric("ROC-AUC", f"{auc:.3f}")
    r2.metric("Positive class share", f"{y_test.mean() * 100:.1f}%")
    r3.metric("Regression sample", f"{len(df5_sdh):,}")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont=dict(color=DARK, size=16, family="JetBrains Mono"),
            showscale=False,
        ))

        fig_cm.update_layout(**gl(
            title=dict(text="Confusion Matrix", font=dict(color=DARK, size=13)),
            xaxis=ax(),
            yaxis=ax(autorange="reversed"),
            height=360,
            margin=dict(l=70, r=30, t=55, b=50),
        ))

        force_dark_plot_text(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        fig_roc = go.Figure()

        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color=BLUE, width=3),
            name=f"ROC curve, AUC = {auc:.3f}",
        ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=TEXT, width=1.5, dash="dash"),
            name="Random model",
        ))

        fig_roc.update_layout(**gl(
            title=dict(text="ROC Curve", font=dict(color=DARK, size=13)),
            xaxis=ax(title="False Positive Rate", range=[0, 1]),
            yaxis=ax(title="True Positive Rate", range=[0, 1]),
            height=360,
            margin=dict(l=60, r=30, t=55, b=50),
        ))

        force_dark_plot_text(fig_roc)
        st.plotly_chart(fig_roc, use_container_width=True)

    report = classification_report(y_test, y_pred, output_dict=True)

    st.markdown(f"""
    <div class="insight-box">
    The logistic regression model predicts healthcare cost barriers using weighted ACE score,
    food insecurity, bill payment hardship, transport barriers, income and age.
    The model achieved <b>ROC-AUC = {auc:.3f}</b>, meaning these factors meaningfully distinguish
    people who face cost barriers from those who do not.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    <b>Overall conclusion:</b> Adverse childhood experiences and current socioeconomic hardship create
    compounding barriers to healthcare access. Higher ACE scores are consistently associated with worse access,
    and current hardship further increases the likelihood of being unable to afford care.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="text-align:center;color:#c0c0cc;font-size:0.7rem;'
        f'margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e8e4dc">'
        f'BRFSS 2024, Angle 5: Healthcare Access, descriptive n = {len(df5):,}, '
        f'logistic regression n = {len(df5_sdh):,}'
        f'</div>',
        unsafe_allow_html=True
    )

# TAB 5 - Protective Factors
with tabs[5]:
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

    st.markdown("""
    <div style="margin-bottom:1rem">
        <div class="dash-title" style="font-size:1.3rem">Protective Factors</div>
        <div class="dash-sub">
            Does having a safe, supportive adult buffer the effect of high ACEs on mental health?
        </div>
    </div>
    """, unsafe_allow_html=True)

    GROUP_ORDER = [
        "Low ACE",
        "High ACE<br>+ Safe Adult",
        "High ACE<br>+ No Safe Adult"
    ]

    GROUP_COLORS = {
        "Low ACE": "steelblue",
        "High ACE<br>+ Safe Adult": "teal",
        "High ACE<br>+ No Safe Adult": "coral",
    }

    ace_labels = {
        "ACEDEPRS": "Parent depression",
        "ACEDRINK": "Parent alcohol abuse",
        "ACEDRUGS": "Parent drug abuse",
        "ACEPRISN": "Parent incarcerated",
        "ACEDIVRC": "Parental divorce",
        "ACEPUNCH": "Physical abuse",
        "ACEHURT1": "Physically hurt",
        "ACESWEAR": "Verbal abuse",
        "ACETOUCH": "Sexual touch",
        "ACETTHEM": "Sexual threat",
        "ACEHVSEX": "Forced sex",
    }

    lsat_map = {
        1: "Very satisfied",
        2: "Satisfied",
        3: "Dissatisfied",
        4: "Very dissatisfied"
    }

    emts_map = {
        1: "Always",
        2: "Usually",
        3: "Sometimes",
        4: "Rarely/Never"
    }

    order_sat = ["Very satisfied", "Satisfied", "Dissatisfied", "Very dissatisfied"]
    order_sup = ["Always", "Usually", "Sometimes", "Rarely/Never"]

    sat_colors = {
        "Very satisfied": "#b40426",
        "Satisfied": "#f7b89c",
        "Dissatisfied": "#9ec5fe",
        "Very dissatisfied": "#3f51c4",
    }

    def force_dark_plot_text(fig):
        fig.update_layout(
            font=dict(family="Sora, sans-serif", color=DARK, size=12),
            title_font=dict(color=DARK),
            legend=dict(font=dict(color=DARK), title_font=dict(color=DARK)),
        )
        fig.update_xaxes(
            color=DARK,
            title_font=dict(color=DARK),
            tickfont=dict(color=DARK),
            linecolor=DARK,
            tickcolor=DARK,
        )
        fig.update_yaxes(
            color=DARK,
            title_font=dict(color=DARK),
            tickfont=dict(color=DARK),
            linecolor=DARK,
            tickcolor=DARK,
        )
        return fig

    def plotly_clean_layout(title, height=460, legend=True):
        return dict(
            title=dict(text=title, font=dict(color=DARK, size=16), x=0.5),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(family="Sora, sans-serif", color=DARK, size=12),
            height=height,
            margin=dict(l=60, r=35, t=65, b=70),
            xaxis=dict(
                gridcolor="#d9d9d9",
                linecolor=DARK,
                tickcolor=DARK,
                tickfont=dict(color=DARK, size=12),
                title_font=dict(color=DARK),
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor="#d9d9d9",
                linecolor=DARK,
                tickcolor=DARK,
                tickfont=dict(color=DARK, size=12),
                title_font=dict(color=DARK),
                zeroline=False,
            ),
            showlegend=legend,
            legend=dict(
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#dddddd",
                borderwidth=1,
                font=dict(color=DARK, size=10),
                title_font=dict(color=DARK),
            ),
        )

    buf = df[df["ACEADSAF"].notna() | df["ACEADNED"].notna()].copy()

    buf["has_safe_adult"] = (
        (buf["ACEADSAF"] >= 4).fillna(False) |
        (buf["ACEADNED"] >= 4).fillna(False)
    )

    low_ace = buf[buf["ACE_SCORE"] < 4].copy()
    high_safe = buf[(buf["ACE_SCORE"] >= 4) & (buf["has_safe_adult"])].copy()
    high_nosafe = buf[(buf["ACE_SCORE"] >= 4) & (~buf["has_safe_adult"])].copy()

    low_ace["Group"] = "Low ACE"
    high_safe["Group"] = "High ACE<br>+ Safe Adult"
    high_nosafe["Group"] = "High ACE<br>+ No Safe Adult"

    plot_df = pd.concat([low_ace, high_safe, high_nosafe], axis=0)
    plot_df["Group"] = pd.Categorical(plot_df["Group"], categories=GROUP_ORDER, ordered=True)

    st.markdown('<div class="section-header">Key findings</div>', unsafe_allow_html=True)

    dep_low = low_ace["ADDEPEV3"].mean() * 100
    dep_safe = high_safe["ADDEPEV3"].mean() * 100
    dep_nosafe = high_nosafe["ADDEPEV3"].mean() * 100

    mh_safe = high_safe["MENTHLTH"].mean()
    mh_nosafe = high_nosafe["MENTHLTH"].mean()

    dep_diff = dep_nosafe - dep_safe
    mh_diff = mh_nosafe - mh_safe

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Low ACE n", f"{len(low_ace):,}")
    k2.metric("High ACE + Safe n", f"{len(high_safe):,}")
    k3.metric("High ACE + No Safe n", f"{len(high_nosafe):,}")
    k4.metric("Depression gap", f"{dep_diff:.1f} pp")
    k5.metric("Mental health gap", f"{mh_diff:.1f} days")

    st.markdown('<div class="section-header">Insights</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(f"""
        <div class="insight-box">
        Among high-ACE respondents, having a <b>safe adult</b> is linked to lower depression:
        <b>{dep_safe:.1f}%</b> vs <b>{dep_nosafe:.1f}%</b>.
        </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown(f"""
        <div class="insight-box">
        High-ACE respondents with a safe adult report <b>{mh_diff:.1f} fewer</b>
        poor mental health days per month.
        </div>
        """, unsafe_allow_html=True)

    with i3:
        st.markdown("""
        <div class="insight-box">
        Protective relationships appear to buffer the long-term mental health burden linked to childhood adversity.
        </div>
        """, unsafe_allow_html=True)

    # GRAPH 1
    st.markdown('<div class="section-header">1. Safe adult groups, ACEADSAF 4-5</div>', unsafe_allow_html=True)

    safe = df[
        (df["ACE_SCORE"] >= 4.0) &
        (df["ACEADSAF"] >= 4)
    ][["ACEADSAF", "MENTHLTH"]].dropna().copy()

    safe["ACEADSAF"] = safe["ACEADSAF"].astype(int).astype(str)

    fig1 = go.Figure()

    for val in sorted(safe["ACEADSAF"].unique()):
        sub = safe[safe["ACEADSAF"] == val]["MENTHLTH"]

        fig1.add_trace(go.Box(
            y=sub,
            name=val,
            boxpoints="outliers",
            jitter=0.25,
            pointpos=0,
            marker=dict(size=4, opacity=0.6, color="steelblue"),
            line=dict(color="#333333"),
            fillcolor="rgba(70, 130, 180, 0.55)",
            hovertemplate="ACEADSAF: " + val + "<br>MENTHLTH: %{y}<extra></extra>",
            showlegend=False,
        ))

    fig1.update_layout(**plotly_clean_layout("", height=520, legend=False))
    fig1.update_layout(
        xaxis=dict(title="ACEADSAF", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK)),
        yaxis=dict(title="MENTHLTH", range=[-1, 31], dtick=5, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK)),
    )
    force_dark_plot_text(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    # GRAPH 2
    st.markdown('<div class="section-header">2. Non-safe adult groups, ACEADSAF 1-3</div>', unsafe_allow_html=True)

    nsafe = df[
        (df["ACE_SCORE"] >= 4.0) &
        (df["ACEADSAF"] < 4)
    ][["ACEADSAF", "MENTHLTH"]].dropna().copy()

    nsafe["ACEADSAF"] = nsafe["ACEADSAF"].astype(int).astype(str)

    fig2 = go.Figure()

    for val in sorted(nsafe["ACEADSAF"].unique()):
        sub = nsafe[nsafe["ACEADSAF"] == val]["MENTHLTH"]

        fig2.add_trace(go.Box(
            y=sub,
            name=val,
            boxpoints="outliers",
            jitter=0.25,
            pointpos=0,
            marker=dict(size=4, opacity=0.6, color="steelblue"),
            line=dict(color="#333333"),
            fillcolor="rgba(70, 130, 180, 0.55)",
            hovertemplate="ACEADSAF: " + val + "<br>MENTHLTH: %{y}<extra></extra>",
            showlegend=False,
        ))

    fig2.update_layout(**plotly_clean_layout("", height=520, legend=False))
    fig2.update_layout(
        xaxis=dict(title="ACEADSAF", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK)),
        yaxis=dict(title="MENTHLTH", range=[-1, 31], dtick=5, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK)),
    )
    force_dark_plot_text(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    # GRAPH 3
    st.markdown('<div class="section-header">3. ACE item prevalence and ACE score distribution</div>', unsafe_allow_html=True)

    fig3 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("ACE Item Prevalence", "ACE Score Distribution"),
        horizontal_spacing=0.18
    )

    prev = df[list(ace_labels)].mean().mul(100)
    prev.index = list(ace_labels.values())
    prev = prev.sort_values()

    fig3.add_trace(go.Bar(
        x=prev.values,
        y=prev.index,
        orientation="h",
        marker_color="steelblue",
        hovertemplate="%{y}<br>Prevalence: %{x:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    score_counts = df["ACE_SCORE"].value_counts().sort_index()

    fig3.add_trace(go.Bar(
        x=score_counts.index.astype(int).astype(str),
        y=score_counts.values,
        marker_color="steelblue",
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="ACE Score: %{x}<br>Count: %{y:,}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig3.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Sora, sans-serif", color=DARK, size=12),
        height=520,
        margin=dict(l=180, r=35, t=70, b=70),
        showlegend=False,
    )

    fig3.update_xaxes(title_text="Prevalence (%)", ticksuffix="%", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=1)
    fig3.update_yaxes(gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), row=1, col=1)
    fig3.update_xaxes(title_text="ACE Score", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=2)
    fig3.update_yaxes(title_text="Count", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=2)

    force_dark_plot_text(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    # GRAPH 4
    st.markdown('<div class="section-header">4. Support and health indicators</div>', unsafe_allow_html=True)

    fig4 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Life Satisfaction (LSATISFY)",
            "Emotional Support (EMTSUPRT)",
            "Poor Physical Health Days (PHYSHLTH)",
            "Poor Mental Health Days (MENTHLTH)"
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.20
    )

    lsat_counts = df["LSATISFY"].dropna().astype(int).map(lsat_map).value_counts().reindex(order_sat)

    fig4.add_trace(go.Bar(
        x=lsat_counts.index,
        y=lsat_counts.values,
        marker_color="teal",
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    emts_counts = df["EMTSUPRT"].dropna().astype(int).map(emts_map).value_counts().reindex(order_sup)

    fig4.add_trace(go.Bar(
        x=emts_counts.index,
        y=emts_counts.values,
        marker_color="coral",
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig4.add_trace(go.Histogram(
        x=df["PHYSHLTH"].dropna(),
        xbins=dict(start=-0.5, end=30.5, size=1),
        marker_color="steelblue",
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="Days: %{x}<br>Count: %{y:,}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig4.add_trace(go.Histogram(
        x=df["MENTHLTH"].dropna(),
        xbins=dict(start=-0.5, end=30.5, size=1),
        marker_color="mediumpurple",
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="Days: %{x}<br>Count: %{y:,}<extra></extra>",
        showlegend=False,
    ), row=2, col=2)

    fig4.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Sora, sans-serif", color=DARK, size=12),
        height=760,
        margin=dict(l=70, r=35, t=80, b=80),
        showlegend=False,
        bargap=0.08,
    )

    fig4.update_xaxes(tickangle=20, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=1)
    fig4.update_xaxes(tickangle=20, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=2)
    fig4.update_xaxes(title_text="Days in past 30", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=2, col=1)
    fig4.update_xaxes(title_text="Days in past 30", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=2, col=2)

    fig4.update_yaxes(title_text="Count", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=1)
    fig4.update_yaxes(title_text="Count", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=2)
    fig4.update_yaxes(title_text="Count", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=2, col=1)
    fig4.update_yaxes(title_text="Count", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=2, col=2)

    force_dark_plot_text(fig4)
    st.plotly_chart(fig4, use_container_width=True)

    # GRAPH 5
    st.markdown('<div class="section-header">5. Mean ACE score by life satisfaction</div>', unsafe_allow_html=True)

    sub_lsat = df[["LSATISFY", "ACE_SCORE"]].dropna().copy()
    sub_lsat["LSATISFY"] = sub_lsat["LSATISFY"].astype(int).map(lsat_map)

    ace_lsat = (
        sub_lsat
        .groupby("LSATISFY")["ACE_SCORE"]
        .agg(["mean", "sem"])
        .reindex(order_sat)
    )

    fig5 = go.Figure(go.Bar(
        x=ace_lsat.index,
        y=ace_lsat["mean"],
        marker_color="teal",
        marker_line_color="white",
        marker_line_width=1,
        error_y=dict(type="data", array=ace_lsat["sem"], color=DARK, thickness=1.5, width=5),
        hovertemplate="%{x}<br>Mean ACE Score: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))

    fig5.update_layout(**plotly_clean_layout("Mean ACE Score by Life Satisfaction", height=520, legend=False))
    fig5.update_layout(
        xaxis=dict(title="", tickangle=15, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK)),
        yaxis=dict(title="Mean ACE Score", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK)),
    )

    force_dark_plot_text(fig5)
    st.plotly_chart(fig5, use_container_width=True)

    # GRAPH 6
    st.markdown('<div class="section-header">6. Poor mental health days by group</div>', unsafe_allow_html=True)

    fig6 = go.Figure()

    for group in GROUP_ORDER:
        sub = plot_df[plot_df["Group"] == group]["MENTHLTH"].dropna()

        fig6.add_trace(go.Box(
            y=sub,
            name=group,
            boxpoints="outliers",
            jitter=0.25,
            pointpos=0,
            marker=dict(size=4, opacity=0.60, color=GROUP_COLORS[group], line=dict(color="#333333", width=0.4)),
            line=dict(color="#333333", width=1.2),
            fillcolor=GROUP_COLORS[group],
            opacity=0.85,
            hovertemplate=group.replace("<br>", " ") + "<br>MENTHLTH: %{y}<extra></extra>",
            showlegend=False,
        ))

    fig6.update_layout(**plotly_clean_layout("Poor Mental Health Days (MENTHLTH)", height=520, legend=False))
    fig6.update_layout(
        xaxis=dict(title="", categoryorder="array", categoryarray=GROUP_ORDER, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK)),
        yaxis=dict(title="Days in past 30", range=[-1.5, 31.5], dtick=5, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK)),
    )

    force_dark_plot_text(fig6)
    st.plotly_chart(fig6, use_container_width=True)

    # GRAPH 7
    st.markdown('<div class="section-header">7. Life satisfaction and depression outcomes</div>', unsafe_allow_html=True)

    fig7 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Life Satisfaction — % per category",
            "Ever Diagnosed with Depression (ADDEPEV3)"
        ),
        horizontal_spacing=0.16
    )

    lsat_props = (
        plot_df[["Group", "LSATISFY"]]
        .dropna()
        .assign(LSATISFY=lambda d: d["LSATISFY"].astype(int).map(lsat_map))
        .groupby(["Group", "LSATISFY"])
        .size()
        .groupby(level=0)
        .transform(lambda x: x / x.sum() * 100)
        .reset_index(name="pct")
        .pivot(index="Group", columns="LSATISFY", values="pct")
        .reindex(GROUP_ORDER)
        .reindex(columns=order_sat)
        .fillna(0)
    )

    for cat in order_sat:
        fig7.add_trace(go.Bar(
            name=cat,
            x=GROUP_ORDER,
            y=lsat_props[cat].values,
            marker_color=sat_colors[cat],
            marker_line_color="white",
            marker_line_width=1,
            width=0.17,
            hovertemplate="%{x}<br>" + cat + ": %{y:.1f}%<extra></extra>",
        ), row=1, col=1)

    dep_pcts = (
        plot_df[["Group", "ADDEPEV3"]]
        .dropna()
        .groupby("Group")["ADDEPEV3"]
        .mean()
        .mul(100)
        .reindex(GROUP_ORDER)
    )

    fig7.add_trace(go.Bar(
        x=GROUP_ORDER,
        y=dep_pcts.values,
        marker_color=[GROUP_COLORS[g] for g in GROUP_ORDER],
        marker_line_color="white",
        marker_line_width=1,
        width=0.5,
        hovertemplate="%{x}<br>Depression: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig7.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Sora, sans-serif", color=DARK, size=12),
        height=560,
        margin=dict(l=70, r=35, t=80, b=90),
        barmode="group",
        legend=dict(
            title="Satisfaction",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#dddddd",
            borderwidth=1,
            font=dict(size=9, color=DARK),
            title_font=dict(color=DARK),
            x=0.47,
            y=0.98,
            xanchor="right",
            yanchor="top",
        ),
    )

    fig7.update_xaxes(categoryorder="array", categoryarray=GROUP_ORDER, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), tickangle=15, row=1, col=1)
    fig7.update_yaxes(title_text="% of group", ticksuffix="%", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=1)

    fig7.update_xaxes(categoryorder="array", categoryarray=GROUP_ORDER, gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), row=1, col=2)
    fig7.update_yaxes(title_text="Prevalence (%)", ticksuffix="%", gridcolor="#d9d9d9", linecolor=DARK, tickfont=dict(color=DARK), title_font=dict(color=DARK), range=[0, dep_pcts.max() * 1.18], row=1, col=2)

    force_dark_plot_text(fig7)
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown('<div class="section-header">Statistical results</div>', unsafe_allow_html=True)

    g1 = high_safe["MENTHLTH"].dropna()
    g2 = high_nosafe["MENTHLTH"].dropna()

    stat, p = mannwhitneyu(g1, g2, alternative="two-sided")

    dep_s = high_safe["ADDEPEV3"].dropna()
    dep_ns = high_nosafe["ADDEPEV3"].dropna()

    x_s, n_s = int(dep_s.sum()), len(dep_s)
    x_ns, n_ns = int(dep_ns.sum()), len(dep_ns)

    _, p_dep = proportions_ztest([x_s, x_ns], [n_s, n_ns])
    ci_lo_p, ci_hi_p = confint_proportions_2indep(x_s, n_s, x_ns, n_ns)

    st.markdown(f"""
    <div class="insight-box">
    <b>Mann-Whitney U test:</b> High ACE + Safe vs High ACE + No Safe for MENTHLTH:
    U = {stat:,.0f}, p = {p:.4f}.<br><br>
    <b>Depression proportion test:</b> Safe = {dep_s.mean() * 100:.1f}%
    ({x_s}/{n_s}), No Safe = {dep_ns.mean() * 100:.1f}% ({x_ns}/{n_ns}),
    difference = {(dep_s.mean() - dep_ns.mean()) * 100:+.1f} pp,
    95% CI [{ci_lo_p:.3f}, {ci_hi_p:.3f}], p = {p_dep:.4f}.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="text-align:center;color:#c0c0cc;font-size:0.7rem;'
        f'margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e8e4dc">'
        f'BRFSS 2024, Angle 6: Protective Factors, '
        f'Low ACE n = {len(low_ace):,}, '
        f'High ACE + Safe n = {len(high_safe):,}, '
        f'High ACE + No Safe n = {len(high_nosafe):,}'
        f'</div>',
        unsafe_allow_html=True
    )

# TAB 6 - Depression Risk (Neural Network)
with tabs[6]:
    import joblib

    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="dash-title" style="font-size:1.3rem">Personal Depression Risk</div>
        <div class="dash-sub">Neural network trained on BRFSS 2024, enter your profile to get a personalized risk estimate</div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        model_nn = joblib.load("/Users/kvantageorg/Desktop/Data_prj_db/model_nn.pkl")
        scaler   = joblib.load("/Users/kvantageorg/Desktop/Data_prj_db/scaler_nn.pkl")
        return model_nn, scaler

    model_nn, scaler = load_model()

    st.markdown('<div class="section-header">Your profile</div>', unsafe_allow_html=True)

    with st.expander("What are ACEs? Click to learn more"):
        st.markdown("""
        **ACEs (Adverse Childhood Experiences)** are traumatic events that happen before age 18.
        This survey measures 11 types:

        **Household dysfunction**
        - Parent/guardian with depression or mental illness
        - Parent/guardian with alcohol or drug problems
        - Household member in prison
        - Parents separated or divorced
        - Witnessing domestic violence

        **Abuse**
        - Physical abuse (being hit, punched)
        - Emotional abuse (being sworn at, humiliated)
        - Sexual abuse

        **Your ACE Score** = how many of these 11 you experienced (0-11)

        *Example: if you had a parent with alcohol problems + witnessed violence + were emotionally abused = ACE score of 3*
        """)

    col_a, col_b = st.columns(2)

    with col_a:
        ace_input = st.slider("How many ACE types did you experience? (0-11)", 0, 11, 0)
        age_input = st.selectbox("Age group", [
            "18-24", "25-29", "30-34", "35-39", "40-44",
            "45-49", "50-54", "55-59", "60-64", "65-69",
            "70-74", "75-79", "80+"
        ])
        age_map_rev = {
            "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
            "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
            "70-74": 11, "75-79": 12, "80+": 13
        }

    with col_b:
        sex_input    = st.radio("Sex", ["Male", "Female"], horizontal=True)
        income_input = st.selectbox("Annual household income", [
            "<$15k", "$15-25k", "$25-35k", "$35-50k", "$50-100k", "$100-200k", "$200k+"
        ])
        lonely_input = st.radio("Do you often feel lonely?", ["No", "Yes"], horizontal=True)
        income_map_rev = {
            "<$15k": 1, "$15-25k": 2, "$25-35k": 3,
            "$35-50k": 4, "$50-100k": 5, "$100-200k": 6, "$200k+": 7
        }

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Calculate my risk", type="primary"):
        X_input = np.array([[
            ace_input,
            age_map_rev[age_input],
            1.0 if sex_input == "Female" else 0.0,
            income_map_rev[income_input],
            1.0 if lonely_input == "Yes" else 0.0,
        ]])
        X_scaled = scaler.transform(X_input)
        prob = model_nn.predict_proba(X_scaled)[0][1] * 100

        avg_by_ace = {0: 9.3, 1: 12.0, 2: 15.9, 3: 20.0, 4: 26.7, 5: 31.0,
                      6: 35.0, 7: 38.0, 8: 40.0, 9: 41.0, 10: 41.5, 11: 42.1}
        avg = avg_by_ace.get(ace_input, 42.1)

        st.markdown('<div class="section-header">Your result</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Your estimated risk", f"{prob:.1f}%")
        r2.metric("Average for your ACE group", f"{avg:.1f}%")
        delta = prob - avg
        r3.metric("vs group average", f"{delta:+.1f}%")

        if prob < 15:
            color, label = GREEN, "Low risk"
        elif prob < 30:
            color, label = YELLOW, "Moderate risk"
        elif prob < 45:
            color, label = ORANGE, "High risk"
        else:
            color, label = RED, "Very high risk"

        st.markdown(f"""
        <div style="background:{color}22; border-left: 4px solid {color};
             border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0;">
            <div style="font-size:1.1rem; font-weight:600; color:{color}">{label}</div>
            <div style="font-size:0.85rem; color:#6b6b7b; margin-top:0.3rem">
                Estimated depression risk: <b>{prob:.1f}%</b> based on your profile
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">What this means</div>', unsafe_allow_html=True)
        tips = []
        if ace_input >= 4:
            tips.append("High ACE score suggests significant childhood adversity. This is one of the strongest predictors in our model.")
        if lonely_input == "Yes":
            tips.append("Loneliness is a major risk factor. People who feel lonely show 46% depression rates in our data.")
        if lonely_input == "No" and ace_input >= 4:
            tips.append("Despite high ACEs, not feeling lonely is a strong protective factor.")
        if income_map_rev[income_input] <= 2:
            tips.append("Lower income is associated with reduced healthcare access and higher stress.")
        tips.append("Emotional support significantly buffers ACE effects. Those with strong support show 14% lower depression rates even with 5+ ACEs.")
        tips.append("This is an estimate based on population data, not a clinical diagnosis. Please consult a professional if you have concerns.")

        for tip in tips:
            st.markdown(f"<div style='padding: 0.4rem 0; font-size:0.85rem; color:#1a1a2e'>{tip}</div>",
                        unsafe_allow_html=True)


# TAB 7 - AI Advisor (Bot)
with tabs[7]:
    from groq import Groq

    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="dash-title" style="font-size:1.3rem">ACE Risk Advisor</div>
        <div class="dash-sub">Describe your background and get a personalized risk summary based on BRFSS 2024 data</div>
    </div>
    """, unsafe_allow_html=True)

    SYSTEM_PROMPT = """
    You are a health data advisor specializing in Adverse Childhood Experiences (ACEs).
    Use these real statistics from BRFSS 2024 survey (n=35,969 respondents with ACE data):

    Depression rates by ACE group:
    - 0 ACEs: 9.3% | 1-2 ACEs: 15.9% | 3-4 ACEs: 26.7% | 5+ ACEs: 42.1%

    Mean poor mental health days (last 30 days):
    - 0 ACEs: 2.2 days | 1-2 ACEs: 3.5 days | 3-4 ACEs: 5.6 days | 5+ ACEs: 8.7 days

    Loneliness: lonely people have 46% depression rate vs 15% non-lonely.
    Emotional support buffer: high ACEs + high support = 35.9% depression vs 50% with low support.
    Each additional ACE increases depression odds by 33% (OR = 1.332).

    When user describes their background:
    1. Estimate their likely ACE score range
    2. Quote the relevant statistics
    3. Mention protective factors (emotional support helps significantly)
    4. Always recommend professional help
    5. Be empathetic, never clinical or cold
    6. Keep response under 200 words
    7. Respond in the same language the user writes in
    Never diagnose. Never say anything is certain.
    """

    if "groq_history" not in st.session_state:
        st.session_state.groq_history = [
            {"role": "assistant", "content": "Hi! I can help you understand how childhood experiences may relate to adult health outcomes based on real survey data. Tell me a bit about your background or ask me anything about ACE risks."}
        ]

    for msg in st.session_state.groq_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Describe your experiences or ask about ACE health risks...")

    if user_input:
        st.session_state.groq_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                             + st.session_state.groq_history,
                    stream=True,
                )
                for chunk in completion:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"API error: {e}")

        if full_response:
            st.session_state.groq_history.append({"role": "assistant", "content": full_response})

    if len(st.session_state.groq_history) > 1:
        if st.button("Clear chat", type="secondary"):
            st.session_state.groq_history = [
                {"role": "assistant", "content": "Hi! I can help you understand how childhood experiences may relate to adult health outcomes based on real survey data. Tell me a bit about your background or ask me anything about ACE risks."}
            ]
            st.rerun()
