import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Flight Price Intelligence System",
    page_icon="✈️",
    layout="wide"
)

# ---------------- CACHED LOADING ----------------

@st.cache_data
def load_data():
    return pd.read_csv("airlines_flights_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_metrics():
    with open("model_metrics.json", "r") as f:
        return json.load(f)

df = load_data()
model = load_model()
metrics = load_metrics()

# Feature importance data for charts
feat_imp_data = metrics.get("feature_importance", {})

# ---------------- HEADER ----------------

st.title("✈️ Flight Ticket Price Intelligence Dashboard")
st.caption("Smart Airline Price Prediction & Analytics")

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🛫 Flight Input Parameters")

airline = st.sidebar.selectbox("Airline", sorted(df["airline"].unique()))
source_city = st.sidebar.selectbox("Source City", sorted(df["source_city"].unique()))

dest_options = sorted([c for c in df["destination_city"].unique() if c != source_city])
destination_city = st.sidebar.selectbox("Destination City", dest_options)

departure_time = st.sidebar.selectbox("Departure Time", df["departure_time"].unique())
arrival_time = st.sidebar.selectbox("Arrival Time", df["arrival_time"].unique())
stops = st.sidebar.selectbox("Stops", df["stops"].unique())
class_type = st.sidebar.selectbox("Class", df["class"].unique())

dur_min = float(df["duration"].min())
dur_max = float(df["duration"].max())
duration = st.sidebar.slider("Duration (hours)", dur_min, dur_max, 5.0, step=0.5)

days_left = st.sidebar.slider("Days Left to Departure", 1, 60, 15)

# ---------------- INPUT DATA ----------------

input_df = pd.DataFrame({
    "airline": [airline],
    "source_city": [source_city],
    "departure_time": [departure_time],
    "stops": [stops],
    "arrival_time": [arrival_time],
    "destination_city": [destination_city],
    "class": [class_type],
    "duration": [duration],
    "days_left": [days_left]
})

# ---------------- PREDICTION ----------------

st.subheader("🎯 Price Prediction")

prediction = model.predict(input_df)[0]

# Contextual stats for the selected route + class
route_mask = (
    (df["airline"] == airline) &
    (df["source_city"] == source_city) &
    (df["destination_city"] == destination_city) &
    (df["class"] == class_type)
)
route_df = df[route_mask]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Predicted Price", f"₹ {prediction:,.0f}")

with col2:
    avg_price = df["price"].mean()
    diff = prediction - avg_price
    st.metric("Overall Average", f"₹ {avg_price:,.0f}", delta=f"₹ {diff:,.0f}", delta_color="inverse")

with col3:
    if len(route_df) > 0:
        route_avg = route_df["price"].mean()
        st.metric("Route Average", f"₹ {route_avg:,.0f}", delta=f"₹ {prediction - route_avg:,.0f}", delta_color="inverse")
    else:
        st.metric("Route Average", "N/A")

with col4:
    if len(route_df) > 0:
        percentile = (route_df["price"] < prediction).mean() * 100
        st.metric("Price Percentile", f"{percentile:.0f}%")
    else:
        st.metric("Price Percentile", "N/A")

# Price suggestion
if len(route_df) > 0:
    route_min = route_df["price"].min()
    route_max = route_df["price"].max()
    if prediction <= route_df["price"].quantile(0.25):
        st.success(f"Great deal! This price is in the bottom 25% for this route (range: ₹{route_min:,} – ₹{route_max:,})")
    elif prediction >= route_df["price"].quantile(0.75):
        st.warning(f"Above average! This price is in the top 25% for this route (range: ₹{route_min:,} – ₹{route_max:,})")
    else:
        st.info(f"Fair price for this route (range: ₹{route_min:,} – ₹{route_max:,})")

st.divider()

# ---------------- TABS ----------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Analytics",
    "🗺️ Route Analysis",
    "🔍 Insights & Simulator",
    "📋 Dataset"
])

# ============ TAB 1: PRICE ANALYTICS ============
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Airline × Class Heatmap")
        heatmap_data = df.pivot_table(
            values="price", index="airline", columns="class", aggfunc="mean"
        )
        fig = px.imshow(
            heatmap_data, text_auto=",.0f",
            color_continuous_scale="RdYlGn_r",
            labels=dict(color="Avg Price (₹)")
        )
        fig.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Price Distribution by Class")
        fig2 = px.histogram(
            df, x="price", color="class", nbins=80,
            barmode="overlay", opacity=0.7,
            labels={"price": "Price (₹)"}
        )
        fig2.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Airline Price Comparison")
        fig3 = px.box(
            df, x="airline", y="price", color="class",
            labels={"price": "Price (₹)"}
        )
        fig3.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("#### Stops vs Price")
        fig_stops = px.violin(
            df, x="stops", y="price", color="class", box=True,
            labels={"price": "Price (₹)"}
        )
        fig_stops.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig_stops, use_container_width=True)

    # Price vs Days Left trend
    st.markdown("#### Price Trend by Days Left to Departure")
    trend = df.groupby(["days_left", "class"])["price"].mean().reset_index()
    fig4 = px.line(
        trend, x="days_left", y="price", color="class",
        labels={"days_left": "Days Left", "price": "Avg Price (₹)"}
    )
    fig4.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig4, use_container_width=True)

# ============ TAB 2: ROUTE ANALYSIS ============
with tab2:
    st.markdown("#### Route-level Price Summary")

    route_stats = df.groupby(["source_city", "destination_city", "class"]).agg(
        avg_price=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        flights=("price", "count")
    ).reset_index()
    route_stats["avg_price"] = route_stats["avg_price"].round(0)

    selected_class = st.selectbox("Filter by Class", ["All"] + sorted(df["class"].unique().tolist()))
    if selected_class != "All":
        route_stats = route_stats[route_stats["class"] == selected_class]

    st.dataframe(
        route_stats.sort_values("avg_price", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "avg_price": st.column_config.NumberColumn("Avg Price", format="₹%,.0f"),
            "min_price": st.column_config.NumberColumn("Min Price", format="₹%,.0f"),
            "max_price": st.column_config.NumberColumn("Max Price", format="₹%,.0f"),
        }
    )

    # Map
    st.markdown("#### Flight Route Map")
    city_coords = {
        "Delhi": [28.6139, 77.2090],
        "Mumbai": [19.0760, 72.8777],
        "Bangalore": [12.9716, 77.5946],
        "Kolkata": [22.5726, 88.3639],
        "Chennai": [13.0827, 80.2707],
        "Hyderabad": [17.3850, 78.4867]
    }

    if source_city in city_coords and destination_city in city_coords:
        src = city_coords[source_city]
        dst = city_coords[destination_city]

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattergeo(
            lon=[src[1], dst[1]],
            lat=[src[0], dst[0]],
            mode="markers+text+lines",
            text=[source_city, destination_city],
            textposition="top center",
            marker=dict(size=12, color=["green", "red"]),
            line=dict(width=2, color="royalblue")
        ))
        fig_map.update_geos(
            scope="asia",
            center=dict(lat=22, lon=80),
            projection_scale=4,
            showland=True
        )
        fig_map.update_layout(height=450, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_map, use_container_width=True)

    # Departure time analysis
    st.markdown("#### Price by Departure Time")
    dep_prices = df.groupby(["departure_time", "class"])["price"].mean().reset_index()
    time_order = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    dep_prices["departure_time"] = pd.Categorical(dep_prices["departure_time"], categories=time_order, ordered=True)
    dep_prices = dep_prices.sort_values("departure_time")
    fig_dep = px.bar(
        dep_prices, x="departure_time", y="price", color="class",
        barmode="group", labels={"price": "Avg Price (₹)", "departure_time": "Departure Time"}
    )
    fig_dep.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_dep, use_container_width=True)

# ============ TAB 3: MODEL INSIGHTS ============
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Key Price Drivers")
        if feat_imp_data:
            fi_df = pd.DataFrame(
                list(feat_imp_data.items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Viridis"
            )
            fig_fi.update_layout(height=500, margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)

    with col_b:
        st.markdown("#### Price Sensitivity Simulator")
        st.caption("See how price changes as days-left varies for your selected route")

        sim_days = list(range(1, 50, 2))
        sim_prices = []
        for d in sim_days:
            sim_input = input_df.copy()
            sim_input["days_left"] = d
            sim_prices.append(model.predict(sim_input)[0])

        sim_df = pd.DataFrame({"Days Left": sim_days, "Predicted Price (₹)": sim_prices})
        fig_sim = px.area(
            sim_df, x="Days Left", y="Predicted Price (₹)",
            markers=True
        )
        fig_sim.update_layout(height=500, margin=dict(t=20, b=20))
        st.plotly_chart(fig_sim, use_container_width=True)

# ============ TAB 4: DATASET ============
with tab4:
    st.markdown(f"#### Dataset Overview ({len(df):,} records)")

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Airlines", df["airline"].nunique())
    stat_col2.metric("Routes", df.groupby(["source_city", "destination_city"]).ngroups)
    stat_col3.metric("Avg Price", f"₹ {df['price'].mean():,.0f}")
    stat_col4.metric("Price Range", f"₹ {df['price'].min():,} – ₹ {df['price'].max():,}")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )