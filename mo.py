
# ### Yellowstone Flood Mobility Analysis: Modular and Reproducible Workflow
# 
# This notebook presents an end-to-end, reproducible pipeline for spatial mobility analysis.
# The analysis covers multiple Yellowstone study regions, including Cody, Jackson, Yellowstone National Park, and Gardiner.
# Additional regions include West Yellowstone and Cooke City/Silver Gate.
# 
# The workflow consists of the following steps:
# 1. Loads and cleans block group, state population, and mobility data.
# 2. Extracts a study region (e.g., Cody, WY) and clips spatial units.
# 3. Merges the Advan mobility data for that region.
# 4. Builds hourly and daily arrival patterns.
# 5. Extracts and maps origin-destination flows (county and state).
# 6. Summarizes who is visiting (pie charts, pre/post flood maps).
# 7. Models relationships between distance, population, and visitation.
# 
# The workflow is fully modular.
# - all reusable logic lives in functions,
# - Each place (Cody, Jackson, etc.) is run with one `run_full_pipeline_for_place()` call.
# 

### 0. Imports, Plot Style, and Global Constants

import os
from pathlib import Path
import json
import calendar
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import networkx as nx
import osmnx as ox
from shapely.ops import unary_union
from pyproj import Geod
import folium
from matplotlib.ticker import PercentFormatter

# Global Warning suppression
import warnings 
warnings.filterwarnings ('ignore')

# Plot style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Filepaths 
SHAPEFILE_WY_CBG = r'F:\Yellow Stone Flood-2022\floodanalysis\Data\tl_2018_56_bg\tl_2018_56_bg.shp'
SHAPEFILE_MT_CBG = r'F:\Yellow Stone Flood-2022\floodanalysis\Data\tl_2018_30_bg\tl_2018_30_bg.shp'  
STATE_POP_FILE   = r'F:\Yellow Stone Flood-2022\floodanalysis\Data\States Population\state_population_2022.shp'
ADVAN_CSV_WY     = r'F:\Yellow Stone Flood-2022\floodanalysis\Data\Advan Mobility\Advan_NEIGHBORHOOD_PATTERN_US_WY.csv'
ADVAN_CSV_MT     = r'F:\Yellow Stone Flood-2022\floodanalysis\Data\Advan Mobility\Advan_NEIGHBORHOOD_Montana\Advan_NEIGHBORHOOD_PATTERN_US_MT.csv'
US_COUNTIES_FILE = r'D:/WY-COVID Data/R Geometric Datasets/us_counties_shifted.shp'
US_STATES_FILE   = r'D:/WY-COVID Data/R Geometric Datasets/state_geometry/us_states_shifted.shp'
CBG_2019_FILE    = r'F:\Yellow Stone Flood-2022\floodanalysis\Data\us_cbgs\cb_2019_us_bg_500k\cb_2019_us_bg_500k.shp'
STUDY_REGION_SHP = r'F:\Yellow Stone Flood-2022\floodanalysis\Data\NewStudyRegion\Study_area.shp'

# USPS - FIPS lookup
USPS_TO_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11",
    "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21",
    "LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30",
    "NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39",
    "OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49",
    "VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56"
}

### 1 Data Loading Helper
def load_blockgroups(shapefile_paths: list [str]) -> gpd.GeoDataFrame:
    """
    Load the Wyoming, and Montaha CBG shapefile, convert it to WGS84 (EPSG:4326),
    and make sure the GEOID field is a string.
    """
    gdfs = []
    for path in shapefile_paths:
        gdf = gpd.read_file(path)
        gdf = gdf.to_crs(epsg=4326)
        gdf["GEOID"] = gdf["GEOID"].astype("string")
        gdfs.append(gdf)
    # Combine all state CBGs
        combined_cbg = pd.concat(gdfs, ignore_index=True)
    return combined_cbg


def load_state_pop(state_pop_file: str) -> gpd.GeoDataFrame:
    """
    Load the state population polygons and make sure the GEOID is two digits.
    Project the data to EPSG:2163 and calculate the centroids using that coordinate system.
    """
    gdf = gpd.read_file(state_pop_file)
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(2)
    gdf_proj = gdf.to_crs(epsg=2163)
    gdf_proj["centroid"] = gdf_proj.geometry.centroid
    return gdf_proj


def load_advan(csv_paths: list[str], year=2022) -> pd.DataFrame:
    """
     Read the Advan mobility CSV file, convert the timestamps, and keep only the data from the desired year.
    """
    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["AREA"] = df["AREA"].astype("string")
        df["DATE_RANGE_START"] = pd.to_datetime(
            df["DATE_RANGE_START"],
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce"
        )
        df_year = df[df["Y"] == year].copy()
        dfs.append(df_year)

    # Combine all state data
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


### 2 Geospatial Helpers

def get_place_geodata(place_name: str, target_crs) -> gpd.GeoDataFrame:
    """
     Use OSMnx to find the location of a place, such as a city or park, and then convert it to coordinate system.
    """
    gdf = ox.geocode_to_gdf(place_name)
    return gdf.to_crs(target_crs)


def clip_cbgs_to_place(
    wy_cbg: gpd.GeoDataFrame,
    place_gdf: gpd.GeoDataFrame,
    exclude_geoids=None
) -> gpd.GeoDataFrame:
    """
    Combine block group (CBG) polygons from the entire study area, such as Wyoming and Montana, with the place boundary.
    Remove any GEOIDs that are not relevant.
    """
    union_geom = place_gdf.geometry.union_all()
    subset = wy_cbg[wy_cbg.intersects(union_geom)].copy()
    if exclude_geoids:
        subset = subset[~subset["GEOID"].isin(exclude_geoids)]
    return subset


def get_drive_network(place_name: str, to_crs="EPSG:4326"):
    """
    Download a road network that can be driven within a specified location.
    Returns graph, nodes_gdf, edges_gdf.
    """
    G = ox.graph_from_place(place_name, network_type="drive", simplify=True)
    G = ox.project_graph(G, to_crs=to_crs)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    return G, nodes, edges


def get_drive_network_from_polygon(polygon, to_crs="EPSG:4326"):
    """
     Download a road network for any area you define.
     This is helpful for combining several counties or creating own study area.
    """
    G = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
    nodes, edges = ox.graph_to_gdfs(G)
    return G, nodes, edges


def build_custom_region(polygons: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """
    Merge multiple county polygons into a single unified polygon.
    """
    merged = gpd.GeoDataFrame(
        {"geometry": [unary_union([g.geometry.iloc[0] for g in polygons])]},
        crs=polygons[0].crs
    )
    return merged


def add_distance_to_poi(
    gdf_states: gpd.GeoDataFrame,
    poi_name: str,
    distance_col_prefix: str = "poi"
) -> gpd.GeoDataFrame:
    """
    Compute distance (in km) from each state's centroid to the POI.
    """
    poi_gdf = ox.geocode_to_gdf(poi_name).to_crs(gdf_states.crs)
    poi_pt = poi_gdf.geometry.iloc[0].representative_point()
    dist_m = gdf_states["centroid"].distance(poi_pt)
    gdf_states[f"{distance_col_prefix}_dist_m"] = dist_m
    gdf_states[f"{distance_col_prefix}_dist_km"] = dist_m / 1000.0
    return gdf_states

### 3 Mobility Aggregation & Transformation Helpers

def subset_advan_for_place(
    df_2022: pd.DataFrame,
    cbg_subset: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
     Execute a left join to merge Advan rows corresponding to GEOIDs found in the cbg_subset.
     Include the month column, labeled M, in the resulting dataset.
    """
    merged = cbg_subset.merge(
        df_2022,
        left_on="GEOID",
        right_on="AREA",
        how="inner"
    )
    merged["DATE_RANGE_START"] = pd.to_datetime(
        merged["DATE_RANGE_START"], errors="coerce"
    )
    merged["M"] = merged["DATE_RANGE_START"].dt.month
    return merged


def build_hourly_series(
    place_adv: gpd.GeoDataFrame,
    months_to_use=(5,6,7,8),
    aug1_only=True
):
    """
     Computes the total hourly arrivals for May, June, and July by summing across rows.
     For August, includes only the first entry (August 1) when aug1_only is True.
     Returns a concatenated Series indexed by hourly timestamps.
    """
    hourly_series_list = []

    for month in months_to_use:
        month_data = place_adv[place_adv["M"].astype(int) == month]
        if month_data.empty:
            continue

        if (month == 8) and aug1_only:
            row = month_data.iloc[0]
            hourly_list = (
                json.loads(row["STOPS_BY_EACH_HOUR"])
                if isinstance(row["STOPS_BY_EACH_HOUR"], str)
                else row["STOPS_BY_EACH_HOUR"]
            )
            start_date = pd.to_datetime(row["DATE_RANGE_START"]).normalize()
            idx = pd.date_range(start_date, periods=len(hourly_list), freq="h")
            series = pd.Series(hourly_list, index=idx)
            series = series[series.index.date == start_date.date()]
        else:
            hourly_lists = month_data["STOPS_BY_EACH_HOUR"].apply(json.loads).tolist()
            hourly_arr = np.vstack(hourly_lists)
            summed_hourly = hourly_arr.sum(axis=0)
            start_date = pd.to_datetime(
                month_data["DATE_RANGE_START"].iloc[0]
            ).normalize()
            idx = pd.date_range(start_date, periods=summed_hourly.size, freq="h")
            series = pd.Series(summed_hourly, index=idx)

        hourly_series_list.append(series)

    full_hourly_series = pd.concat(hourly_series_list)
    return full_hourly_series


def build_daily_series(
    place_adv: gpd.GeoDataFrame,
    months_to_use=(5,6,7),
    include_aug1=True
):
    """
     Sum daily arrivals for May, June, and July.
     Optionally include August 1 as an aggregated value.
     Return a concatenated series indexed by calendar day.
    """
    series_list = []
    for month in months_to_use:
        mdf = place_adv[place_adv["M"].astype(int) == month]
        daily_lists = mdf["STOPS_BY_DAY"].apply(json.loads).tolist()
        daily_arr = np.vstack(daily_lists)
        city_daily = daily_arr.sum(axis=0)

        start_date = pd.to_datetime(
            mdf["DATE_RANGE_START"].iloc[0]
        ).normalize()
        idx = pd.date_range(start_date, periods=city_daily.size, freq="D")
        series_list.append(pd.Series(city_daily, index=idx))

    if include_aug1:
        aug = place_adv[place_adv["M"].astype(int) == 8]
        if not aug.empty:
            raw_daily = aug["STOPS_BY_DAY"].iloc[0]
            daily_list = (
                json.loads(raw_daily)
                if isinstance(raw_daily, str)
                else raw_daily
            )
            aug1_count = daily_list[0]
            aug1_date = pd.to_datetime(
                aug["DATE_RANGE_START"].iloc[0]
            ).normalize()
            series_list.append(pd.Series([aug1_count], index=[aug1_date]))

    return pd.concat(series_list)


def explode_device_homes(
    place_adv: gpd.GeoDataFrame,
    group_by="state"
):
    """
      The DEVICE_HOME_AREAS dataset is transformed into rows representing origin-destination pairs.
      If group_by is set to 'county', the first five digits of the origin Census Block Group (CBG), which correspond to the county FIPS code, are used.
      If group_by is set to 'state', the first two digits of the origin CBG, which correspond to the state FIPS code, are used.
      The function returns a dictionary with keys representing months and values representing dataframes for May, June, and July.
      Each dataframe contains the columns: origin, destination CBG, and weight.
    """
    assert group_by in ("state", "county")
    
    results = {}
    
    for m in [5, 6, 7]:
        rows = []
        
        month_data = place_adv[place_adv["M"] == m]
        
        for _, r in month_data.iterrows():
            dest = r["AREA"]
            homes = json.loads(r["DEVICE_HOME_AREAS"])
            
            for ocbg, cnt in homes.items():
                if group_by == "county":
                    rows.append({
                        'origin_cnty': ocbg[:5],
                        'dest_cbg':    dest,
                        'weight':      int(cnt)
                    })
                else:  
                    rows.append({
                        'origin_state': ocbg[:2],
                        'dest_cbg':     dest,
                        'weight':       int(cnt)
                    })
        
        dfm = pd.DataFrame(rows)
        
        if dfm.empty:
            col_name = 'origin_cnty' if group_by == "county" else 'origin_state'
            results[m] = pd.DataFrame(columns=[col_name, 'dest_cbg', 'weight'])
        else:
            if group_by == "county":
                results[m] = (
                    dfm.groupby(['origin_cnty', 'dest_cbg'])['weight']
                       .sum()
                       .reset_index()
                )
            else:  
                results[m] = (
                    dfm.groupby(['origin_state', 'dest_cbg'])['weight']
                       .sum()
                       .reset_index()
                )
    
    return results


def summarize_visits_by_state_group(
    place_adv: gpd.GeoDataFrame,
    group_map: dict[str,str],
    groups_order=('ID','CO','SD','WY','MT','Other')
):
    """
      Classify visits into the following groups: ID, CO, SD, WY, MT, and Other.
      Aggregate these groups monthly to generate data suitable for a pie chart visualization.
    """
    records=[]
    for m in [5,6,7]:
        dfm = place_adv[place_adv["M"]==m]
        for _, r in dfm.iterrows():
            homes = json.loads(r["DEVICE_HOME_AREAS"])
            for ocbg, cnt in homes.items():
                st = ocbg[:2]
                grp = group_map.get(st, 'Other')
                records.append({
                    "month": m,
                    "group": grp,
                    "weight": int(cnt)
                })
    summary = pd.DataFrame(records)
    pivot = (
        summary.pivot_table(
            index="month",
            columns="group",
            values="weight",
            aggfunc="sum"
        )
        .fillna(0)
        .reset_index()
    )
    cols = ["month"]+[g for g in groups_order if g in pivot.columns]
    pivot = pivot[cols]
    return pivot


def build_state_level_table(place_adv: gpd.GeoDataFrame) -> pd.DataFrame:

    """
      Total weight is calculated for each month and state Federal Information Processing Standards (FIPS) code.
      This procedure produces a long-form dataset appropriate for regression analysis and distance-decay modeling.
    """
    rows=[]
    for m in [5,6,7]:
        dfm = place_adv[place_adv["M"]==m]
        for _, r in dfm.iterrows():
            homes = json.loads(r["DEVICE_HOME_AREAS"])
            for ocbg, cnt in homes.items():
                st_fips = ocbg[:2]
                rows.append({
                    "month": m,
                    "GEOID": st_fips,
                    "weight": int(cnt)
                })
    long_df = (
        pd.DataFrame(rows)
        .groupby(["month","GEOID"], as_index=False)["weight"]
        .sum()
        .sort_values(["month","GEOID"])
        .reset_index(drop=True)
    )
    return long_df

### 4 Mobility Aggregation and Transformation Helpers

def subset_advan_for_place(
    df_2022: pd.DataFrame,
    cbg_subset: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
      Perform a left join of Advan rows for GEOIDs present in the cbg_subset.
      Add the month column M to the resulting dataset.
    """
    merged = cbg_subset.merge(
        df_2022,
        left_on="GEOID",
        right_on="AREA",
        how="inner"
    )
    merged["DATE_RANGE_START"] = pd.to_datetime(
        merged["DATE_RANGE_START"], errors="coerce"
    )
    merged["M"] = merged["DATE_RANGE_START"].dt.month
    return merged


def build_hourly_series(
    place_adv: gpd.GeoDataFrame,
    months_to_use=(5,6,7,8),
    aug1_only=True
):
    """
      Aggregate hourly arrivals for May, June, and July across all rows.
      For August, include only the first entry (August 1) if aug1_only is set to True.
      Return a concatenated Series indexed by hourly timestamps.
    """
    hourly_series_list = []

    for month in months_to_use:
        month_data = place_adv[place_adv["M"].astype(int) == month]
        if month_data.empty:
            continue

        if (month == 8) and aug1_only:
            row = month_data.iloc[0]
            hourly_list = (
                json.loads(row["STOPS_BY_EACH_HOUR"])
                if isinstance(row["STOPS_BY_EACH_HOUR"], str)
                else row["STOPS_BY_EACH_HOUR"]
            )
            start_date = pd.to_datetime(row["DATE_RANGE_START"]).normalize()
            idx = pd.date_range(start_date, periods=len(hourly_list), freq="h")
            series = pd.Series(hourly_list, index=idx)
            series = series[series.index.date == start_date.date()]
        else:
            hourly_lists = month_data["STOPS_BY_EACH_HOUR"].apply(json.loads).tolist()
            hourly_arr = np.vstack(hourly_lists)
            summed_hourly = hourly_arr.sum(axis=0)
            start_date = pd.to_datetime(
                month_data["DATE_RANGE_START"].iloc[0]
            ).normalize()
            idx = pd.date_range(start_date, periods=summed_hourly.size, freq="h")
            series = pd.Series(summed_hourly, index=idx)

        hourly_series_list.append(series)

    full_hourly_series = pd.concat(hourly_series_list)
    return full_hourly_series


def build_daily_series(
    place_adv: gpd.GeoDataFrame,
    months_to_use=(5,6,7),
    include_aug1=True
):
    """
     Calculate the sum of daily arrivals for MOptionally include August 1 as a single value.ug 1 as single value.
     Return a concatenated Series indexed by calendar day.
    """
    series_list = []
    for month in months_to_use:
        mdf = place_adv[place_adv["M"].astype(int) == month]
        daily_lists = mdf["STOPS_BY_DAY"].apply(json.loads).tolist()
        daily_arr = np.vstack(daily_lists)
        city_daily = daily_arr.sum(axis=0)

        start_date = pd.to_datetime(
            mdf["DATE_RANGE_START"].iloc[0]
        ).normalize()
        idx = pd.date_range(start_date, periods=city_daily.size, freq="D")
        series_list.append(pd.Series(city_daily, index=idx))

    if include_aug1:
        aug = place_adv[place_adv["M"].astype(int) == 8]
        if not aug.empty:
            raw_daily = aug["STOPS_BY_DAY"].iloc[0]
            daily_list = (
                json.loads(raw_daily)
                if isinstance(raw_daily, str)
                else raw_daily
            )
            aug1_count = daily_list[0]
            aug1_date = pd.to_datetime(
                aug["DATE_RANGE_START"].iloc[0]
            ).normalize()
            series_list.append(pd.Series([aug1_count], index=[aug1_date]))

    return pd.concat(series_list)


def explode_device_homes(
    place_adv: gpd.GeoDataFrame,
    group_by="state"
):
    """
       Convert DEVICE_HOME_AREAS into rows representing origin and destination pairs.
       When group_by is set to 'county', use the first five digits of the origin CBG to represent the county FIPS code.
       When group_by is set to 'state', use the first two digits of the origin CBG to represent the state FIPS code.
       Return a dictionary with month numbers as keys and the corresponding DataFrames for May, June, and July as values.
       Each DataFrame contains the columns: origin, destination CBG, and weight.
    """
    assert group_by in ("state", "county"), "group_by must be 'state' or 'county'"
    results = {}
    for m in [5, 6, 7]:
        rows = []
        for _, r in place_adv[place_adv["M"] == m].iterrows():
            dest = r["AREA"]
            homes = json.loads(r["DEVICE_HOME_AREAS"])
            for ocbg, cnt in homes.items():
                if group_by == "county":
                    rows.append({
                        'origin_cnty': ocbg[:5],
                        'dest_cbg': dest,
                        'weight': int(cnt)
                    })
                else:  
                    rows.append({
                        'origin_state': ocbg[:2],
                        'dest_cbg': dest,
                        'weight': int(cnt)
                    })
        dfm = pd.DataFrame(rows)
        
        if dfm.empty:
            if group_by == "county":
                results[m] = pd.DataFrame(columns=['origin_cnty', 'dest_cbg', 'weight'])
            else:
                results[m] = pd.DataFrame(columns=['origin_state', 'dest_cbg', 'weight'])
        else:
            if group_by == "county":
                results[m] = (
                    dfm.groupby(['origin_cnty', 'dest_cbg'])['weight']
                    .sum()
                    .reset_index()
                )
            else:  
                results[m] = (
                    dfm.groupby(['origin_state', 'dest_cbg'])['weight']
                    .sum()
                    .reset_index()
                )
    return results

def summarize_visits_by_state_group(
    place_adv: gpd.GeoDataFrame,
    group_map: dict[str,str],
    groups_order=('ID','CO','SD','WY','MT','Other')
):
    """
       Group visits by custom categories: ID, CO, SD, WY, MT, and Other.
       Perform this aggregation for each month to generate input for pie charts.
    """
    records=[]
    for m in [5,6,7]:
        dfm = place_adv[place_adv["M"]==m]
        for _, r in dfm.iterrows():
            homes = json.loads(r["DEVICE_HOME_AREAS"])
            for ocbg, cnt in homes.items():
                st = ocbg[:2]
                grp = group_map.get(st, 'Other')
                records.append({
                    "month": m,
                    "group": grp,
                    "weight": int(cnt)
                })
    summary = pd.DataFrame(records)
    pivot = (
        summary.pivot_table(
            index="month",
            columns="group",
            values="weight",
            aggfunc="sum"
        )
        .fillna(0)
        .reset_index()
    )
    cols = ["month"]+[g for g in groups_order if g in pivot.columns]
    pivot = pivot[cols]
    return pivot


def build_state_level_table(place_adv: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregate total weight by (month, state FIPS).
    Produces long-form table used for regression & distance-decay.
    """
    rows=[]
    for m in [5,6,7]:
        dfm = place_adv[place_adv["M"]==m]
        for _, r in dfm.iterrows():
            homes = json.loads(r["DEVICE_HOME_AREAS"])
            for ocbg, cnt in homes.items():
                st_fips = ocbg[:2]
                rows.append({
                    "month": m,
                    "GEOID": st_fips,
                    "weight": int(cnt)
                })
    long_df = (
        pd.DataFrame(rows)
        .groupby(["month","GEOID"], as_index=False)["weight"]
        .sum()
        .sort_values(["month","GEOID"])
        .reset_index(drop=True)
    )
    return long_df

### 5 Plotting Helpers
#### 5.1 Time Series Plots

def plot_hourly_series(hourly_series: pd.Series, place_label: str):
    fig, ax = plt.subplots(figsize=(12,4))
    hourly_series.plot(ax=ax)

    ax.set_title(
        f'Hourly Arrival Patterns in {place_label}\n(May – July & August 1, 2022)',
        fontsize=14, fontstyle='italic', pad=15
    )
    ax.set_xlabel('Local Date & Hour', fontsize=12)
    ax.set_ylabel('Count of Total Hourly Devices', fontsize=12)

    days_to_show = [1,9,16,23]
    tick_dates=[]
    for month in [5,6,7]:
        last_day = calendar.monthrange(2022, month)[1]
        for day in days_to_show+[last_day]:
            if day <= last_day:
                tick_dates.append(pd.Timestamp(2022, month, day))
    tick_dates.append(pd.Timestamp(2022,8,1))

    data_dates = hourly_series.index.normalize().unique()
    tick_dates = [d for d in tick_dates if d in data_dates]

    ax.set_xticks(tick_dates)

    def custom_formatter(x, pos):
        date = pd.to_datetime(x)
        if date.day == 1:
            return date.strftime('%b %Y')
        else:
            return f"{date.day:02d}"

    ax.set_xticklabels([custom_formatter(d,None) for d in tick_dates], rotation=45)
    plt.tight_layout()
    return fig, ax


def plot_daily_series(daily_series: pd.Series, place_label: str):
    fig, ax = plt.subplots(figsize=(12,4))
    daily_series.plot(ax=ax)

    ax.set_title(
        f'Daily Arrival Patterns in {place_label}\n(May – July & August 1, 2022)',
        fontsize=14, fontstyle='italic', pad=15
    )
    ax.set_xlabel('Local Date', fontsize=12)
    ax.set_ylabel('Arrival Count', fontsize=12)

    ax.set_xlim(pd.to_datetime('2022-05-01'), pd.to_datetime('2022-08-01'))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1,15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.minorticks_off()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax


#### 5.2 Origin Destination Map

def make_od_map(
    place_adv: gpd.GeoDataFrame,
    cbg_centroids: pd.Series,
    month: int,
    center_latlon: tuple[float,float],
    threshold_quantile=0.8
):
    """
       Generate an interactive origin-destination (OD) map for a specified month.
       Filter the data to display only the top flows based on quantile thresholds.
    """
    df_m = place_adv[place_adv["M"] == month]
    recs = []
    for _, row in df_m.iterrows():
        dest = row["AREA"]
        home_map = json.loads(row["DEVICE_HOME_AREAS"])
        for origin, cnt in home_map.items():
            recs.append((origin, dest, int(cnt)))
    edges_df = pd.DataFrame(recs, columns=["origin","destination","weight"])

    if threshold_quantile > 0:
        q = edges_df["weight"].quantile(threshold_quantile)
        edges_df = edges_df[edges_df["weight"] >= q]

    m = folium.Map(location=center_latlon, zoom_start=10, tiles="cartodbpositron")

    for _, r in edges_df.iterrows():
        o, d, w = r["origin"], r["destination"], r["weight"]
        if o in cbg_centroids and d in cbg_centroids:
            coords = [
                [cbg_centroids[o].y, cbg_centroids[o].x],
                [cbg_centroids[d].y, cbg_centroids[d].x]
            ]
            folium.PolyLine(
                locations=coords,
                weight=max(w / edges_df["weight"].max() * 10, 1),
                color="crimson",
                opacity=0.6,
                tooltip=f"{w} devices: {o} → {d}"
            ).add_to(m)

    for dest in edges_df["destination"].unique():
        if dest in cbg_centroids:
            folium.CircleMarker(
                location=[cbg_centroids[dest].y, cbg_centroids[dest].x],
                radius=4,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.7,
                tooltip=f"Destination: {dest}"
            ).add_to(m)

    return m

#### 5.4 County/State Flow Arc Panels
def plot_flow_maps(
    flow_dict: dict[int,pd.DataFrame],
    outline_gdf: gpd.GeoDataFrame,
    outline_label: str,
    dest_centroids: dict[str,object],
    origin_centroids: dict[str,object],
    main_title: str,
    month_names={5:"May",6:"June",7:"July"},
    dpi=600,
    log_scale=True
):
    """
    Plot 1x3 panel flow maps with arcs from origins to destinations.
    
    Parameters:
    -----------
    flow_dict : dict with keys 5,6,7 and DataFrames with columns:
                - For county: ['origin_cnty', 'dest_cbg', 'weight']
                - For state: ['origin_state', 'dest_cbg', 'weight']
    outline_label : "county" or "state"
    """
    geod = Geod(ellps="WGS84")
    months=[5,6,7]
    panel_labels=["(a)","(b)","(c)"]

    # Determine which origin column to use
    origin_col = 'origin_state' if outline_label == "state" else 'origin_cnty'

    all_w = pd.concat([df.weight for df in flow_dict.values()])
    if log_scale:
        norm = mpl.colors.LogNorm(vmin=max(all_w.min(),1), vmax=all_w.max())
    else:
        norm = mpl.colors.Normalize(vmin=all_w.min(), vmax=all_w.max())
    cmap = mpl.cm.plasma_r

    xmin,ymin,xmax,ymax = outline_gdf.total_bounds
    fig, axes = plt.subplots(1,3,figsize=(18,5),dpi=dpi)
    fig.suptitle(main_title, fontsize=18, weight="bold", y=0.96)

    for ax, m, plabel in zip(axes, months, panel_labels):
        outline_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor='#888888',
            linewidth=0.3 if outline_label=="county" else 0.5
        )
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
        ax.set_axis_off()

        dfm = flow_dict[m]
        for _, row in dfm.iterrows():
            o, dc, w = row[origin_col], row["dest_cbg"], row["weight"]
            
            if (o in origin_centroids) and (dc in dest_centroids):
                lon1, lat1 = origin_centroids[o].x, origin_centroids[o].y
                lon2, lat2 = dest_centroids[dc].x, dest_centroids[dc].y
                pts = geod.npts(lon1,lat1,lon2,lat2,30 if outline_label=="county" else 40)
                xs=[lon1]+[p[0] for p in pts]+[lon2]
                ys=[lat1]+[p[1] for p in pts]+[lat2]
                ax.plot(
                    xs, ys,
                    color=cmap(norm(w)),
                    linewidth=(0.3+1.2*norm(w)) if outline_label=="county"
                             else (0.5+2.0*norm(w)),
                    alpha=0.7,
                    solid_capstyle="round"
                )

        dx=[pt.x for pt in dest_centroids.values()]
        dy=[pt.y for pt in dest_centroids.values()]
        ax.scatter(
            dx, dy,
            s=2 if outline_label=="county" else 3,
            c="k",
            alpha=0.3,
            zorder=3
        )

        ax.add_patch(Rectangle(
            (0,0),1,1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=0.8,
            clip_on=False
        ))
        ax.text(
            -0.015,1.02,plabel,
            transform=ax.transAxes,
            fontsize=14,fontweight="bold",
            va="bottom"
        )
        ax.text(
            0.5,1.02,month_names[m],
            transform=ax.transAxes,
            ha="center",va="bottom",
            fontsize=14
        )

        axins=inset_axes(
            ax,width="30%",height="3%",loc="lower center",
            bbox_to_anchor=(0,-0.07,1,1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        cb = mpl.colorbar.ColorbarBase(
            axins,cmap=cmap,norm=norm,
            orientation="horizontal"
        )
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=6,pad=1)
        axins.set_xlabel(
            "Total monthly device counts"
            + (" (log scale)" if log_scale else ""),
            fontsize=8
        )

    plt.tight_layout(rect=[0,0,1,0.94])
    return fig, axes

#### 5.5 Pie Charts

def plot_monthly_pies(
    pivot_df: pd.DataFrame,
    place_label: str,
    month_names={5:"May",6:"June",7:"July"},
    groups=('ID','CO','SD','WY','MT','Other'),
    colors_map=None
):
    """
    Generate a set of three pie charts, each representing the origin contributions by group for May, June, and July, respectively.
    """
    if colors_map is None:
        colors_map = {
            'ID':    '#1f77b4',
            'CO':    '#ff7f0e',
            'SD':    '#2ca02c',
            'WY':    '#d62728',
            'MT':    '#9467bd',
            'Other': '#8c564b'
        }

    fig, axes = plt.subplots(1,3,figsize=(15,5),subplot_kw={'aspect':'equal'})

    for ax, (_, row) in zip(axes, pivot_df.iterrows()):
        m = row['month']
        sizes = [row[g] if g in row else 0 for g in groups]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=groups,
            autopct='%1.1f%%',
            startangle=90,
            colors=[colors_map[g] for g in groups],
            pctdistance=0.75,
            textprops={'fontsize':10,'color':'black'}
        )

        for w in wedges:
            w.set_edgecolor('white')
            w.set_linewidth(1)

        ax.set_title(f"{month_names[m]}", fontsize=12, weight='bold')

    fig.suptitle(
        f"Share of Device Visits to {place_label} \n(May–July 2022)",
        fontsize=14,
        weight='bold',
        y=1.05
    )
    plt.tight_layout()
    return fig, axes

#### 5.6 Choropleth Mapping by Spending

def prep_spending_choropleth(
    us_states_gdf: gpd.GeoDataFrame,
    spending_df: pd.DataFrame,
    before_col="Before Flood",
    during_col="During Flood",
    after_col="After Flood",
    as_fraction=True
):
    """
    Aggregate state-level totals for the periods before, during, and after the flood, associate these values with corresponding state polygons, and normalize the resulting data.
    """
    tmp = us_states_gdf.merge(spending_df, on="STATEFP", how="left")
    g = tmp.copy()

    for c in [before_col, during_col, after_col]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)

    if as_fraction:
        for c in [before_col, during_col, after_col]:
            s = g[c].sum()
            g[c] = g[c] / s if s else 0
        norm = Normalize(
            vmin=0,
            vmax=max(g[[before_col,during_col,after_col]].max())
        )
        cbar_label = "Share of period total"
    else:
        vmin = g[[before_col,during_col,after_col]].min().min()
        vmax = g[[before_col,during_col,after_col]].max().max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar_label = "Total monthly device counts"

    return g, norm, cbar_label


def plot_spending_choropleth_panels(
    g: gpd.GeoDataFrame,
    norm: Normalize,
    cbar_label: str,
    title: str,
    before_col="Before Flood",
    during_col="During Flood",
    after_col="After Flood"
):
    """
    1x3 choropleth panels for Before/During/After.
    """
    fig, axes = plt.subplots(1,3,figsize=(18,6))

    for ax, col, ttl in zip(
        axes,
        [before_col, during_col, after_col],
        ["(a) Before", "(b) During", "(c) After"]
    ):
        g.plot(
            column=col,
            cmap="viridis",
            norm=norm,
            edgecolor="white",
            linewidth=0.4,
            ax=ax
        )
        ax.set_title(ttl, fontsize=13)
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm); sm._A = []
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.035,
        pad=0.08
    )
    cbar.set_label(cbar_label)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0,0.04,1,0.92])
    return fig, axes


#### 5.7 Regression Panels + 3D Scatter Penels

def compute_regression_stats(
    df_states_month: pd.DataFrame,
    pop_col="population",
    w_col="weight"
):
    """
    For each month, perform a linear regression of weight as a function of population.
    The procedure returns a statistical summary table and a dictionary containing the fitted regression lines and confidence interval bands.
    """
    out_stats=[]
    out_fits={}
    for month in [5,6,7]:
        mdf = df_states_month[df_states_month["month"]==month].dropna(
            subset=[w_col,pop_col]
        )
        x = mdf[pop_col].values
        y = mdf[w_col].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        r_squared = r_value**2

        def predict_ci(x_new, x, y, confidence=0.95):
            n=len(x)
            x_mean = np.mean(x)
            sxx = np.sum((x-x_mean)**2)
            sxy = np.sum((x-x_mean)*(y-np.mean(y)))
            syy = np.sum((y-np.mean(y))**2)
            s_yx = np.sqrt((syy - sxy**2/sxx)/(n-2))
            t_val = stats.t.ppf((1+confidence)/2, n-2)

            y_pred = slope*x_new + intercept
            se_pred = s_yx * np.sqrt(1/n + (x_new-x_mean)**2/sxx)
            ci_lower = y_pred - t_val*se_pred
            ci_upper = y_pred + t_val*se_pred
            return y_pred, ci_lower, ci_upper

        x_line = np.linspace(x.min(), x.max(), 100)
        y_pred, ci_lower, ci_upper = predict_ci(x_line, x, y)

        out_stats.append({
            "Month": month,
            "R²": r_squared,
            "p-value": p_value,
            "Slope": slope,
            "n": len(mdf)
        })

        out_fits[month] = {
            "df": mdf,
            "x_line": x_line,
            "y_pred": y_pred,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "r_squared": r_squared
        }

    return out_stats, out_fits


def plot_regression_panels(
    fits_dict,
    place_label: str,
    month_names={5:'May',6:'June',7:'July'},
    colors={5:'#1f77b4',6:'#d62728',7:'#2ca02c'}
):
    """
    Make 1x3 regression panels with CI bands and stats.
    """
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    panel_labels = ['A','B','C']

    xlims=[]; ylims=[]
    for month in [5,6,7]:
        dfm = fits_dict[month]["df"]
        xlims += list(dfm["population"])
        ylims += list(dfm["weight"])
    x_min,x_max = min(xlims), max(xlims)
    y_min,y_max = min(ylims), max(ylims)

    for i,(month,label) in enumerate(zip([5,6,7],panel_labels)):
        ax=axes[i]
        fit=fits_dict[month]
        dfm=fit["df"]
        col=colors[month]

        ax.fill_between(
            fit["x_line"], fit["ci_lower"], fit["ci_upper"],
            alpha=0.2, color=col, label='95% CI'
        )

        ax.scatter(
            dfm["population"], dfm["weight"],
            c=col, alpha=0.7, s=30,
            edgecolors='white', linewidth=0.5, zorder=3
        )

        ax.plot(fit["x_line"], fit["y_pred"], color=col, linewidth=2, zorder=2)

        p_val = fit["p_value"]
        if p_val < 0.001: p_text="p < 0.001"
        elif p_val < 0.01: p_text="p < 0.01"
        elif p_val < 0.05: p_text="p < 0.05"
        else: p_text=f"p = {p_val:.3f}"

        stats_text = (
            f'R² = {fit["r_squared"]:.3f}\n'
            f'{p_text}\n'
            f'n = {len(dfm)}'
        )
        ax.text(
            0.05,0.95,stats_text,
            transform=ax.transAxes,
            fontsize=9,va='top',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='gray',
                alpha=0.9
            )
        )

        ax.set_xlabel('State Population (×10⁶)', fontweight='bold')
        if i==0:
            ax.set_ylabel('Movement Weight', fontweight='bold')
        ax.set_title(month_names[month], fontweight='bold', fontsize=12)

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x,p: f'{x/1e6:.0f}')
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x,p: f'{x:.0f}')
        )

        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        ax.text(-0.15,1.05,label,transform=ax.transAxes,fontsize=8)

    fig.suptitle(
        f'Relationship between State Population and Movement Weight by Month ({place_label})',
        fontsize=14,
        fontweight='bold',
        y=0.95
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig, axes


def plot_3d_panels(
    df_states_month: pd.DataFrame,
    place_label: str,
    distance_col: str,
    weight_col="weight",
    month_names={5:'May',6:'June',7:'July'}
):
    """
     Create a set of three 3D scatter plots arranged in a 1x3 panel layout, displaying population, distance, and weight as axes.
     Ensure that all panels utilize a unified color scale.
    """
    months = [5,6,7]
    mask = df_states_month["month"].isin(months)
    w_all = df_states_month.loc[mask, weight_col].to_numpy()
    norm = Normalize(vmin=np.nanmin(w_all), vmax=np.nanmax(w_all))

    fig = plt.figure(figsize=(18,6), constrained_layout=True)
    axes=[]

    for i,(m) in enumerate(months, start=1):
        ax = fig.add_subplot(1,3,i,projection='3d')
        axes.append(ax)

        dfm = df_states_month[df_states_month["month"]==m]
        x = dfm["population"].to_numpy()
        y = dfm[distance_col].to_numpy()
        z = dfm[weight_col].to_numpy()

        sc = ax.scatter3D(
            x,y,z,
            c=z, cmap='viridis', norm=norm,
            s=60, alpha=0.85,
            edgecolors='white', linewidth=0.5
        )

        ax.set_title(month_names[m], fontsize=13, pad=20, weight='bold')
        ax.set_xlabel('State Population', labelpad=10)
        ax.set_ylabel(f'Distance to {place_label.split(",")[0]} (km)', labelpad=10)
        ax.set_zlabel('Device Counts', labelpad=10)

        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='z', labelsize=9)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        location='bottom',
        orientation='horizontal',
        fraction=0.05,
        pad=0.08,
        aspect=30
    )
    cbar.set_label('Weight', fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Relationship Between State Population, Distance to {place_label.split(',')[0]}, and Movement Weight by Month",
        fontsize=14,
        fontweight='bold'
    )
    return fig, axes


### 5.8 State of Origin Choropleth Analysis

def analyze_state_origin_for_place(
    csv_path: str,
    us_states_gdf: gpd.GeoDataFrame,
    place_label: str,
    as_fraction: bool = True,
    save_fig_path: str = None,
    save_processed_csv: bool = False
) -> dict:
    """
     This function identifies the origin of visitors from each state for a specified location. 
     The function reads a CSV file containing the number of visitors from each state.
     The dataset includes three flood periods: before, during, and after. The function processes this information and generates a map with three panels to display the origin of visitors for each period.
    """
    # Process CSV
    origin_df = pd.read_csv(csv_path)
    origin_df.columns = [c.strip() for c in origin_df.columns]

    # Standardize column schema
    #   Case B (already clean): CSV already has STATEFP and the 3 flood-period columns.
    #   Case A (messy Cody-style): CSV has STUSPS and long verbose headers.

    if (
        "STATEFP" in origin_df.columns and
        "Before Flood" in origin_df.columns and
        "During Flood" in origin_df.columns and
        "After Flood" in origin_df.columns
    ):
        df_clean = origin_df[["STATEFP", "Before Flood", "During Flood", "After Flood"]].copy()

    else:
        rename_map = {
            "Unnamed: 0": "STUSPS",  
            "State": "STUSPS",
            "Before Flood State of Orign Totals": "Before Flood",
            "Before Flood State of Origin Totals": "Before Flood",
            "During Flood State of Orign Totals": "During Flood",
            "During Flood State of Origin Totals": "During Flood",
            "After Flood State of Orign Totals": "After Flood",
            "After Flood State of Origin Totals": "After Flood",
        }
        origin_df = origin_df.rename(columns=rename_map)
        if "STUSPS" not in origin_df.columns:
            raise ValueError(
                "Input CSV is neither Cooke City-style (STATEFP already present) nor "
                "Cody-style (STUSPS present). Please check column headers."
            )

        # Map USPS state abbreviations 
        USPS_TO_FIPS = {
            "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11",
            "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21",
            "LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30",
            "NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39",
            "OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49",
            "VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","PR":"72"
        }

        # Create STATEFP as string FIPS from abbrev.
        origin_df["STATEFP"] = origin_df["STUSPS"].map(USPS_TO_FIPS)

        # Keep only what we need in a consistent order
        df_clean = origin_df[["STATEFP", "Before Flood", "During Flood", "After Flood"]].copy()

    # Ensure STATEFP is zero-padded string
    df_clean["STATEFP"] = df_clean["STATEFP"].astype(str).str.zfill(2)

    # Clean numeric columns
    for col in ["Before Flood", "During Flood", "After Flood"]:
        df_clean[col] = (
            df_clean[col]
            .astype(str)                
            .str.replace(",", "", regex=False) 
            .str.replace("%", "", regex=False) 
        )
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    if save_processed_csv:
        out_csv = csv_path.replace(".csv", "_processed.csv")
        df_clean.to_csv(out_csv, index=False)

    # Spatial join with state geometries
    g = us_states_gdf.merge(df_clean, on="STATEFP", how="left")

    for col in ["Before Flood", "During Flood", "After Flood"]:
        g[col] = g[col].fillna(0)

    # Normalize columns to fractions 
    if as_fraction:
        for col in ["Before Flood", "During Flood", "After Flood"]:
            total_val = g[col].sum()
            if total_val > 0:
                g[col] = g[col] / total_val

        vmin = 0
        vmax = max(g[["Before Flood", "During Flood", "After Flood"]].max())
        cbar_label = "Share of period total"
        from matplotlib.ticker import PercentFormatter
        formatter = PercentFormatter(1.0)
    else:
        vmin = g[["Before Flood", "During Flood", "After Flood"]].min().min()
        vmax = g[["Before Flood", "During Flood", "After Flood"]].max().max()
        cbar_label = "Total count"
        formatter = None  

    # Protect against a degenerate colorbar
    if vmax == vmin:
        vmax = vmin + 1e-9

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot 1×3 choropleth panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        ("Before Flood", "(a) Before"),
        ("During Flood", "(b) During"),
        ("After Flood",  "(c) After")
    ]

    for ax, (col, ttl) in zip(axes, panels):
        g.plot(
            column=col,
            cmap="viridis",
            norm=norm,
            edgecolor="white",
            linewidth=0.4,
            ax=ax
        )
        ax.set_title(ttl, fontsize=13)
        ax.axis("off")  

    # Create a single shared colorbar underneath all 3 panels
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm._A = [] 
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.035,
        pad=0.08
    )
    cbar.set_label(cbar_label, fontsize=12)
    if formatter:
        cbar.ax.xaxis.set_major_formatter(formatter)

    # Big figure title
    fig.suptitle(
        f"Fraction of Total Spending — {place_label} (Before / During / After 2022 Flood)",
        fontsize=16,
        fontweight="bold"
    )

    # Tight layout with some top space kept for suptitle
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300, bbox_inches="tight")

    # Return useful objects for further analysis 
    return {
        "gdf": g,     
        "fig": fig,  
        "axes": axes  
    }

# ### 6 Master Pipeline
def run_full_pipeline_for_place(
    place_name: str,
    pretty_label: str,
    geoid_prefixes: list[str],
    wy_cbg: gpd.GeoDataFrame,
    df_2022: pd.DataFrame,
    gdf_proj_states: gpd.GeoDataFrame,
    us_counties_file: str = US_COUNTIES_FILE,
    us_states_file: str = US_STATES_FILE,
    cbg_file: str = CBG_2019_FILE,
    exclude_geoids=None,
    outdir="outputs"
):
    """
    End-to-end workflow for an individual location, such as Cody, Wyoming.
    """

    os.makedirs(outdir, exist_ok=True)

    # Spatial footprint

    if isinstance(place_name, list):
        # if multiple places, get geodata for each and combine
        place_gdfs = [get_place_geodata(pn, wy_cbg.crs) for pn in place_name]
        place_gdf = pd.concat(place_gdfs, ignore_index=True)
        # Use first place name for distance calculations
        poi_name_str = place_name[0]
        distance_prefix = place_name[0].split(",")[0].lower().replace(" ", "_")
    else:
        place_gdf = get_place_geodata(place_name, wy_cbg.crs)
        poi_name_str = place_name
        distance_prefix = place_name.split(",")[0].lower().replace(" ", "_")
    
    cbg_subset = clip_cbgs_to_place(
        wy_cbg,
        place_gdf,
        exclude_geoids=exclude_geoids
    )


    # Filter the df_2022 dataset using specified prefixes to exclude unrelated AREA rows.
    df_sub = df_2022[
        df_2022["AREA"].str[:5].isin(geoid_prefixes)
        |
        df_2022["AREA"].str[:2].isin([p[:2] for p in geoid_prefixes])
        |
        df_2022["AREA"].str.startswith(tuple(geoid_prefixes))
    ]
    # Attach Advan mobility to spatial CBGs
    place_adv = subset_advan_for_place(df_sub, cbg_subset)

    # Time series (hourly/daily)
    hourly_series = build_hourly_series(place_adv)
    daily_series  = build_daily_series(place_adv)

    # OD map prep
    cbg_all = gpd.read_file(cbg_file)
    cbg_all["centroid"] = cbg_all.geometry.centroid
    centroid_lookup = cbg_all.set_index("GEOID")["centroid"].to_dict()

    od_map_june = make_od_map(
        place_adv,
        cbg_centroids=centroid_lookup,
        month=6,
        center_latlon=(
            place_gdf.geometry.iloc[0].centroid.y,
            place_gdf.geometry.iloc[0].centroid.x
        ),
        threshold_quantile=0.8
    )

    # Flow maps (county + state)
    counties = gpd.read_file(us_counties_file).to_crs(epsg=4326)
    keep_fp = [f"{i:02d}" for i in range(1,57)] + ['72']
    counties["STATEFP"] = counties["GEOID"].str[:2]
    counties = counties[counties["STATEFP"].isin(keep_fp)].copy()
    counties.geometry = counties.geometry.simplify(0.01, preserve_topology=True)
    counties["centroid"] = counties.geometry.centroid
    county_centroids = counties.set_index("GEOID")["centroid"].to_dict()

    us_states = gpd.read_file(us_states_file).to_crs(epsg=4326)
    drops = ["60","66","69","78"]
    us_states = us_states[~us_states["STATEFP"].isin(drops)].copy()
    us_states.geometry = us_states.geometry.simplify(0.01, preserve_topology=True)
    us_states["centroid"] = us_states.geometry.centroid
    state_centroids = us_states.set_index("STATEFP")["centroid"].to_dict()

    county_flows = explode_device_homes(place_adv, group_by="county")
    state_flows  = explode_device_homes(place_adv, group_by="state")

    county_flow_fig, _ = plot_flow_maps(
        flow_dict=county_flows,
        outline_gdf=counties,
        outline_label="county",
        dest_centroids=place_adv.set_index("AREA").geometry.centroid.to_dict(),
        origin_centroids=county_centroids,
        main_title=f"Movement Flows from Counties to {pretty_label}: May–July 2022"
    )

    state_flow_fig, _ = plot_flow_maps(
        flow_dict=state_flows,
        outline_gdf=us_states,
        outline_label="state",
        dest_centroids=place_adv.set_index("AREA").geometry.centroid.to_dict(),
        origin_centroids=state_centroids,
        main_title=f"Movement Flows from U.S. States to {pretty_label}: May–July 2022",
        log_scale=True
    )

    # Pie charts for major source regions
    group_map = {
        '16': 'ID',
        '08': 'CO',
        '46': 'SD',
        '56': 'WY',
        '30': 'MT'
    }
    pie_pivot = summarize_visits_by_state_group(place_adv, group_map)
    pie_fig, _ = plot_monthly_pies(
        pivot_df=pie_pivot,
        place_label=pretty_label
    )

    # State-level regression + distance decay
    vertical_summary = build_state_level_table(place_adv)

    merged_states = gdf_proj_states.merge(
        vertical_summary,
        on="GEOID",
        how="left"
    )
    merged_states = merged_states[merged_states["GEOID"] != "CA"].copy()

    # Handle Place_name as either string or list
    if isinstance (place_name, list):
        poi_name_str = place_name[0]
    else:
        poi_name_str = place_name

    merged_states = add_distance_to_poi(
        merged_states,
        poi_name=poi_name_str,
        distance_col_prefix=distance_prefix
    )

    stats_list, fits_dict = compute_regression_stats(merged_states)
    reg_fig, _ = plot_regression_panels(
        fits_dict,
        place_label=pretty_label
    )

    dist_col = [c for c in merged_states.columns if c.endswith("_dist_km")][0]
    scatter3d_fig, _ = plot_3d_panels(
        merged_states,
        place_label=pretty_label,
        distance_col=dist_col
    )

# (Optional) Save output artifacts to disk
# od_map_june.save(os.path.join(outdir, "od_map_june.html")# reg_fig.savefig(os.path.join(outdir, "regression_panels.png"), dpi=300)00)
# scatter3d_fig.savefig(os.path.join(outdir, "3d_panels.png"), dpi=300)  # etc.
    return {
        "place_adv": place_adv,
        "hourly_series": hourly_series,
        "daily_series": daily_series,
        "vertical_summary_states": vertical_summary,
        "merged_states_for_modeling": merged_states,
        "regression_stats": stats_list,
        "od_map_example": od_map_june,
        "figs": {
            "county_flow": county_flow_fig,
            "state_flow":  state_flow_fig,
            "pie":         pie_fig,
            "regression":  reg_fig,
            "scatter3d":   scatter3d_fig
        }
    }


__all__ = [
    "load_blockgroups",
    "load_state_pop",
    "load_advan",
    "get_place_geodata",
    "clip_cbgs_to_place",
    "get_drive_network",
    "get_drive_network_from_polygon",
    "build_custom_region",
    "add_distance_to_poi",
    "subset_advan_for_place",
    "build_hourly_series",
    "build_daily_series",
    "explode_device_homes",
    "summarize_visits_by_state_group",
    "build_state_level_table",
    "plot_hourly_series",
    "plot_daily_series",
    "make_od_map",
    "plot_flow_maps",
    "plot_monthly_pies",
    "prep_spending_choropleth",
    "plot_spending_choropleth_panels",
    "compute_regression_stats",
    "plot_regression_panels",
    "plot_3d_panels",
    "run_full_pipeline_for_place",
]



