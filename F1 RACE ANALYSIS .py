import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.font_manager import FontProperties 
import pandas as pd 
import numpy as np 
from collections import Counter

import fastf1
import os
from pathlib import Path

# Create a local cache directory relative to where the app runs
cache_dir = Path(os.getcwd()) / "fastf1_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache(str(cache_dir))

print(f"✅ FastF1 cache enabled at: {cache_dir}")

import os
import requests

# GitHub raw font URLs
F1_BOLD_URL = "https://github.com/SumanGouda/F1-Analysis-Project/raw/refs/heads/main/F1%20Font/Formula1-Bold_web_0.ttf"
F1_REGULAR_URL = "https://github.com/SumanGouda/F1-Analysis-Project/raw/refs/heads/main/F1%20Font/Formula1-Regular_web_0.ttf"

# Local paths (in your working folder)
F1_BOLD_PATH = "D:\DATA ANALYSIS PROJECTS\F1\F1 Font\Formula1-Bold_web_0.ttf"
F1_REGULAR_PATH = "D:\DATA ANALYSIS PROJECTS\F1\F1 Font\Formula1-Regular_web_0.ttf"

def download_font(url, path):
    """Download font if not already present."""
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download {url}")
    else:
        print(f"{path} already exists.")

# Download both fonts
download_font(F1_BOLD_URL, F1_BOLD_PATH)
download_font(F1_REGULAR_URL, F1_REGULAR_PATH)

st.set_page_config(layout="wide")

st.markdown("""
<style>
.element-container:has(.stPlotlyChart) {
    width: 100% !important;
    max-width: 1800px !important;
    margin: auto;
}
.stPlotlyChart iframe {
    width: 100% !important;
    height: 850px !important;
}
</style>
""", unsafe_allow_html=True)


class RaceDataAnalysis:
    def __init__(self, df, driver_1, driver_2, results_df, font, place, year):
        self.df = df
        self.driver_1 = driver_1.upper()
        self.driver_2 = driver_2.upper()
        self.results_df = results_df
        self.font = font
        self.place = place
        self.year = year

        # --- DRIVER 1 DATA ---
        try:
            self.drv1_data = self.df.pick_driver(self.driver_1).copy()
            self.drv1_data['Seconds'] = pd.to_timedelta(
                self.drv1_data['LapTime'], errors='coerce'
            ).dt.total_seconds()
        except Exception as e:
            print(f"⚠️ Error processing data for {self.driver_1}: {e}")
            self.drv1_data = pd.DataFrame({'Seconds': [np.nan]})

        # Telemetry for Driver 1
        try:
            self.drv1_tel_data = self.drv1_data.get_telemetry().add_distance()
        except Exception as e:
            print(f"⚠️ Telemetry error for {self.driver_1}: {e}")
            self.drv1_tel_data = pd.DataFrame()

        # Driver 1 color
        try:
            team_row = self.results_df[self.results_df['Abbreviation'] == self.driver_1]
            self.drv1_color = (
                "#" + str(team_row['TeamColor'].values[0]) if not team_row.empty else "#FF5733"
            )
        except Exception:
            self.drv1_color = "#FF5733"

        # --- DRIVER 2 DATA ---
        try:
            self.drv2_data = self.df.pick_driver(self.driver_2).copy()
            self.drv2_data['Seconds'] = pd.to_timedelta(
                self.drv2_data['LapTime'], errors='coerce'
            ).dt.total_seconds()
        except Exception as e:
            print(f"⚠️ Error processing data for {self.driver_2}: {e}")
            self.drv2_data = pd.DataFrame({'Seconds': [np.nan]})

        # Telemetry for Driver 2
        try:
            self.drv2_tel_data = self.drv2_data.get_telemetry().add_distance()
        except Exception as e:
            print(f"⚠️ Telemetry error for {self.driver_2}: {e}")
            self.drv2_tel_data = pd.DataFrame()

        # Driver 2 color
        try:
            team_row = self.results_df[self.results_df['Abbreviation'] == self.driver_2]
            self.drv2_color = (
                "#" + str(team_row['TeamColor'].values[0]) if not team_row.empty else "#3357FF"
            )
        except Exception:
            self.drv2_color = "#3357FF"

        # Handle duplicate colors
        if self.drv1_color == self.drv2_color:
            self.drv2_color = "#8D006A"

        # Lap time lists
        self.drv1_laptime = self.drv1_data.get('Seconds', pd.Series()).dropna().tolist()
        self.drv2_laptime = self.drv2_data.get('Seconds', pd.Series()).dropna().tolist()

        # --- POSITION TABLE PREPARATION ---
        try:
            pivot_df = self.df.pivot_table(index='Driver', columns='LapNumber', values='Position')
            req_cols = ['Abbreviation', 'TeamColor', 'DriverNumber', 'TeamName', 'GridPosition']
            req_col = self.results_df.reindex(columns=req_cols)
            self.new_df = (
                pd.merge(req_col, pivot_df.reset_index(), left_on='Abbreviation', right_on='Driver', how='left')
                .drop(columns=['Driver'])
                .sort_values(by='GridPosition', ascending=True)
            )
        except Exception as e:
            print(f"⚠️ Error creating position table: {e}")
            self.new_df = pd.DataFrame()

    def get_overall_track_positions(self):  
            # Create the figure
            fig = go.Figure()

            # Plot each driver's line
            for idx, row in self.new_df.iterrows():
                driver = row['Abbreviation']
                # Safely get team color (handle float values)
                try:
                    color_str = str(row['TeamColor'])
                    color_code = f"#{color_str}"
                except (KeyError, AttributeError):
                    color_code = "#FAF5F5"  # fallback color

                lap_data = row.iloc[4:].values
                laps = np.arange(1, len(lap_data) + 1)

                fig.add_trace(
                    go.Scatter(
                        x=laps,
                        y=lap_data,
                        mode='lines',
                        name=driver,
                        line=dict(color=color_code, width=4),
                        hovertemplate=f"<b>{driver}</b><br>Lap: %{{x}}<extra></extra>"
                    )
                )

            # Set axis limits and ticks
            x_max = self.new_df.shape[1] - 4    # Subtracting 4 for the first 4 columns whihc are not laps position columns 
            fig.update_xaxes(
                range=[-1, x_max + 2],
                tickvals=list(range(0, int(x_max), 2)),
                tickfont=dict(size=10, family='Formula1 Display Regular', color="white"),
                title_text="Lap Number",
                title_standoff=20,
                title_font=dict(size=18, family='Formula1 Display Regular', color="white"),
                color="white",
                showline=False,
                zeroline=False,
                showgrid=False
            )
            fig.update_yaxes(
                tickvals=list(range(1, 21)),
                ticktext=self.new_df['Abbreviation'].tolist(),
                tickfont=dict(size=10, family='Formula1 Display Regular', color="white"),
                title_text="Driver",
                title_standoff=20,
                title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
                color="white",
                autorange="reversed",
                automargin=True, 
                ticklabelposition="outside",
                showline=False,
                zeroline=False,
                showgrid=False
            )

            # Set background and grid
            fig.update_layout(
                width=1500,   # wider
                height=700,   # taller
                autosize=False,
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white", family='Formula1 Display Regular'),
                title=dict(
                    text="{place} {year}".format(year=self.year, place=self.place),
                    font=dict(size=25, family='Formula1 Display Regular', color="#e10600"),
                    x=0.5,
                    y=1,
                    xanchor='center',
                    yanchor='top',
                    pad=dict(t=15)
                ),
                showlegend=False,
                margin=dict(l=60, r=60, t=60, b=60),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.8)', gridwidth=1)
            )
            return fig

    def get_net_postition_gain_loss(self):
        fig = go.Figure()  

        for driver in self.new_df['Abbreviation'].tolist():
            driver_row = self.new_df[self.new_df['Abbreviation'] == driver]

            # Get positions - convert to scalar values using .item()
            start_pos = driver_row['GridPosition'].item()  # Gets single value
            last_col_value = driver_row.iloc[0,-1]         # Gets single value
            second_last_col = driver_row.iloc[0,-2]        # Gets single value

            # Choose end position ( If driver got lapped then choose the second last value)
            end_pos = last_col_value if not pd.isna(last_col_value) else second_last_col
            
            if pd.isna(end_pos):
                diff = start_pos - 20       # If the final value is NaN, subtract 20 instead
            else:
                diff = start_pos - end_pos

            try:
                color_str = str(results.loc[results['Abbreviation'] == driver, 'TeamColor'].values[0])
                color_code = f"#{color_str}"
            except (KeyError, AttributeError):
                color_code = "#FAF5F5"  # fallback color  
            
            fig.add_trace(
                go.Bar(
                    x=[driver],
                    y=[diff],
                    name=driver,
                    marker_color=color_code,
                    width = 0.6,
                    marker_line_width=0,
                    hovertemplate=f"<b>{driver}</b><br>Net Position Gain/Loss: {diff}<extra></extra>"
                )
            )
            fig.update_layout(
                xaxis=dict(
                    tickfont=dict(size=10, family='Formula1 Display Regular', color="white"),
                    title_text="Driver",
                    title_standoff=20,
                    title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
                    color="white",
                    showline=False,
                    zeroline=False,
                    showgrid=False
                ),
                yaxis=dict(
                    tickfont=dict(size=10, family='Formula1 Display Regular', color="white"),
                    title_text="Net Position Gain/Loss",
                    title_standoff=20,
                    title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
                    color="white",
                    showline=False,
                    showgrid=True, 
                    gridcolor='rgba(128,128,128,0.8)', 
                    gridwidth=1
                ),
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white", family='Formula1 Display Regular'),
                title=dict(
                    text="{place} {year}".format(year=self.year, place=self.place),
                    font=dict(size=18, family='Formula1 Display Regular', color="#e10600"),
                    x=0.5,
                    y=1,
                    xanchor='center',
                    yanchor='top',
                    pad=dict(t=15)
                ),
                showlegend=False,
                width=1500,
                height=800,
                margin=dict(l=30, r=30, t=40, b=30)
                
            )

        return fig
    
    def box_plot_analysis(self):
        fig = go.Figure()
        import matplotlib.colors as mcolors
        from collections import Counter

        # Prepare data
        drivers = self.new_df['Abbreviation'].tolist()
        drv_iqr_min, drv_iqr_max = [], []

        for driver in drivers:
            # Get driver laps data
            drv_data = self.df.pick_driver(driver)

            # ✅ Ensure "Seconds" column exists
            if 'Seconds' not in drv_data.columns:
                if 'LapTime' in drv_data.columns:
                    drv_data['Seconds'] = (
                        pd.to_timedelta(drv_data['LapTime'], errors='coerce')
                        .dt.total_seconds()
                    )
                else:
                    print(f"⚠️ Missing LapTime for {driver}")
                    drv_data['Seconds'] = np.nan

            # Color setup
            color = '#' + str(self.new_df.loc[self.new_df['Abbreviation'] == driver, 'TeamColor'].values[0])
            rgba = mcolors.to_rgba(color, alpha=0.2)
            rgba_str = f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})'

            # Add boxplot trace
            fig.add_trace(go.Box(
                y=drv_data['Seconds'].dropna(),
                name=driver,
                marker_color=color,
                fillcolor=rgba_str,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=0,
                showlegend=False
            ))

            # IQR computation
            clean_seconds = drv_data['Seconds'].dropna()
            if clean_seconds.empty:
                continue
            q1, q3 = clean_seconds.quantile([0.25, 0.75])
            iqr_min = q1 - 1.5 * (q3 - q1)
            iqr_max = q3 + 1.5 * (q3 - q1)
            drv_iqr_min.append(iqr_min)
            drv_iqr_max.append(iqr_max)

        # Y-axis range
        if drv_iqr_min and drv_iqr_max:
            y_min = int(min(drv_iqr_min)) - 0.6
            y_max = int(max(drv_iqr_max)) + 3
        else:
            y_min, y_max = 0, 200  # fallback

        fig.update_layout(
            yaxis=dict(
                range=[y_min, y_max],
                zeroline=False,
                showline=False,
                gridcolor='rgba(128,128,128,0.8)',
                gridwidth=1,
                title_text="Lap Time (Seconds)",
                title_standoff=20,
                title_font=dict(size=15, family='Formula1 Display Regular', color="white")
            ),
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", family='Formula1 Display Regular'),
            title=dict(
                text=f"F1 {self.place} GP {self.year}",
                font=dict(size=20, family='Formula1 Display Regular', color="red"),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                pad=dict(t=10)
            ),
            width=1500,
            height=800
        )
        return fig

    def __assign_and_clean_telemetry(self, data_df, telemetry_df):

        def filter_integer_meters(series):
            """
            Filters a pandas Series of distance measurements to keep only integer meter values,
            preserving both positive and negative values.
            
            Args:
                series (pd.Series): Series containing distance measurements
                
            Returns:
                pd.Series: Filtered series containing only integer meter values in increasing order
            """
            
            # Round to nearest integer (meter)
            telemetry_df['X_rounded'] = telemetry_df['X'].round()
            
            # For each integer value, keep the first occurrence that rounds to it
            # This preserves both positive and negative values
            filtered = (telemetry_df.groupby('X_rounded', as_index=False)
                        .first()
                        .sort_values('X'))
            
            # Return just the original X values in order
            return filtered['X'].reset_index(drop=True)
        
        data_df['LapStartTime'] = data_df['LapStartTime'].astype('timedelta64[ns]')
        data_df['LapTime'] = data_df['LapTime'].astype('timedelta64[ns]')
        data_df['LapEndTime'] = data_df['LapStartTime'] + data_df['LapTime']
        
        filtered_series = filter_integer_meters(telemetry_df['X'])
        filtered_tel_df = telemetry_df[telemetry_df['X'].isin(filtered_series)].copy()
        
        # Assign lap numbers to telemetry data
        lap_numbers = []
        for index, row in filtered_tel_df.iterrows():
            lap_found = False
            for _, lap in data_df.iterrows():
                if lap['LapStartTime'] <= row['SessionTime'] <= lap['LapEndTime']:
                    lap_numbers.append(lap['LapNumber'])
                    lap_found = True
                    break
            if not lap_found:
                lap_numbers.append(None)
        
        filtered_tel_df['LapNumber'] = lap_numbers
        
        # Select only the required columns
        selected_cols = ['LapNumber', 'X', 'Speed', 'nGear', 'Brake', 'RPM', 'Throttle']
        cleaned_tel_df = filtered_tel_df[selected_cols].sort_values(['LapNumber', 'X']).reset_index(drop=True)
        
        return cleaned_tel_df
    
    def get_lap_time_comparison(self):
        drv1_laptime = self.drv1_data['Seconds'].tolist()
        drv2_laptime = self.drv2_data['Seconds'].tolist()
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.drv1_data['LapNumber'].tolist(),
                y=drv1_laptime,
                mode='lines',
                name=self.driver_1,
                line=dict(color=self.drv1_color, width=2)
            )
        )
        import matplotlib.colors as mcolors
        color = mcolors.to_rgba(self.drv2_color)
        rgba = mcolors.to_rgba(color, alpha=0.2)  
        rgba_str = f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})'
        fig.add_trace(
            go.Scatter(
                x=self.drv2_data['LapNumber'].tolist(),
                y=drv2_laptime,
                mode='lines',
                name=self.driver_2,
                line=dict(color=self.drv2_color, width=2),
                fill='tonexty', 
                fillcolor=rgba_str  
            )
        )
        ymin = int(self.drv1_data['Seconds'].quantile(0.05)) - 1
        ymax = int(self.drv1_data['Seconds'].quantile(0.75)) + 2
        max_lap_no = int(max(self.drv1_data['LapNumber'].max(), self.drv2_data['LapNumber'].max()))
        
        fig.update_xaxes(
            title_text="Lap Number",
            range=[0, max_lap_no],
            zeroline=False,
            title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
            showgrid=False,
            tickvals = list(range(1, max_lap_no + 1, 2)),  
            tickfont=dict(size=10, family='Formula1 Display Regular', color="white")
            )
        fig.update_yaxes(
            range=[ymin, ymax],
            tickvals=list(range(ymin + 1, ymax, 1)),
            tickfont=dict(size=10, family='Formula1 Display Regular', color="white"),            
            dtick=1 ,
            title_text="Lap Time (s)",
            title_font=dict(size=10, family='Formula1 Display Regular', color="white"),
            gridcolor='rgba(128,128,128,0.4)', 
            gridwidth=1
            )
        fig.update_layout(
            title=dict(
                    text=f"{self.place} {self.year}, {self.driver_1} vs {self.driver_2} Comparison of Lap Times",
                    font=dict(size=25, family='Formula1 Display Regular', color="red"),
                    x=0.5,
                    y=1,
                    xanchor='center',
                    yanchor='top',
                    pad=dict(t=15)
                ),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color="white"),
            width = 2000,
            height = 800
        )
        return fig

    # This one will take a long time
    def get_average_lap_speed_per_lap(self):
                
        drv1_tel_data = self.__assign_and_clean_telemetry(self.drv1_data, self.drv1_tel_data)
        drv2_tel_data = self.__assign_and_clean_telemetry(self.drv2_data, self.drv2_tel_data)
        
        drv1_avg_speed = drv1_tel_data.groupby('LapNumber')['Speed'].mean().reset_index()
        drv2_avg_speed = drv2_tel_data.groupby('LapNumber')['Speed'].mean().reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # First create the merged DataFrame
        merged = pd.merge(
            drv1_avg_speed[['LapNumber', 'Speed']].rename(columns={'Speed': 'Speed_1'}),
            drv2_avg_speed[['LapNumber', 'Speed']].rename(columns={'Speed': 'Speed_2'}),
            on='LapNumber',
            how='outer'  # Keep all laps from both drivers
        ).sort_values('LapNumber')

        # Forward fill missing values (optional - comment out if you want gaps)
        merged[['Speed_1', 'Speed_2']] = merged[['Speed_1', 'Speed_2']].ffill()

        # Now create the plot with properly aligned data
        fig.add_trace(go.Scatter(
            x=merged['LapNumber'],
            y=merged['Speed_1'],
            mode='lines',
            line=dict(color=self.drv1_color, width=3),
            name=self.driver_1,
            hovertemplate=f"{self.driver_1}<br>Speed: %{{y:.1f}} km/h<extra></extra>",
            hoverlabel=dict(
                bgcolor='white',
                font_size=5,
                font_family='Formula1 Display Regular',
                font_color=self.drv1_color
            )
        ))

        fig.add_trace(go.Scatter(
            x=merged['LapNumber'],
            y=merged['Speed_2'],
            mode='lines',
            line=dict(color=self.drv2_color, width=3),
            name=self.driver_2,
            fill='tonexty',
            fillcolor='rgba(165, 221, 255, 0.5)',
            hovertemplate=f"{self.driver_2}<br>Speed: %{{y:.1f}} km/h<extra></extra>",
            hoverlabel=dict(
                bgcolor='white',
                font_size=5,
                font_family='Formula1 Display Regular',
                font_color=self.drv2_color
            )
        ))
        
        
        # Update layout to match matplotlib style
        
        X_max = max(drv1_avg_speed['LapNumber'].max(), drv2_avg_speed['LapNumber'].max())
        
        y_max = max(
            drv1_avg_speed['Speed'].max() if not drv1_avg_speed.empty else 0,
            drv2_avg_speed['Speed'].max() if not drv2_avg_speed.empty else 0
        ) * 1.05  # Add 5% padding

        # Get min speed with 5% lower padding (but not below 0)
        y_min = max(
            min(
                drv1_avg_speed['Speed'].min() if not drv1_avg_speed.empty else float('inf'),
                drv2_avg_speed['Speed'].min() if not drv2_avg_speed.empty else float('inf')
            ) * 0.95,  # Add 5% padding
            0  # Don't go below 0 for speed values
        )

        fig.update_layout(
            width=1400,  # Adjust width in pixels
            height=600,
            title=dict(
                    text=f"{self.place} {self.year}, {self.driver_1} vs {self.driver_2} Average Lap Speed",
                    font=dict(size=25, family='Formula1 Display Regular', color="red"),
                    x=0.5,
                    y=1,
                    xanchor='center',
                    yanchor='top',
                    pad=dict(t=15)
                ),
            plot_bgcolor='black',  
            paper_bgcolor='black',
            xaxis=dict(
                title='Lap Number',
                title_font=dict(
                    family='Formula1 Display Regular',
                    size=18,
                    color='white'
                ),
                tickfont=dict(
                    family='Formula1 Display Regular',
                    size=10,
                    color='white'
                ),
                range=[0, X_max + 1],
                dtick=2,
                showgrid=False,
                linecolor='black'            ),
            yaxis=dict(
                title='Average Speed (kM/H)',
                title_font=dict(
                    family='Formula1 Display Regular',
                    size=20,
                    color='white'
                ),
                tickfont=dict(
                    family='Formula1 Display Regular',
                    size=10,
                    color='white'
                ),
                range=[y_min - 10, y_max + 10],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.2)',
                linecolor='black'
            ),
            legend=dict(
                font=dict(
                    family='Formula1 Display Regular',
                    size=16
                ),
                x=1,
                y=1,
                xanchor='left',
                yanchor='top'
            ),
            margin=dict(l=100, r=100, t=100, b=100),
            hovermode='x unified'
        )
        
        # Remove all borders
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)
        
        return fig
    
    def overall_laptime_dominance(self):
        # --- Step 1: pick only needed columns and require numeric seconds ---
        drv1 = self.drv1_data[['LapNumber', 'Seconds', 'Compound']].copy()
        drv2 = self.drv2_data[['LapNumber', 'Seconds', 'Compound']].copy()

        drv1['Seconds'] = drv1['Seconds'].fillna(drv1['Seconds'].median())
        drv2['Seconds'] = drv2['Seconds'].fillna(drv2['Seconds'].median())

        # --- Step 2: merge on LapNumber so we only compare laps both drivers completed ---
        merged = pd.merge(
            drv1,
            drv2,
            on='LapNumber',
            how='inner',
            suffixes=('_drv1', '_drv2')
        )
        
        total_compared = len(merged)
        if total_compared == 0:
            # No comparable laps -> return an empty figure with a message
            fig = go.Figure()
            fig.update_layout(
                title="No comparable laps (0)",
                plot_bgcolor='black',
                paper_bgcolor='black',
                width=600,
                height=600,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(
                    text="No comparable laps<br>(0)",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=18, color='white'),
                    align='center'
                )]
            )
            return fig

        # --- Step 3: who was faster on each compared lap ---
        drv1_faster = []
        drv2_faster = []
        for index, row in merged.iterrows():
            if row['Seconds_drv1'] < row['Seconds_drv2']:
                drv1_faster.append(index)
            elif row['Seconds_drv2'] < row['Seconds_drv1']:
                drv2_faster.append(index)
            else:
                drv1_faster.append(index)
                drv2_faster.append(index)
                
        drv1_comp = {"SOFT": 0, "MEDIUM": 0, "HARD": 0, "WET": 0, "INTERMEDIATE": 0}
        drv2_comp = {"SOFT": 0, "MEDIUM": 0, "HARD": 0, "WET": 0, "INTERMEDIATE": 0}

        for lap in set(drv1_faster) | set(drv2_faster):
            if lap in drv1_faster:
                compound = merged.loc[lap, 'Compound_drv1']
                drv1_comp[compound] += 1
            if lap in drv2_faster:
                compound = merged.loc[lap, 'Compound_drv2']
                drv2_comp[compound] += 1
        
        # Compound colors
        compound_colors = {
            'SOFT': '#FF3333',
            'MEDIUM': '#FFFF00',
            'HARD': '#FFFFFF',
            'INTERMEDIATE': '#00FF00',
            'WET': '#0000FF'
        }
        
        def color_for(comp):
            return compound_colors.get(comp, "#EA05D3")

        # Build inner ring values and colors
        inner_values = []
        inner_labels = []
        inner_colors = []
        
        # Driver 1 compounds
        for c in compound_colors.keys():
            v = drv1_comp.get(c, 0)
            if v > 0:
                inner_values.append(int(v))
                inner_labels.append(f"{c}: {v}")
                inner_colors.append(color_for(c))
        
        # Driver 2 compounds
        for c in compound_colors.keys():
            v = drv2_comp.get(c, 0)
            if v > 0:
                inner_values.append(int(v))
                inner_labels.append(f"{c}: {v}")
                inner_colors.append(color_for(c))

        # --- Step 5: Create Plotly nested donut chart ---
        fig = go.Figure()
        
        # Outer ring
        outer_vals = [len(drv1_faster), len(drv2_faster)]
        total = sum(outer_vals)
        
        # Add outer ring
        fig.add_trace(go.Pie(
            values=outer_vals,
            labels=[self.driver_1, self.driver_2],
            hole=0.6,
            domain=dict(x=[0, 1], y=[0, 1]),
            marker=dict(colors=[self.drv1_color, self.drv2_color], line=dict(color='white', width=1.5)),
            textinfo='label+value+percent',
            texttemplate=(
                "<b>%{label}</b><br>" +
                "%{value} laps<br>" +
                "(%{percent})"
            ),
            hoverinfo='label+value+percent',
            name="Outer",
            textfont=dict(size=14)  
        ))
        
        # Inner ring (only if we have inner data)
        if len(inner_values) > 0:
            # Use abbreviations for compound names
            compound_abbr = {
                'SOFT': 'S',
                'MEDIUM': 'M', 
                'HARD': 'H',
                'INTERMEDIATE': 'I',
                'WET': 'W'
            }
            
            inner_labels_abbr = []
            for label in inner_labels:
                for full_name, abbr in compound_abbr.items():
                    if full_name in label:
                        inner_labels_abbr.append(label.replace(full_name, abbr))
                        break
            
            fig.add_trace(go.Pie(
                values=inner_values,
                labels=inner_labels_abbr,
                hole=0.4,
                domain=dict(x=[0.15, 0.85], y=[0.15, 0.85]),  # Smaller domain for inner ring
                marker=dict(colors=inner_colors, line=dict(color='white', width=1.5)),
                textinfo='label',
                hoverinfo='label+value',
                textposition='inside',
                textfont=dict(size=12, color='black'),
                name="Inner"
            ))

        # Update layout
        fig.update_layout(
            title={
                'text': f"{self.place} {self.year}, {self.driver_1} vs {self.driver_2} Number Of Laps Dominance",
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=16, color='#e10600', family='Formula1 Display Regular')
            },
            showlegend=False,
            plot_bgcolor='black',  
            paper_bgcolor='black',
            width=700,
            height=700,
            annotations=[dict(
                text=f"{total_compared}<br>laps",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14, color='white', family='Formula1 Display Regular'),
                align='center'
            )]
        )

        return fig
    
    def lap_telemtry_comparision(self, lapnumber, data):
        # Prepare telemetry data with cleaned and assigned lap numbers
        try:
            drv1_lap_data = self.drv1_data[self.drv1_data['LapNumber'] == lapnumber]
            drv2_lap_data = self.drv2_data[self.drv2_data['LapNumber'] == lapnumber]
            
            if len(drv1_lap_data) == 0 or len(drv2_lap_data) == 0:
                raise ValueError(f"Lap {lapnumber} not found for one or both drivers")
                
        except KeyError:
            raise KeyError(f"{lapnumber} missing in data")  # Stop if column doesn't exist
                    
        drv1_tel_data = drv1_lap_data.get_telemetry()
        drv2_tel_data = drv2_lap_data.get_telemetry()

        # Create figure
        fig = go.Figure()

        # Add traces for both drivers
        fig.add_trace(go.Scatter(
            x=drv1_tel_data['Distance'],
            y=drv1_tel_data[data],
            mode='lines',
            name=self.driver_1,
            line=dict(color=self.drv1_color, width=3)
        ))

        fig.add_trace(go.Scatter(
            x=drv2_tel_data['Distance'],
            y=drv2_tel_data[data],
            mode='lines',
            name=self.driver_2,
            line=dict(color=self.drv2_color, width=3)
        ))

        # Update layout with styling
        fig.update_layout(
            width=2200,
            height=800,
            title={
                'text': f"{self.place} {self.year}, {data} Comparision for lapnumber {lapnumber}",
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Anchor to center
                'font': dict(size=15, family='Formula1 Display Regular', color="#e10600")  # F1 red
            },
            plot_bgcolor='black',  
            paper_bgcolor='black',
            font=dict(size=10, family='Formula1 Display Regular', color="white"),
            
            # Legend
            legend=dict(
                font=dict(family='Formula1 Display Regular', size=10),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='black',
                bordercolor='rgba(255,255,255,0.5)'
            ),
            
            # Margins
            margin=dict(l=100, r=100, t=150, b=100)
        )
        
        fig.update_yaxes(
            title_text=f'{data}',
            title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
            tickfont=dict(size=10, family='Formula1 Display Regular', color="white"), 
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=0.5,
            minor=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.3)',
                gridwidth=0.5,
            ),
            ticks='outside'
        )

        fig.update_xaxes( 
            range=[-100, int(max(drv1_tel_data['Distance'].max(), drv2_tel_data['Distance'].max())) + 100], 
            title_text="Distance",
            title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
            tickfont=dict(size=10, family='Formula1 Display Regular', color="white"),
            ticks='outside',
            dtick=500,
            zeroline=False, 
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=0.5,
            minor_gridcolor='rgba(128, 128, 128, 0.3)',
            minor_griddash='dot'
        )

        return fig
    
    def track_dominance(self, lapnumber):
        try:
            # Prepare telemetry data with cleaned and assigned lap numbers
            drv1_lap_data = self.drv1_data[self.drv1_data['LapNumber'] == lapnumber]
            drv2_lap_data = self.drv2_data[self.drv2_data['LapNumber'] == lapnumber]
            
            if len(drv1_lap_data) == 0 or len(drv2_lap_data) == 0:
                st.warning(f"Lap {lapnumber} not found for one or both drivers")
                return None
                
        except KeyError:
            st.error("'LapNumber' column missing in data")
            return None
                    
        try:
            drv1_tel_data = drv1_lap_data.get_telemetry()
            drv2_tel_data = drv2_lap_data.get_telemetry()      
            
            min_len = min(len(drv1_tel_data), len(drv2_tel_data))
            if min_len == 0:
                st.warning("No telemetry data available for selected lap")
                return None
                
            x = drv1_tel_data['Y'].values[:min_len]
            y = drv1_tel_data['X'].values[:min_len]
            
            drv1_speed = drv1_tel_data['Speed'].values[:min_len]
            drv2_speed = drv2_tel_data['Speed'].values[:min_len]
            
            colors = [self.drv1_color if d1 > d2 else self.drv2_color for d1, d2 in zip(drv1_speed, drv2_speed)]
            
            # Create plotly figure instead of matplotlib
            import plotly.graph_objects as go
            
            # Create the trace with colored segments
            fig = go.Figure()
            
            for i in range(len(x)-1):
                # Determine driver name based on color
                driver_name = self.driver_1 if colors[i] == self.drv1_color else self.driver_2
                
                fig.add_trace(go.Scatter(
                    x=[x[i], x[i+1]],
                    y=[y[i], y[i+1]],
                    mode='lines',
                    line=dict(color=colors[i], width=4),
                    showlegend=False,
                    hovertemplate=f"<span style='font-size: 16px; font-weight: bold;'>{driver_name}</span><extra></extra>"
                ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f"{self.year} {self.place} Performance trace comparison of lap {lapnumber}",
                    'x': 0.5,  # Center the title horizontally
                    'xanchor': 'center',  # Anchor point for horizontal positioning
                    'yanchor': 'top',  # Anchor point for vertical positioning
                    'font': {
                        'family': "Formula1 Display Regular",
                        'size': 16,
                        'color': "#e10600"  
                    }
                },
                plot_bgcolor='black',  
                paper_bgcolor='black',
                width=600,
                height=600,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating track dominance plot: {str(e)}")
            return None
    
    def get_pit_stop_analysis(self):
        
        
        all_time_deltas = []  # To store both driver's deltas
        max_time = 0

        # Collect time deltas for each driver
        for drv_data in [self.drv1_data, self.drv2_data]:
            pit_out_lap = drv_data[(drv_data['PitOutTime'].notna()) & (drv_data['TyreLife'] == 1)]['LapNumber'].tolist()
            time_delta = []

            for lap in pit_out_lap:
                pit_out_lap_time = drv_data[drv_data['LapNumber'] == lap]['Seconds'].values[0]
                pit_in_lap_time = drv_data[drv_data['LapNumber'] == lap - 1]['Seconds'].values[0]

                green_flag_race_lap = drv_data[(drv_data['TrackStatus'].astype(int) == 1) & (drv_data['PitInTime'].isna()) & (drv_data['PitOutTime'].isna())]['LapNumber'].tolist()

                # Calculate difference from pit lap (can be negative or positive)
                lap_diffs = [(laps, lap - laps) for laps in green_flag_race_lap]

                # Separate laps before and after
                laps_before = [(laps, diff) for laps, diff in lap_diffs if diff > 0]
                laps_after = [(laps, diff) for laps, diff in lap_diffs if diff < 0]

                # Sort both by proximity to pit lap
                laps_before_sorted = sorted(laps_before, key=lambda x: x[1])  # smallest positive difference
                laps_after_sorted = sorted(laps_after, key=lambda x: abs(x[1]))  # smallest positive outlap

                # Pick up to 4 laps before; if fewer, fill with after laps
                close_5_laps = [laps for laps, _ in laps_before_sorted[:4]]
                if len(close_5_laps) < 5:
                    needed = 5 - len(close_5_laps)
                    close_5_laps += [laps for laps, _ in laps_after_sorted[:needed]]

                avg_lap_time = drv_data[drv_data['LapNumber'].isin(close_5_laps)]['Seconds'].mean()

                time_lost_by_pit = (pit_out_lap_time + pit_in_lap_time) - (avg_lap_time * 2)
                time_delta.append(time_lost_by_pit)

            all_time_deltas.append(time_delta)
            max_time = max(max_time, max(time_delta))
        
        # Assume equal number of pit stops for both drivers (or pad if needed)
        num_pits = max(len(all_time_deltas[0]), len(all_time_deltas[1]))
        xaxis = np.linspace(1, num_pits+1, num_pits)
          
        fig = go.Figure()
        
        bar_width = 0.8
        # Plot Graph 
        fig.add_trace(go.Bar(
            x=[xi - bar_width / 2 for xi in xaxis],
            y=all_time_deltas[0],
            name=self.driver_1,
            marker=dict(color=self.drv1_color, line=dict(width=0)),
            hovertext = [f"Pit: {i+1}, Time: {y:.2f} sec" for i, y in enumerate(all_time_deltas[0])],
            hoverinfo='text'
            ))
        fig.add_trace(go.Bar(
            x=[xi + bar_width / 2 for xi in xaxis],
            y=all_time_deltas[1],
            name=self.driver_2,
            marker=dict(color=self.drv2_color, line=dict(width=0)),
            hovertext = [f"Pit: {i+1}, Time: {y:.2f} sec" for i, y in enumerate(all_time_deltas[1])],
            hoverinfo='text'
            ))
        
        fig.update_layout(
            width=1200,
            height=500,
            title={
                'text': "Pit Time Delta",
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Anchor to center
                'font': dict(size=18, family='Formula1 Display Regular', color="#e10600")  # F1 red
            },
            plot_bgcolor='black',  
            paper_bgcolor='black',
            font=dict(size=10, family='Formula1 Display Regular', color="white"),
            yaxis_title='Time in seconds',
            margin=dict(l=80, r=40, t=80, b=60)
        )

        # Update Y-axis (grid alpha = 0.3)
        fig.update_yaxes(
            gridcolor='rgba(255, 255, 255, 0.3)',  # Light white grid with alpha
            gridwidth=0.5,
            tickfont=dict(color='white')
            )

        # Update X-axis
        fig.update_xaxes(
            range=[0, num_pits + 2],  # Set x-axis limits
            showticklabels=False,  # Turn off x-axis tick labels
            ticks='',  # Hide ticks completely
            showgrid=False  # Turn off x-axis grid lines
            )

        return fig

    def Lap_time_analysis(self):
        if self.drv1_data is None or self.drv2_data is None:
            print("Driver data not available for analysis.")
            return None
        
        else : 
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=self.drv1_data['LapNumber'].tolist(),
                    y=self.drv1_data['Seconds'].tolist(),
                    name=self.driver_1,
                    width=0.3,
                    offset=-0.15,
                    marker=dict(color=self.drv1_color, line=dict(width=0, color=None)),
                    hovertext = [f"Lap: {i+1}, Time: {y:.2f} sec" for i, y in enumerate(self.drv1_data['Seconds'].tolist())],
                    hoverinfo='text'
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=self.drv2_data['LapNumber'].tolist(),
                    y=self.drv2_data['Seconds'].tolist(),
                    name=self.driver_2,
                    width=0.3,
                    offset=0.15,
                    marker=dict(color=self.drv2_color, line=dict(width=0, color=None)),
                    hovertext = [f"Lap: {i+1}, Time: {y:.2f} sec" for i, y in enumerate(self.drv2_data['Seconds'].tolist())],
                    hoverinfo='text'
                )
            )
            
            fig.update_layout(
                width=1500,
                height=500,
                title_text="Lap Time Analysis",
                title_font=dict(size=15, family='Formula1 Display Regular', color="red"),
                plot_bgcolor='black',
                paper_bgcolor='black',
                bargap=0.2,
                barmode='group',
                font=dict(size=10, family='Formula1 Display Regular', color="white"),
                yaxis_title='Time in seconds',  # Y-axis label
                #margin=dict(l=80, r=40, t=80, b=60)  # Optional: adjust margins
            )
            
            # Update Y-axis (grid alpha = 0.3)
            fig.update_yaxes(
                gridcolor='rgba(255, 255, 255, 0.3)',  # Light white grid with alpha
                gridwidth=0.5,
                tickfont=dict(color='white')
            )
            
            # Update X-axis
            fig.update_xaxes(
                range=[0, self.drv1_data['LapNumber'].max() + 2],  # Set x-axis limits
                showticklabels=False,  # Turn off x-axis tick labels
                ticks='',  # Hide ticks completely
                showgrid=False,  # Turn off x-axis grid lines
            )
            
            return fig
                
class Practice_session_analysis:
    def __init__(self, session, results_df, font):
        self.session = session
        self.results_df = results_df
        self.font = font
        
    def bar_plot_analysis(self, drv1, drv2):
        drv1_data = self.session.pick_driver(drv1).sort_values(by=['LapNumber'])
        drv2_data = self.session.pick_driver(drv2).sort_values(by=['LapNumber'])
        
        # Get driver 1 team color safely
        if not self.results_df[self.results_df['Abbreviation'] == drv1].empty:
            self.drv1_color = "#" + str(
                self.results_df[self.results_df['Abbreviation'] == drv1]['TeamColor'].values[0]
            )
        else:
            self.drv1_color = "#FF5733"  # fallback color
        
        # Get driver 2 team color safely
        if not self.results_df[self.results_df['Abbreviation'] == drv2].empty:
            self.drv2_color = "#" + str(
                self.results_df[self.results_df['Abbreviation'] == drv2]['TeamColor'].values[0]
            )
        else:
            self.drv2_color = "#FF5733"  # fallback color
        
        merged_df = pd.concat([drv1_data, drv2_data], ignore_index=True)
        
        if merged_df.empty:
            fig = None
        else:
            hot_lap = merged_df[(merged_df['PitInTime'].isna()) & (merged_df['PitOutTime'].isna())]
            hot_lap['Seconds'] = pd.to_timedelta(hot_lap['LapTime']).dt.total_seconds()
            
            fig = go.Figure()

            for driver, color in [(drv1, self.drv1_color), (drv2, self.drv2_color)]:
                driver_df = hot_lap[hot_lap['Driver'] == driver]

                fig.add_trace(
                    go.Bar(
                        x=driver_df['LapNumber'],
                        y=driver_df['Seconds'],   # Lap times in seconds
                        name=driver,
                        width=0.3,
                        marker=dict(color=color, line=dict(width=0, color=None)),
                        hovertext=[
                            f"Lap: {lap}, Time: {time:.2f}s"
                            for lap, time, pos in zip(driver_df['LapNumber'], driver_df['Seconds'], driver_df['Position'])
                        ],
                        hoverinfo='text'
                    )
                )

            fig.update_layout(
                barmode='group',
                title=dict(
                    text=f"Lap Time Comparison per Hot-Lap",
                    font=dict(size=20, family='Formula1 Display Regular', color="red"),
                    x=0.5,   # center align
                    y=0.95,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis=dict(
                    title="Lap Number",
                    title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
                    tickfont=dict(size=12, family='Formula1 Display Regular', color="white"),
                    gridcolor='rgba(128,128,128,0.3)',  # reduced alpha
                    gridwidth=1
                ),
                yaxis=dict(
                    title="Lap Time (s)",
                    title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
                    tickfont=dict(size=12, family='Formula1 Display Regular', color="white"),
                    gridcolor='rgba(128,128,128,0.3)',  # reduced alpha
                    gridwidth=1,
                    range = [60, None]
                ),
                plot_bgcolor='black',  
                paper_bgcolor='black',
                font=dict(color="white", family='Formula1 Display Regular'),
                width=2000,
                height=500
            )

            return fig
        
    def quickest_lap(self, driver, data):
        
        driver_fastest_lap = (
            self.session.pick_drivers(driver)
            .pick_fastest()
            .get_telemetry()
            .add_distance()
        )
        
        if not self.results_df[self.results_df['Abbreviation'] == driver].empty:
            drv_color = "#" + str(
                self.results_df[self.results_df['Abbreviation'] == driver]['TeamColor'].values[0]
            )
        else:
            drv_color = "#FF5733"  # fallback color

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=driver_fastest_lap['Distance'],
                y=driver_fastest_lap[data],
                mode='lines',
                name=driver,
                line=dict(color=drv_color, width=3),
                hovertemplate=(
                    f"<b>{driver}</b><br>"
                    "Distance: %{x:.2f} meters<br>"
                    f"{str(data).capitalize()}: %{{y:.2f}}<extra></extra>"  # In single brackets for {y} it is showing error, i do not why!
                )
            )
        )

        fig.update_layout(
            title={
                'text': f"{driver} Quickest Lap Analysis",
                'font': dict(size=20, family='Formula1 Display Regular', color="red"),
                'x': 0.5,
                'y': 0.9,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title="Distance",
                title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
                tickfont=dict(size=10, family='Formula1 Display Regular', color="white"),
                gridcolor='rgba(128,128,128,0.3)',
                gridwidth=1,      
            ),
            yaxis=dict(
                title=str(data).capitalize(),
                title_font=dict(size=15, family='Formula1 Display Regular', color="white"),
                tickfont=dict(size=10, family='Formula1 Display Regular', color="white"),
                gridcolor='rgba(128,128,128,0.3)',
                gridwidth=1,
                showline=True,   # Keep y-axis line if desired
                linewidth=2,
                zeroline=False
            ),
            plot_bgcolor='black',  
            paper_bgcolor='black',
            font=dict(color="white", family='Formula1 Display Regular'),
            showlegend=False,
            width=1500,
            height=600
        )

        return fig

    def track_dominance(self, drv1, drv2, lapnumber=None):
        # --- 1. Get telemetry data ---
        drv1_tel_data = self.session.pick_drivers(drv1).get_telemetry()
        drv2_tel_data = self.session.pick_drivers(drv2).get_telemetry()

        # --- 2. Safe team color extraction ---
        if not self.results_df[self.results_df['Abbreviation'] == drv1].empty:
            drv1_color = "#" + str(
                self.results_df[self.results_df['Abbreviation'] == drv1]['TeamColor'].values[0]
            )
        else:
            drv1_color = "#FF5733"

        if not self.results_df[self.results_df['Abbreviation'] == drv2].empty:
            drv2_color = "#" + str(
                self.results_df[self.results_df['Abbreviation'] == drv2]['TeamColor'].values[0]
            )
            if drv2_color == drv1_color:
                drv2_color = "#EE0ABC"
        else:
            drv2_color = "#3357FF"

        # --- 3. Synchronize telemetry ---
        min_len = min(len(drv1_tel_data), len(drv2_tel_data))
        x = drv1_tel_data['Y'].values[:min_len]  # Swapping X and Y for correct track layout
        y = drv1_tel_data['X'].values[:min_len]

        drv1_speed = drv1_tel_data['Speed'].values[:min_len]
        drv2_speed = drv2_tel_data['Speed'].values[:min_len]

        # --- 4. Determine segment colors ---
        colors = [drv1_color if d1 > d2 else drv2_color for d1, d2 in zip(drv1_speed, drv2_speed)]

        # --- 5. Build Plotly figure ---
        fig = go.Figure()

        # Draw colored line segments between consecutive points
        for i in range(len(x) - 1):
            faster_driver = drv1 if drv1_speed[i] > drv2_speed[i] else drv2
            fig.add_trace(go.Scatter(
                x=[x[i], x[i+1]],
                y=[y[i], y[i+1]],
                mode='lines',
                line=dict(color=colors[i], width=4),
                hovertemplate=(
                    f"<b>Segment {i+1}</b><br>"
                    f"Faster: <b>{faster_driver}</b><extra></extra>"
                ),
                showlegend=False
            ))

        # --- 6. Configure layout ---
        title_text = (
            f"<b>Performance trace comparison: {drv1} vs {drv2}</b>"
            if lapnumber is None
            else f"<b> Lap {lapnumber} | {drv1} vs {drv2}</b>"
        )

        fig.update_layout(
            title={
                'text': title_text,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'family': "Formula1 Display Regular",
                    'size': 16,
                    'color': "#e10600"
                }
            },
            plot_bgcolor='black',
            paper_bgcolor='black',
            width=600,
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1)
        )

        fig.show()
        return fig

        
st.sidebar.title("About")

# ---------------------------------------------------------
# 🧠 SESSION MAPPING
# ---------------------------------------------------------
SESSION_MAPPING = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Race": "R",
    "Sprint": "S",
    "Sprint Shootout": "SS",
    "Sprint Qualifying": "SS"
}

from matplotlib.font_manager import FontProperties

# ---------------------------------------------------------
# 🧩 0. Enable FastF1 Cache (outside cached functions)
# ---------------------------------------------------------

os.makedirs("./cache", exist_ok=True)
fastf1.Cache.enable_cache("./cache")


# ---------------------------------------------------------
# 🧩 1. Font Loading with Cache
# ---------------------------------------------------------
@st.cache_resource
def load_f1_fonts(bold_path, regular_path):
    try:
        f1_bold = FontProperties(fname=bold_path)
        f1_regular = FontProperties(fname=regular_path)
        return f1_bold, f1_regular, None
    except Exception as e:
        return FontProperties(weight='bold'), FontProperties(), str(e)

f1_bold, f1_regular, font_err = load_f1_fonts("fonts/F1Bold.ttf", "fonts/F1Regular.ttf")
if font_err:
    st.warning(f"❌ Font load failed, using defaults. Error: {font_err}")
else:
    st.toast("✅ F1 fonts loaded successfully.")

# ---------------------------------------------------------
# 🧩 2. FastF1 Session Loader (Cached)
# ---------------------------------------------------------
@st.cache_data(show_spinner="📦 Loading F1 session data...")
def load_fastf1_session(year, event_name, session_code):
    try:
        session = fastf1.get_session(year, event_name, session_code)
        session.load()
        laps = session.laps
        results = session.results
        drivers = sorted(session.drivers)
        return laps, results, drivers
    except Exception as e:
        raise RuntimeError(f"FastF1 failed to load session: {e}")

# ---------------------------------------------------------
# 🧩 3. Sidebar Selections
# ---------------------------------------------------------
st.sidebar.header("🏁 Session Selection")

year_options = [""] + list(reversed(range(2018, 2026)))
year = st.sidebar.selectbox("Year", year_options, index=0, format_func=lambda x: "Select Year" if x == "" else str(x))

if year:
    try:
        schedule = fastf1.get_event_schedule(year)
        event_opt = [""] + schedule['EventName'].tolist()
    except Exception as e:
        st.error(f"❌ Failed to load schedule for {year}: {e}")
        st.stop()

    event_name = st.sidebar.selectbox("Event", event_opt, index=0, format_func=lambda x: "Select Event" if x == "" else x)

    if event_name:
        event_data = schedule[schedule['EventName'] == event_name].iloc[0]
        available_sessions = [
            s for s in event_data[['Session1', 'Session2', 'Session3', 'Session4', 'Session5']]
            if pd.notna(s)
        ]
        available_sessions = [""] + available_sessions

        session_type = st.sidebar.selectbox(
            "Session", available_sessions, index=0, format_func=lambda x: "Select Session" if x == "" else x
        )

        SESSION_MAPPING = {
            "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
            "Qualifying": "Q", "Sprint": "S", "Race": "R"
        }

        if session_type:
            session_code = SESSION_MAPPING.get(session_type)
            if not session_code:
                st.error(f"Unsupported session type: {session_type}")
                st.stop()

            with st.spinner(f"🏎️ Loading {event_name} {session_type} data..."):
                try:
                    df, results, driver_list = load_fastf1_session(year, event_name, session_code)
                    st.success(f"✅ Loaded {event_name} {session_type}")
                except Exception as e:
                    st.error(f"❌ Failed to load session data: {e}")
                    st.stop()

            # ---------------------------------------------------------
            # Driver Selection
            # ---------------------------------------------------------
            pick_drv1 = st.sidebar.selectbox("Driver 1", results['Abbreviation'], index=0)
            pick_drv2 = st.sidebar.selectbox("Driver 2", results['Abbreviation'], index=1)

            # ---------------------------------------------------------
            # Race or Sprint Sessions
            # ---------------------------------------------------------
            
            if session_code in {'R', 'S'}:
                tabs = [
                    "🏎️ Track Positions",
                    "📈 Net Position Gain",
                    "⏱️ Lap Time Analysis (Box Plot)",
                    "🔍 Lap Time Comparison",
                    "🚀 Average Lap Speed Each Lap",
                    "🏆 Overall Lap Time Dominance",
                    "📊 Telemetry Data",
                    "🚦 Track Dominance",
                    "🏁 Pit Stops"
                ]

                st_tabs = st.tabs(tabs)
                tab_indices = {name: i for i, name in enumerate(tabs)}

                if "active_tab" not in st.session_state:
                    st.session_state.active_tab = tabs[0]

                with st.sidebar:
                    st.header("📂 Dashboard Controls")
                    selected_tab_label = st.radio("First Select Analysis Tab:", tabs)

                st.session_state.active_tab = selected_tab_label
                comparison = RaceDataAnalysis(df, pick_drv1, pick_drv2, results, F1_REGULAR_PATH, event_name, year)
                active_index = tab_indices[st.session_state.active_tab]
                
                with st_tabs[active_index]:
                    if st.session_state.active_tab == "🏎️ Track Positions":
                        try:
                            fig = comparison.get_overall_track_positions()
                            if fig: 
                                st.plotly_chart(fig, use_container_width=True)
                            else: 
                                st.warning("Error in loading track positions.")
                        except Exception as e:
                            st.error(f"Error in track positions: {str(e)}")
                            
                    elif st.session_state.active_tab == "📈 Net Position Gain":
                        try :
                            fig = comparison.get_net_postition_gain_loss()
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Error in loading net position gain.")
                        except Exception as e:
                            st.error(f"Error in net position gain: {str(e)}")
                            
                    elif st.session_state.active_tab == "⏱️ Lap Time Analysis (Box Plot)":
                        try :
                            fig = comparison.box_plot_analysis()
                            if fig: 
                                st.plotly_chart(fig, use_container_width=True)
                            else: 
                                st.warning("Error in loading box plot analysis.")
                        except Exception as e:
                            st.error(f"Error in box plot analysis: {str(e)}")
                            
                    elif st.session_state.active_tab == "🔍 Lap Time Comparison":
                        try:
                            fig = comparison.get_lap_time_comparison()
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else: 
                                st.warning("Error in loading lap time comparison.")
                        except Exception as e:
                            st.error(f"Error in lap time comparison: {str(e)}")
                            
                    elif st.session_state.active_tab == "🚀 Average Lap Speed Each Lap":
                        try :
                            st.warning("This feature takes a little time to generate.")
                            fig = comparison.get_average_lap_speed_per_lap()
                            if fig: 
                                st.plotly_chart(fig, use_container_width=True)
                            else: 
                                st.warning("Error in loading average lap speed per lap.")
                        except Exception as e:
                            st.error(f"Error in average lap speed per lap: {str(e)}")
                            
                    elif st.session_state.active_tab == "🏆 Overall Lap Time Dominance":
                        try :
                            fig = comparison.overall_laptime_dominance()
                            if fig: 
                                st.plotly_chart(fig, use_container_width=True)
                            else: 
                                st.warning("Error in loading overall laptime dominance.")
                        except Exception as e:
                            st.error(f"Error in overall laptime dominance: {str(e)}")
                            
                    elif st.session_state.active_tab == "📊 Telemetry Data":
                        try:
                            lapnumber = st.slider("Select Lap Number", 1, int(df['LapNumber'].max()), 1)
                            data = st.selectbox("Select Telemetry Data", ["Speed", "RPM", "Throttle", "Brake", "nGear"])
                            fig = comparison.lap_telemtry_comparision(lapnumber, data)
                            if fig: 
                                st.plotly_chart(fig, use_container_width=True)
                            else: 
                                st.warning("Error in loading telemetry data.")
                        except Exception as e:
                            st.error(f"Error in telemetry data: {str(e)}")
                            
                    elif st.session_state.active_tab == "🚦 Track Dominance":
                        try:
                            max_lap = int(df['LapNumber'].max())
                            lapnumber = st.slider("Select Lap Number", 1, max_lap, 1)
                            fig = comparison.track_dominance(lapnumber)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not generate track dominance visualization for this lap.")
                        except Exception as e:
                            st.error(f"Error in track dominance: {str(e)}")

                    elif st.session_state.active_tab == "🏁 Pit Stops":
                        try :
                            fig = comparison.get_pit_stop_analysis()
                            if fig: 
                                st.plotly_chart(fig, use_container_width=True)
                            else: 
                                st.warning("Error in loading pit stop analysis.")
                        except:
                            st.warning("Error in loading pit stop analysis.")

            # ---------------------------------------------------------
            # PRACTICE / QUALIFYING / SHOOTOUT SESSIONS
            # ---------------------------------------------------------
            
            elif session_code in {'FP1', 'FP2', 'FP3', 'SS', 'Q'}:
                tabs = ["Bar Plot Analysis", "Quick Lap Analysis", "Track Dominance Map"]
                st_tabs = st.tabs(tabs)
                tab_indices = {name: i for i, name in enumerate(tabs)}

                if "active_tab" not in st.session_state:
                    st.session_state.active_tab = tabs[0]

                with st.sidebar:
                    st.header("📂 Dashboard Controls")
                    selected_tab_label = st.radio("Select Analysis Tab:", tabs)

                st.session_state.active_tab = selected_tab_label
                comparison = Practice_session_analysis(df, results, F1_REGULAR_PATH)
                active_index = tab_indices[st.session_state.active_tab]

                with st_tabs[active_index]:
                    if st.session_state.active_tab == "Bar Plot Analysis":
                        try: 
                            fig = comparison.bar_plot_analysis(pick_drv1, pick_drv2)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Error in loading bar plot analysis.")
                        except Exception as e:
                            st.error(f"Error in bar plot analysis: {str(e)}")
                            
                    elif st.session_state.active_tab == "Quick Lap Analysis":
                        try :
                            driver = st.selectbox("Select Driver", results['Abbreviation'].tolist(), index=0)
                            data = st.selectbox("Select Telemetry Data", ["Speed", "nGear", "Throttle", "Brake", "RPM"], index=0)
                            
                            fig = comparison.quickest_lap(driver, data)
                            if fig: 
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Error in loading quickest lap analysis.")
                        except Exception as e:
                            st.error(f"Error in quickest lap analysis: {str(e)}")
                            
                    elif st.session_state.active_tab == "Track Dominance Map":
                        try:
                            fig = comparison.track_dominance(pick_drv1, pick_drv2)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Error in loading track dominance map.")
                        except Exception as e:
                            st.error(f"Error in track dominance map: {str(e)}")
            
            else:
                st.warning("⚠️ Session type not recognized or data unavailable.")

    elif not event_name:
        st.warning("👈 Now select an event.")
else:
    st.warning("👈 Now select a year.")

