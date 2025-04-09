import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
from datetime import datetime
import google.generativeai as genai
import io
import base64
from bs4 import BeautifulSoup
import markdown
import os

warnings.filterwarnings('ignore')

def get_gemini_key():
    is_cloud = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "1"
    key = None

    if is_cloud:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            st.error("❌ Gemini API key missing in cloud environment! Add GEMINI_API_KEY to your environment variables")
    else:
        try:
            from config import GEMINI_API_KEY
            key = GEMINI_API_KEY
        except ImportError:
            st.error("""🔑 Gemini API Limit Reached""")
    return key

st.set_page_config(page_title="Tariff Impact Analysis", page_icon="📊", layout="wide")

st.title("Effects of Tariffs on Global Markets")
st.markdown("""
This application analyzes the impact of tariffs on global markets by examining market indexes 
between April 1-8, 2025. The analysis includes animated visualizations of market performance 
and insights on affected countries and sectors.
""")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('global_indices.csv')
        df['Percent Change (%)'] = pd.to_numeric(df['Percent Change (%)'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()
    
def display_report(report_text):
    if not report_text:
        return
    
    st.markdown('<h2 class="report-title">Global Tariff Impact Report</h2>', unsafe_allow_html=True)
    st.markdown(report_text)

    st.markdown("""
        <style>
            .report-title {
                font-size: 58px !important;
                color: #ff;
                text-align: center;
                margin-bottom: 30px;
            }
            .section-header {
                font-size: 22px !important;
                color: #1e3a8a;
                border-bottom: 2px solid #1e3a8a;
                padding-bottom: 5px;
                margin-top: 20px;
            }
            .highlight {
                background-color: #f0f9ff;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .negative {
                color: #ef4444;
                font-weight: bold;
            }
            .positive {
                color: #10b981;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

def generate_gemini_report(news_df, indices_df, api_key):
    genai.configure(api_key=api_key)
    market_data = indices_df.to_dict('records')
    news_headlines = news_df.head(5).to_dict('records')
    
    prompt = f"""
    Analyze this financial market data and current trade news to generate a comprehensive, data-driven report on tariff impacts:
    
    MARKET DATA (April 1-8, 2025):
    {market_data}
    
    RECENT TRADE NEWS:
    {news_headlines}
    
    Create a polished, professional report with the following EXACT sections:
    
    Executive Summary
    • Provide a concise overview of key findings and tariff impact (150 words)
    • Include 3 most significant market movements with precise percentages
    • Highlight critical correlation between tariff news and market reaction
    
    Market Performance Analysis
    • Analyze major indices performance with specific percentage changes
    • Compare sector performance (identify top 3 performers and bottom 3 underperformers)
    • Include volatility metrics and trading volume analysis where relevant
    • Identify specific price movements correlated with tariff announcements
    
    Tariff Policy Evaluation
    • Analyze specific tariff measures mentioned in the news
    • Evaluate potential economic impact using concrete metrics (GDP effect, inflation implications)
    • Compare with historical tariff impacts using relevant precedents
    
    Supply Chain Disruption Assessment
    • Identify key industries facing supply chain challenges
    • Quantify impact on input costs and pricing power (use percentages)
    • Highlight companies/sectors with geographic exposure concerns
    
    Consumer Impact Analysis
    • Project effects on consumer prices with specific percentage estimates
    • Analyze potential shifts in consumer spending patterns
    • Identify categories of goods most affected
    
    Investor Strategy Recommendations
    • Provide tactical asset allocation suggestions with specific weighting changes
    • Identify 3-5 defensive positioning strategies with clear rationales
    • Suggest specific sectors for overweight/underweight positions
    
    International Trade Implications
    • Analyze impact on major trading partners (focus on largest 3-4 relationships)
    • Evaluate currency implications with specific exchange rate projections
    • Assess potential retaliatory measures and their market impact
    
    Future Outlook & Timeline
    • Project key milestones for tariff implementation
    • Identify critical indicators to monitor over next 30/60/90 days
    • Provide probability assessment of various scenarios
    
    Interesting Take
    . Give your take on some "winners" are actually losing long-term market share, while certain "losers" are 
        developing resilient trade alternatives that could position them better for the future.

    FORMAT REQUIREMENTS:
    • Use professional, concise language suitable for sophisticated investors
    • Include specific data points, percentages, and numbers (e.g., -10.5%, not -10.5 percent)
    • Bold key insights and important figures
    • Create one brief bullet-point conclusion at the end of each section
    • Ensure analytical depth while maintaining readability
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Failed to generate report: {str(e)}")
        return None
    
@st.cache_data
def load_news_data():
    try:
        news_df = pd.read_csv('global_finance_news.csv')
        return news_df
    except Exception as e:
        st.error(f"Error loading news data: {str(e)}")
        return pd.DataFrame()

def main():
    df = load_data()
    news_df = load_news_data()
    
    if df.empty:
        st.error("No data available for analysis.")
        return
        
    df_sorted = df.sort_values(by='Percent Change (%)')
    
    country_iso_codes = {
    'United States': 'us', 'Germany': 'de', 'United Kingdom': 'gb', 'France': 'fr',
    'Japan': 'jp', 'Canada': 'ca', 'Australia': 'au', 'Brazil': 'br',
    'India': 'in', 'South Korea': 'kr', 'China': 'cn', 'Hong Kong': 'hk',
    'Taiwan': 'tw', 'Netherlands': 'nl', 'Switzerland': 'ch', 'Italy': 'it',
    'Spain': 'es', 'Sweden': 'se', 'Belgium': 'be', 'Norway': 'no',
    'Denmark': 'dk', 'Finland': 'fi', 'Portugal': 'pt', 'Greece': 'gr',
    'Poland': 'pl', 'Turkey': 'tr',
    'South Africa': 'za', 'Nigeria': 'ng', 'Egypt': 'eg', 'Kenya': 'ke',
    'Russia': 'ru', 'Myanmar': 'mm'
    }

    country_iso3_codes = {
        'United States': 'USA', 'Germany': 'DEU', 'United Kingdom': 'GBR', 'France': 'FRA',
        'Japan': 'JPN', 'Canada': 'CAN', 'Australia': 'AUS', 'Brazil': 'BRA',
        'India': 'IND', 'South Korea': 'KOR', 'China': 'CHN', 'Hong Kong': 'HKG',
        'Taiwan': 'TWN', 'Netherlands': 'NLD', 'Switzerland': 'CHE', 'Italy': 'ITA',
        'Spain': 'ESP', 'Sweden': 'SWE', 'Belgium': 'BEL', 'Norway': 'NOR',
        'Denmark': 'DNK', 'Finland': 'FIN', 'Portugal': 'PRT', 'Greece': 'GRC',
        'Poland': 'POL', 'Turkey': 'TUR',
        'South Africa': 'ZAF', 'Nigeria': 'NGA', 'Egypt': 'EGY', 'Kenya': 'KEN',
        'Russia': 'RUS', 'Myanmar': 'MMR'
    }

    min_change = df_sorted['Percent Change (%)'].min()
    max_change = df_sorted['Percent Change (%)'].max()
    
    df['ISO3'] = df['Country'].map(country_iso3_codes)
    
    # Create tabs for Market Overview and News
    tab1, tab2 = st.tabs(["Market Overview", "News"])
    
    with tab1:
        col_stats, col_map = st.columns([1, 3])
        
        with col_stats:
            st.markdown("<h4 style='text-align: center;'>Analysis Period</h4>", unsafe_allow_html=True)
            st.markdown("<h6 style='text-align: center;'>1st - 8th April, 2025</h6>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='text-align: center; border: 1px solid #f0f0f0; border-radius: 5px; padding: 5px 3px; margin-bottom: 5px;'>"
                      f"<div style='font-size: 0.7em; color: gray;'>Most Affected</div>"
                      f"<div style='font-weight: bold; font-size: 0.9em;'>{df_sorted.iloc[0]['Country']}</div>"
                      f"<div style='color: #ff4b4b; font-weight: bold; font-size: 0.9em;'>{df_sorted.iloc[0]['Percent Change (%)']:.2f}%</div>"
                      f"</div>", unsafe_allow_html=True)
        
            st.markdown(f"<div style='text-align: center; border: 1px solid #f0f0f0; border-radius: 5px; padding: 5px 3px; margin-bottom: 5px;'>"
                      f"<div style='font-size: 0.7em; color: gray;'>Average Change</div>"
                      f"<div style='font-weight: bold; font-size: 0.9em;'>&nbsp;</div>"
                      f"<div style='color: #ff4b4b; font-weight: bold; font-size: 0.9em;'>{df['Percent Change (%)'].mean():.2f}%</div>"
                      f"</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='text-align: center; border: 1px solid #f0f0f0; border-radius: 5px; padding: 5px 3px;'>"
                      f"<div style='font-size: 0.7em; color: gray;'>Least Affected</div>"
                      f"<div style='font-weight: bold; font-size: 0.9em;'>{df_sorted.iloc[-1]['Country']}</div>"
                      f"<div style='color: #ff4b4b; font-weight: bold; font-size: 0.9em;'>{df_sorted.iloc[-1]['Percent Change (%)']:.2f}%</div>"
                      f"</div>", unsafe_allow_html=True)
    
        with col_map:
            st.markdown("<h4 style='text-align: center;'>Global Market Impact</h4>", unsafe_allow_html=True)
            
            fig = px.choropleth(
                df,
                locations='ISO3',
                color='Percent Change (%)',
                hover_name='Country',
                color_continuous_scale='Reds_r',
                range_color=[min_change, 0],
                labels={'Percent Change (%)': 'Market Change (%)'},
                title='',
                height=300,
            )
            
            for i, row in df.iterrows():
                fig.add_annotation(
                    x=row['ISO3'],
                    y=row['Country'],
                    text=f"{row['Percent Change (%)']:.1f}%",
                    showarrow=False,
                    font=dict(size=8, color='black'),
                    xref="x",
                    yref="y"
                )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='equirectangular'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab1:
        st.markdown("<h4 style='text-align: center;'>Market Impact Overview</h4>", unsafe_allow_html=True)
        
        with st.container():
            num_cols = 1
            rows = [st.columns(num_cols) for _ in range((len(df_sorted) + num_cols - 1) // num_cols)]
            
            cols = [col for row in rows for col in row]
            
            for i, (_, row) in enumerate(df_sorted.iterrows()):
                if i < len(cols):
                    with cols[i]:
                        country = row['Country']
                        index_name = row['Index']
                        percent_change = row['Percent Change (%)']
                        iso_code = country_iso_codes.get(country, 'xx').lower()
                        
                        normalized_value = (percent_change - min_change) / (max_change - min_change)
                        color = f"rgba(255, {int(normalized_value * 255)}, {int(normalized_value * 255)}, 0.8)"
                        
                        st.markdown(
                            f"""
                            <div style='display: flex; width: 100%; margin-bottom: 10px;'>
                                <div style='width: 20%;'>
                                    <div style='display: flex; align-items: center;'>
                                        <img src='https://flagcdn.com/16x12/{iso_code}.png' width='16' style='margin-right: 5px;'>
                                        <span style='font-size: 0.8em; font-weight: bold;'>{country}</span>
                                    </div>
                                    <div style='font-size: 0.7em; color: gray;'>{index_name}</div>
                                </div>
                                <div style='width: 80%; padding-left: 10px;'>
                                    <div style='background-color: {color}; width: 100%; height: 12px; border-radius: 2px; display: flex; align-items: center; justify-content: center;'>
                                        <span style='color: white; font-weight: bold; font-size: 0.7em;'>{percent_change:.1f}%</span>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    
    with tab1:
        st.markdown("<h4 style='text-align: center;'>AI-Powered Tariff Impact Analysis</h4>", unsafe_allow_html=True)
        
        indices_df = pd.read_csv("global_indices.csv")       
        
        with st.spinner('🧠 Generating AI-powered analysis...'):
            api_key = get_gemini_key()
            if not api_key:
                st.error("🚫 Gemini API key not configured. See error messages above for setup instructions")
                return
            report = generate_gemini_report(news_df, indices_df, api_key)
            
            if report:
                st.markdown('<h2 class="report-title">Global Tariff Impact Report</h2>', 
                           unsafe_allow_html=True)
                st.divider()
                
                st.markdown(report, unsafe_allow_html=True)
                
                report_map_fig = px.choropleth(
                    df,
                    locations='ISO3',
                    color='Percent Change (%)',
                    hover_name='Country',
                    color_continuous_scale='Reds_r',
                    range_color=[min_change, 0],
                    labels={'Percent Change (%)': 'Market Change (%)'},
                    height=400
                )
                
                report_map_fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    geo=dict(
                        showframe=False,
                        showcoastlines=True,
                        projection_type='equirectangular'
                    ),
                    coloraxis_colorbar=dict(
                        title="Market Change (%)",
                        thicknessmode="pixels", thickness=20,
                        lenmode="pixels", len=300,
                        yanchor="middle"
                    )
                )
                
                map_html = report_map_fig.to_html(include_plotlyjs='cdn', full_html=False)
                
                report_html = markdown.markdown(report)
                soup = BeautifulSoup(report_html, 'html.parser')
                
                for h2 in soup.find_all('h2'):
                    h2['class'] = h2.get('class', []) + ['section-header']
                
                for strong in soup.find_all('strong'):
                    text = strong.get_text()
                    if any(neg in text.lower() for neg in ['decline', 'drop', 'fall', 'decrease', '-', 'negative']):
                        strong['class'] = strong.get('class', []) + ['negative']
                    elif any(pos in text.lower() for pos in ['increase', 'rise', 'grow', 'positive', '+']):
                        strong['class'] = strong.get('class', []) + ['positive']
                
                table_html = "<h2 class='section-header'>Global Market Indices Data</h2>"
                table_html += "<p>The table below provides detailed market data for global indices during the analysis period (April 1-8, 2025).</p>"
                table_html += "<table class='data-table'>"
                table_html += "<thead><tr>"
                table_html += "<th>Country</th><th>Index</th><th>Ticker</th><th>Price on Apr 1</th><th>Price on Apr 8</th><th>Change (%)</th>"
                table_html += "</tr></thead><tbody>"
                
                indices_df_sorted = indices_df.sort_values(by='Percent Change (%)')
                
                for _, row in indices_df_sorted.iterrows():
                    percent_change = row['Percent Change (%)']
                    change_class = 'negative' if percent_change < 0 else 'positive'
                    table_html += f"<tr>"
                    table_html += f"<td>{row['Country']}</td>"
                    table_html += f"<td>{row['Index']}</td>"
                    table_html += f"<td>{row['Ticker']}</td>"
                    table_html += f"<td>{row['Price on 2025-04-01']}</td>"
                    table_html += f"<td>{row['Price on 2025-04-08']}</td>"
                    table_html += f"<td class='{change_class}'>{percent_change:.2f}%</td>"
                    table_html += f"</tr>"
                
                table_html += "</tbody></table>"
                
                report_html = f'<div class="report-content">{str(soup)}</div>'
                report_html += f'<div class="data-table-container">{table_html}</div>'
                
                html_content = f"""
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Global Tariff Impact Report</title>
                    <style>
                        body {{ 
                            font-family: 'Segoe UI', Arial, sans-serif; 
                            margin: 0; 
                            padding: 0; 
                            color: #333; 
                            line-height: 1.6;
                            background-color: #f9f9f9;
                        }}
                        .container {{ 
                            max-width: 1200px; 
                            margin: 0 auto; 
                            padding: 40px 20px; 
                            background-color: white;
                            box-shadow: 0 0 20px rgba(0,0,0,0.05);
                        }}
                        header {{ 
                            text-align: center; 
                            margin-bottom: 40px; 
                            padding-bottom: 20px;
                            border-bottom: 1px solid #eaeaea;
                        }}
                        h1 {{ 
                            color: #1e3a8a; 
                            font-size: 32px; 
                            margin-bottom: 10px;
                        }}
                        h2 {{ 
                            color: #1e3a8a; 
                            font-size: 24px;
                            border-bottom: 2px solid #1e3a8a; 
                            padding-bottom: 8px; 
                            margin-top: 40px;
                        }}
                        h3 {{ 
                            color: #2563eb; 
                            font-size: 20px; 
                            margin-top: 30px;
                        }}
                        p {{ 
                            margin-bottom: 16px; 
                            text-align: justify;
                        }}
                        ul, ol {{ 
                            margin-bottom: 20px; 
                            padding-left: 25px;
                        }}
                        li {{ 
                            margin-bottom: 8px; 
                        }}
                        .highlight {{ 
                            background-color: #f0f9ff; 
                            padding: 20px; 
                            border-radius: 10px; 
                            margin: 25px 0; 
                            border-left: 4px solid #3b82f6;
                        }}
                        .negative {{ 
                            color: #ef4444; 
                            font-weight: bold; 
                        }}
                        .positive {{ 
                            color: #10b981; 
                            font-weight: bold; 
                        }}
                        .map-container {{ 
                            margin: 40px 0; 
                            padding: 20px; 
                            background-color: white; 
                            border-radius: 10px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                        }}
                        .map-title {{ 
                            text-align: center; 
                            margin-bottom: 20px; 
                            color: #1e3a8a;
                            font-size: 24px;
                        }}
                        table {{ 
                            width: 100%; 
                            border-collapse: collapse; 
                            margin: 25px 0; 
                            font-size: 14px; 
                        }}
                        th {{ 
                            background-color: #1e3a8a; 
                            color: white; 
                            font-weight: bold; 
                            padding: 12px; 
                            text-align: left; 
                        }}
                        td {{ 
                            padding: 10px 12px; 
                            border-bottom: 1px solid #eaeaea; 
                        }}
                        tr:nth-child(even) {{
                            background-color: #f8fafc; 
                        }}
                        .data-table-container {{
                            margin: 40px 0;
                            overflow-x: auto;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                            border-radius: 10px;
                            background-color: white;
                            padding: 20px;
                        }}
                        .data-table {{
                            width: 100%;
                            min-width: 800px;
                        }}
                        .data-table th {{
                            position: sticky;
                            top: 0;
                            z-index: 10;
                        }}
                        .footer {{
                            text-align: center; 
                            margin-top: 60px; 
                            padding-top: 20px; 
                            border-top: 1px solid #eaeaea; 
                            color: #6b7280; 
                            font-size: 14px; 
                        }}
                        strong, b {{ 
                            font-weight: 600; 
                            color: #1f2937; 
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <header>
                            <h1>Global Tariff Impact Report</h1>
                            <h4>Analysis Period: April 1st - 8th, 2025</h>
                        </header>
                        
                        <div class="content">
                            {report_html}
                            
                            <div class="map-container">
                                <h2 class="map-title">Global Market Impact Map</h2>
                                <p>The map below illustrates the percentage change in major market indices across different countries during the analysis period.</p>
                                {map_html}
                            </div>
                        </div>
                        
                        <div class="footer">
                            <p>Generated on {datetime.now().strftime("%B %d, %Y")} | Echoes of Liberation Day Analysis</p>
                        </div>
                    </div>
                </body>
                </html>
                """           
                
                b64 = base64.b64encode(html_content.encode()).decode()
                href = f'data:text/html;base64,{b64}'
                
                st.markdown(
                    f'<a href="{href}" download="tariff_report_{datetime.now().strftime("%Y%m%d")}.html">📥 Download Full Report</a>',
                    unsafe_allow_html=True
                )
    
    with tab2:
        st.markdown("<h2 style='text-align: center;'>Global Finance News</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Latest news related to global markets and tariff impacts</p>", unsafe_allow_html=True)
        
        countries = sorted(news_df['country'].unique().tolist())
        selected_country = st.selectbox("Filter news by country", ["All Countries"] + countries)
        
        if selected_country != "All Countries":
            filtered_news = news_df[news_df['country'] == selected_country]
        else:
            filtered_news = news_df
        
        filtered_news = filtered_news.sort_values(by='date', ascending=False)
        
        st.markdown("""
        <style>
        .news-container {
            background-color: rgb(14, 17, 23);
            border: 1px solid #f0f0f0;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .news-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .news-source {
            font-weight: bold;
            color: rgb(250, 250, 250);
        }
        .news-date {
            color: #6b7280;
            font-size: 0.9em;
        }
        .news-title {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .news-link {
            display: inline-block;
            margin-top: 10px;
            color: #2563eb;
            text-decoration: none;
        }
        .news-link:hover {
            text-decoration: underline;
        }
        .country-tag {
            display: inline-block;
            background-color: #e5e7eb;
            color: #4b5563;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-right: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        for _, news in filtered_news.iterrows():
            country = news['country']
            date = news['date']
            source = news['source']
            title = news['title']
            url = news['url']
            
            # Get country ISO code for flag
            iso_code = country_iso_codes.get(country, 'xx').lower()
            
            st.markdown(f"""
            <div class="news-container">
                <div class="news-header">
                    <div>
                        <span class="country-tag">
                            <img src="https://flagcdn.com/16x12/{iso_code}.png" width="16" style="margin-right: 5px; vertical-align: middle;">
                            {country}
                        </span>
                    </div>
                    <div class="news-date">{date}</div>
                </div>
                <div class="news-title">{title}</div>
                <div class="news-source">Source: {source}</div>
                <a href="{url}" target="_blank" class="news-link">Read full article →</a>
            </div>
            """, unsafe_allow_html=True)
                  
if __name__ == "__main__":
    main()