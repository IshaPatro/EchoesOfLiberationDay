# Echoes of Liberation Day: Tariff Impact Analysis

A Streamlit web application that analyzes the effects of tariffs on global markets by examining market indexes and news data between April 1-8, 2024. The application provides visualizations of market performance and AI-powered insights on affected countries and sectors.

## Features

- **Market Performance Analysis**: Visualizes the percentage changes in major global market indexes during the specified period.
- **News Analysis**: Collects and displays relevant news articles from the analyzed period using yfinance.
- **AI-Generated Insights**: Uses a transformer model to analyze market data and news sentiment, generating insights about tariff impacts.
- **Interactive Visualizations**: Provides interactive charts and graphs for better data understanding.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/EchoesOfLiberationDay.git
   cd EchoesOfLiberationDay
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501).

3. Use the sidebar to select which market indexes you want to analyze.

4. Navigate through the tabs to explore different aspects of the analysis:
   - **Market Performance**: View charts showing how different markets performed during the period.
   - **News Analysis**: Browse news articles related to the selected market indexes.
   - **AI Insights**: Read AI-generated insights about tariff impacts on markets, countries, and sectors.

## Data Sources

- Market data is fetched from Yahoo Finance using the yfinance library.
- News data is also collected from Yahoo Finance's news API.

## Technical Details

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Plotly, Matplotlib
- **Natural Language Processing**: Hugging Face Transformers

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and data processing logic.
- `requirements.txt`: List of Python dependencies required for the project.

## Future Enhancements

- Add more sophisticated NLP models for better news analysis.
- Include more data sources for comprehensive market analysis.
- Implement predictive analytics to forecast potential future impacts of tariffs.
- Add user authentication for personalized analysis.

## License

MIT