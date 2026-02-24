# 🐎 Enhanced Ragozin Sheets Parser

A comprehensive horse racing analysis system that uses AI to parse Ragozin performance sheets with enhanced symbol analysis and performance insights.

## ✨ New Features

### 🤖 AI-Powered Analysis
- **Horse Analysis**: AI-generated comprehensive analysis of each horse's overall performance
- **Performance Trends**: Analysis of performance trends over time (improving, declining, consistent)
- **Race Analysis**: Detailed analysis of each individual race performance
- **Symbol Interpretation**: AI analysis of symbols before and after Ragozin figures

### 🔍 Enhanced Symbol Analysis
- **Symbol Before**: Identifies and analyzes symbols that appear before Ragozin figures
- **Symbol After**: Identifies and analyzes symbols that appear after Ragozin figures
- **Symbol Context**: AI explains what each symbol means in the context of that race
- **Symbol Impact**: Analysis of how symbols affect the interpretation of Ragozin figures

### 📊 Modern Frontend
- **Enhanced UI**: Completely rebuilt frontend with modern design
- **AI Insights Display**: Dedicated sections for AI-generated analysis
- **Symbol Visualization**: Charts and graphs for symbol analysis
- **Interactive Analysis**: Expandable sections for detailed race analysis

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd ragozin-sheets-parser

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy the environment template
cp .env.example .env

# Edit .env file with your OpenAI API key
# Get your API key from: https://platform.openai.com/api-keys
```

**Required Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key (required)

**Optional Environment Variables:**
- `OPENAI_API_BASE`: Custom API endpoint (default: https://api.openai.com/v1)
- `OPENAI_MODEL`: Model to use (default: gpt-4o)
- `BACKEND_HOST`: Backend host (default: localhost)
- `BACKEND_PORT`: Backend port (default: 8000)
- `FRONTEND_PORT`: Frontend port (default: 8502)

### 3. Set OpenAI API Key
```bash
# Option 1: Use the automated setup script (recommended)
python setup_env.py

# Option 2: Edit .env file manually
# Add your API key to the .env file

# Option 3: Use the legacy setup script
python set_api_key.py
```

### 3. Start the System
```bash
# Start both API and frontend
python start.py

# Or start individually:
# API server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Enhanced frontend
streamlit run streamlit_app_new.py
```

## 📁 File Structure

```
├── gpt_parser_alternative.py    # Enhanced GPT parser with AI analysis
├── streamlit_app_new.py        # New enhanced frontend
├── api.py                      # FastAPI backend
├── test_enhanced_parser.py     # Test script for enhanced features
├── requirements.txt            # Python dependencies
├── README_ENHANCED.md         # This file
└── cd062525.pdf               # Sample Ragozin sheet
```

## 🔧 Enhanced Data Structure

### Horse Entry
```python
@dataclass
class HorseEntry:
    name: str
    trainer: str = ""
    races: List[RaceEntry] = None
    horse_analysis: str = ""        # AI-generated overall analysis
    performance_trend: str = ""     # AI-generated trend analysis
```

### Race Entry
```python
@dataclass
class RaceEntry:
    date: str = ""
    track: str = ""
    race_type: str = ""
    ragozin_figure: float = 0.0
    finish_position: int = 0
    odds: float = 0.0
    jockey: str = ""
    weight: float = 0.0
    surface: str = ""
    distance: str = ""
    comments: str = ""
    symbol_before: str = ""         # Symbol before Ragozin figure
    symbol_after: str = ""          # Symbol after Ragozin figure
    race_analysis: str = ""         # AI-generated race analysis
```

## 🎯 Enhanced GPT Prompt

The enhanced prompt includes:

1. **Symbol Analysis Requirements**:
   - Identify symbols before and after Ragozin figures
   - Explain what each symbol means
   - Analyze how symbols affect interpretation

2. **Analysis Requirements**:
   - Detailed race performance analysis
   - Overall horse performance analysis
   - Performance trend analysis
   - Surface and distance preferences

3. **Enhanced JSON Structure**:
   - Includes all new fields for analysis
   - Structured for comprehensive insights

## 📊 Frontend Features

### Horses Overview Page
- **AI Analysis Display**: Shows horse and performance trend analysis
- **Symbol Analysis**: Displays symbols before/after Ragozin figures
- **Race History**: Enhanced race history with AI insights
- **Interactive Expanders**: Detailed analysis for each horse

### Individual Horse Analysis Page
- **Deep Analysis**: Comprehensive analysis of individual horses
- **Performance Trends**: Visual charts of performance over time
- **Symbol Analysis**: Pie charts showing symbol distribution
- **Detailed Race Analysis**: Expandable sections for each race

### Race Analysis Page
- **Overall Statistics**: Comprehensive race statistics
- **Symbol Analysis**: Bar charts of symbol distribution
- **Filtering**: Filter by track, surface, and race type
- **Visualizations**: Multiple chart types for analysis

## 🧪 Testing

### Test Enhanced Parser
```bash
python test_enhanced_parser.py
```

This will:
- Test the enhanced parser with new fields
- Verify symbol analysis functionality
- Check AI analysis generation
- Export enhanced results

### Expected Output
```
Testing Enhanced Ragozin Parser...
✅ Parser initialized successfully
🔄 Parsing PDF with enhanced analysis...
✅ Parsing completed: X horses found

🏇 Horse: KANTHAROS
   Trainer: John Smith
   Horse Analysis: This horse shows consistent performance...
   Performance Trend: Improving trend over recent races...
   Races: 5
     Race 1:
       Date: 2024-01-01
       Track: CD
       Ragozin: 85.5
       Symbol Before: 'Y'
       Symbol After: 'w'
       Race Analysis: Excellent performance with rail trip...

🎉 Enhanced fields are working correctly!
✅ Enhanced parser test completed successfully!
```

## 🔍 Symbol Analysis Examples

### Common Symbols
- **Y**: Rail trip (advantageous inside position)
- **F**: First lasix (medication indicator)
- **w**: Won race
- **b**: Bled (nosebleed during race)
- **c**: Claimed (horse was claimed by new owner)

### AI Analysis Examples
- **Race Analysis**: "Strong performance with a Ragozin figure of 85.5, indicating excellent speed. The 'Y' symbol shows the horse had a rail trip advantage, which likely contributed to the win."
- **Horse Analysis**: "This horse demonstrates consistent improvement over recent races, showing strong potential for future success. Best performances on dirt surfaces at distances of 6-7 furlongs."
- **Performance Trend**: "Clear upward trend in performance, with Ragozin figures improving from 90+ to mid-80s range, indicating the horse is hitting peak form."

## 🚀 Usage

1. **Upload PDF**: Use the enhanced upload page to parse Ragozin sheets
2. **View Overview**: See all horses with AI analysis in the overview page
3. **Deep Analysis**: Select individual horses for detailed analysis
4. **Race Analysis**: Analyze overall race patterns and symbol distribution
5. **Export Data**: Download enhanced JSON/CSV files with all analysis

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### API Configuration
- **Base URL**: `http://localhost:8000`
- **Model**: `gpt-4o` (configurable in parser)
- **Timeout**: 60 seconds per page

## 📈 Performance

- **Processing Time**: ~2-3 minutes per PDF page
- **Accuracy**: Enhanced with symbol analysis and AI insights
- **Memory Usage**: Optimized for large PDF files
- **Error Handling**: Robust error handling for missing data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the test scripts for functionality
2. Verify API key configuration
3. Check the enhanced parser logs
4. Review the symbol analysis output

---

**Enhanced Ragozin Sheets Parser** - Bringing AI-powered insights to horse racing analysis! 🐎🤖 