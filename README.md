# 🌦️ Weather JSON Agent

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## 📋 Overview

Weather JSON Agent is an advanced, AI-powered system that processes weather forecast data from Taiwan's Central Weather Bureau and provides intelligent responses to natural language queries. Leveraging the power of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), this agent transforms complex weather JSON data into meaningful, conversational responses.

![Weather Agent Architecture](https://via.placeholder.com/800x400?text=Weather+Agent+Architecture)

## 🌟 Features

- **Intelligent Query Processing**: Understands natural language questions about weather forecasts
- **Dynamic RAG System**: Uses a sophisticated state graph to determine the best response strategy
- **Adaptive Response Generation**: Routes between RAG and plain LLM responses based on document relevance
- **Vector Database Integration**: Efficiently stores and retrieves weather information using Chroma
- **Support for Multiple Weather Elements**: Processes various forecast elements (temperature, precipitation probability, etc.)
- **Flexible LLM Support**: Works with both local and cloud-based language models
- **Comprehensive Logging**: Tracks token usage and system decisions

## 🚀 How It Works

The Weather JSON Agent implements a sophisticated workflow using LangGraph:

1. **Document Retrieval**: Retrieves relevant weather documents based on the user query
2. **Document Grading**: Evaluates document relevance to the query
3. **Intelligent Routing**: Decides whether to use RAG or direct LLM response
4. **Response Generation**: Creates human-readable weather information

![Workflow Diagram](https://via.placeholder.com/800x300?text=Weather+Agent+Workflow)

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/Chrisliao0806/weather_json_agent.git
cd weather_json_agent

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export WEB_KEY="your_weather_api_key"
```

## 🔧 Usage

### Command Line Interface

```bash
python main.py --question "明天台北的天氣如何？" --file-path "./json_file/weather_output.json" --use_local_llm True
```

### Parameters

- `--question`: Your weather-related question (default: "明天台北的天氣如何？")
- `--file-path`: Path to weather JSON data (default: "./json_file/weather_output.json")
- `--use_local_llm`: Whether to use a local LLM (default: True)
- `--log-level`: Logging level (default: INFO)

## 📊 System Architecture

The system consists of several key components:

- **WeatherAgent**: Core class that orchestrates the entire workflow
- **Document Embedding**: Converts weather data into vector representations
- **Vector Database**: Stores embeddings for efficient retrieval
- **LLM Integration**: Connects to various language models for response generation
- **Workflow Management**: Uses LangGraph for flexible decision routing
