import logging
from weather_agent import WeatherAgent
from utils.logger import setup_logging

if __name__ == "__main__":
    # Set up logging
    setup_logging(log_level="INFO", log_filename="./logs/weather_agent.log")
    # Initialize the WeatherAgent
    agent = WeatherAgent(
        file_path="./json_file/weather_output.json", use_local_llm=True
    )
    # Run the agent with the specified parameters
    output = agent.workflow(query="嘉義市明天的天氣")
    print(output)
