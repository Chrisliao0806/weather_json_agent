import argparse
import logging
from weather_agent import WeatherAgent
from utils.logger import setup_logging

def parse_args():
    """
    Parse command-line arguments.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Weather Agent CLI")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="明天台北的天氣如何？",
        help="Question to ask the weather agent.",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default="./json_file/weather_output.json",
        help="Path to the JSON file containing weather data.",
    )
    parser.add_argument(
        "--use_local_llm",
        type=bool,
        default=True,
        help="Use local LLM for processing.",
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    # Set up logging
    setup_logging(log_level=args.log_level, log_filename="./logs/weather_agent.log")
    # Initialize the WeatherAgent
    history_list = []
    logging.info("Initializing WeatherAgent...")
    agent = WeatherAgent(
        file_path=args.file_path, use_local_llm=args.use_local_llm
    )
    # Run the agent with the specified parameters
    output, token = agent.workflow(query=args.question)
    print(output['generation'])
    logging.info("Weather agent workflow completed.")
    history_list.append([{"role": "user", "content": args.question}])
    history_list.append([{"role": "assistant", "content": output['generation']}])
    print(history_list)
    logging.info("History list updated.")
    for i in range(10):
        input_question = input("請輸入問題：")
        output, token = agent.workflow(query=input_question, history=history_list)
        print(output['generation'])
        logging.info("Weather agent workflow completed.")
        history_list.append([{"role": "user", "content": input_question}])
        history_list.append([{"role": "assistant", "content": output['generation']}])
        print(history_list)
        logging.info("History list updated.")
