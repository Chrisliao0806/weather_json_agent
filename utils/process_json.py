from langchain.docstore.document import Document


def process_weather_json(json_data, file_path):
    """
    Parse the 36-hour weather forecast JSON structure from the weather bureau,
    and create a list of Document objects. Each Document represents the forecast
    of a specific weather element for a specific location.
    """
    documents = []
    try:
        # Safely read nested structures
        records = json_data.get("records", {})
        description = records.get("datasetDescription", "Unknown forecast type")
        locations = records.get("location", [])

        if not locations:
            print(
                "Warning: 'location' list not found in the 'records' section of the JSON."
            )
            return documents

        for loc in locations:
            location_name = loc.get("locationName", "Unknown location")
            weather_elements = loc.get("weatherElement", [])

            if not weather_elements:
                print(
                    f"Warning: 'weatherElement' list not found for location '{location_name}'."
                )
                continue

            for elem in weather_elements:
                element_name = elem.get(
                    "elementName", "Unknown element"
                )  # e.g., PoP, T, Wx
                time_entries = elem.get("time", [])

                if not time_entries:
                    print(
                        f"Warning: 'time' list not found for location '{location_name}', element '{element_name}'."
                    )
                    continue

                # Combine all forecast data for this location and element into a single Document
                content_lines = [
                    f"Location: {location_name}",
                    f"Weather Element: {element_name}",
                    f"Forecast Type: {description}",
                ]
                unit = "N/A"  # Initialize unit

                for time_entry in time_entries:
                    start_time = time_entry.get("startTime", "N/A")
                    end_time = time_entry.get("endTime", "N/A")
                    parameter = time_entry.get("parameter", {})
                    # parameterName contains the value, parameterUnit contains the unit
                    param_value = parameter.get("parameterName", "N/A")
                    param_unit = parameter.get("parameterUnit", "")  # Unit may be empty

                    # If the unit is 'percentage', append '%' to the value for better readability
                    display_value = (
                        f"{param_value}{'%' if param_unit == 'percentage' else ''}"
                    )
                    if (
                        unit == "N/A" and param_unit
                    ):  # Record the primary unit for this element
                        unit = param_unit

                    content_lines.append(
                        f"- Time: {start_time} to {end_time}, Forecast Value: {display_value} {param_unit if param_unit != 'percentage' else ''}".strip()
                    )

                # Combine into a single string as the Document content
                page_content = "\n".join(content_lines)

                # Create a Document with rich metadata
                metadata = {
                    "source": file_path,
                    "dataset_description": description,
                    "location_name": location_name,
                    "element_name": element_name,  # PoP, T, Wx, etc.
                    "unit": unit,  # Primary unit for this element
                    # Optionally include time range metadata, but it may be more complex
                    "first_start_time": time_entries[0].get("startTime")
                    if time_entries
                    else None,
                    "last_end_time": time_entries[-1].get("endTime")
                    if time_entries
                    else None,
                }
                documents.append(Document(page_content=page_content, metadata=metadata))

    except (KeyError, TypeError, ValueError) as e:
        print(f"Error occurred while processing the weather JSON structure: {e}")

    return documents
