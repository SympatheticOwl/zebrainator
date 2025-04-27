# UIUC CS 437 Lab 5 Adventures in IoT Wildlife Conservation

## Arduino
* arduino dir has the base code for the 2 circuits designed in the iot platform including messaging capabilities
* complex device includes all environment monitoring like humidity, temp, and air quality alongside standard gps and vitals
   * complex device publishes and receives messages
* simple device only adds sound monitoring to the gps and vital data
   * simple device only receives messages

## Log Parsers
[log_parser.py](log_parser.py) -> slightly modified example log parser to include vitals data in the json output

[time_data_log_parser.py](time_data_log_parser.py) -> formats and aggregates the log_parser json output into an event based timeline of where each animal is in chronological order and their current vital state

## Data Visualization
[data_visualization.py](data_visualization.py) -> creates plot data using the basic json file parsed from the slightly modified example log parser for the _READABLE.txt files

[enhanced_data_visualization.py](enhanced_data_visualization.py) -> creates better visualizations using the event timeline output of time_data_log_parser.py
