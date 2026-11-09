# 1. Flow and Sub-Flow Feature Extraction Process Using CICFlowMeter Tool 
## Installation 
```bash
git clone https://github.com/hieulw/cicflowmeter
cd cicflowmeter
uv sync
source .venv/bin/activate
```
## Usage 
```bash
usage: cicflowmeter [-h] (-i INPUT_INTERFACE | -f INPUT_FILE | -d INPUT_DIRECTORY) (-c | -u) [--fields FIELDS] [--merge] [-v] output

positional arguments:
  output                output file name (in csv mode), url (in url mode), or output directory (in directory mode)

options:
  -h, --help            show this help message and exit
  -i INPUT_INTERFACE, --interface INPUT_INTERFACE
                        capture online data from INPUT_INTERFACE
  -f INPUT_FILE, --file INPUT_FILE
                        capture offline data from INPUT_FILE
  -d INPUT_DIRECTORY, --directory INPUT_DIRECTORY
                        capture offline data from all pcap files in INPUT_DIRECTORY
  -c, --csv             output flows as csv
  -u, --url             output flows as request to url
  --fields FIELDS       comma separated fields to include in output (default: all)
  --merge               merge all pcap files into a single CSV (only works with -d/--directory mode)
  -v, --verbose         more verbose
```
## Samples of Usage
- Convert pcap file to flow csv:
``` bash
cicflowmeter -f example.pcap -c flows.csv
```
- Convert all pcap files in a directory(separate CSVs)
``` bash
cicflowmeter -d ./pcap_folder/ -c ./csv_output/
```
- Convert all PCAP files in a directory (merged into single CSV)
``` bash
cicflowmeter -d ./pcap_folder/ -c ./csv_output/ --merge
```
- Sniff packets real-time from interface to flow request: (need root permission)
``` bash
cicflowmeter -i eth0 -u http://localhost:8080/predict
```
  Reference: https://pypi.org/project/cicflowmeter/

# Flow Separation Process 
``` bash
#!/bin/bash

PCAP_DIR='/Path/'

# Loop through all .pcap files in the directory
for PCAP_FILE in "$PCAP_DIR"/*.pcap; do
    # Check if no PCAP files are found
    if [ ! -e "$PCAP_FILE" ]; then
        echo "No PCAP files found in $PCAP_DIR"
        exit 0
    fi

    echo "Processing file: $PCAP_FILE"
    
    cicflowmeter -f  "$PCAP_FILE" -c  "${PCAP_FILE%.pcap}_flow.csv"
    rm "$PCAP_FILE"

	
    echo "Processed: $PCAP_FILE"
done

echo "All PCAP files processed."
```
## The process to select only Application data packet traffic from the flow 

``` bash
#!/bin/bash

INPUT="Provided_doh_file.pcap"
OUTPUT="doh_application_data.pcap"

tshark -r "$INPUT" \
    -Y "tls.record.version == 0x0303 && tls.record.content_type == 23" \
    -F pcap \
    -w "$OUTPUT"

echo "TLS Application Data saved to: $OUTPUT"
```
## Sub-flow Extraction 

``` bash
editcap -c N input.pcap output.pcap
```
Note: N is the number of packets that we assign for each sub-flow 
























  
