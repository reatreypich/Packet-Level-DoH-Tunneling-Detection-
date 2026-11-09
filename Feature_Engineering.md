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
import subprocess 
import os
import pandas as pd 

def sep(folder_path):
    files = os.listdir(folder_path)
    index1 = 0

    for index, file_name in enumerate(files):
        if file_name.endswith('.pcap'):
            a =  'tshark -r '
            b =' -q -z conv,tcp -w ' 
            c ='output.pcap'
            command1= f"{a}{file_name}{b}{index1+1}{c}"
            
            x = 'tshark -r ' 
            y =' -Y "tcp.stream" -T fields -e tcp.stream | sort -n | uniq | while read stream; do tshark -r '
            z=' -w '
            w='flow$stream.pcap -Y "tcp.stream==$stream"; done'
            command2 = f"{x}{index1+1}{c}{y}{index1+1}{c}{z}{index1+1}{w}"
            
            m = 'rm '
            command3 =f"{m}{index1+1}{c}"
            command4 =f"{m}{file_name}"
            subprocess.run(command1, shell=True)
            subprocess.run(command2, shell=True)
            subprocess.run(command3, shell=True)
            subprocess.run(command4, shell=True)
            index1 =index1+1
                   
	folder_path = 'Assigned_Path'
	sep(folder_path)
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

## Specific Block of Feature extraction ( ip_checksum,tcp_checksum, packet_length, and entropy features) 
```bash

#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./extract_features.sh input.pcap output.csv

INPUT="${1:-}"
OUTPUT="${2:-features.csv}"

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
    echo "Usage: $0 input.pcap output.csv"
    exit 1
fi

# Temporary file for extracted packet fields
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# Extract raw packet bytes and relevant fields.
# -x adds a hex dump for each packet.
tshark -r "$INPUT" \
    -Y "ip && tcp" \
    -T fields \
    -e frame.number \
    -e ip.checksum \
    -e tcp.checksum \
    -e frame.len \
    -e ip.flags \
    -e tcp.flags \
    -e tcp.payload \
    -e frame.raw \
    -E separator=/t \
    -E occurrence=f \
    > "$TMP"

# Check whether frame.raw is supported by your
# installed tshark version before running.
gawk -F '\t' '
BEGIN {
    OFS=","
    print "packet_number,ip_checksum,tcp_checksum," \
          "packet_length,packet_entropy,payload_entropy," \
          "segment_1_entropy,segment_2_entropy,segment_3_entropy"
}

function hexval(c,    p) {
    c = tolower(c)
    p = index("0123456789abcdef", c)
    return p ? p-1 : 0
}

function hex2dec(h,    i,n) {
    gsub(/^0x/, "", h)
    n=0
    for (i=1; i<=length(h); i++)
        n=n*16+hexval(substr(h,i,1))
    return n
}

function hex2bin(h,    i,v,j,b) {
    gsub(/^0x/, "", h)
    gsub(/[^0-9a-fA-F]/, "", h)
    b=""
    for (i=1; i<=length(h); i++) {
        v=hexval(substr(h,i,1))
        for (j=3; j>=0; j--)
            b=b int(v/(2^j))%2
    }
    return b
}

# Shannon entropy over byte values
function byte_entropy(hex,    i,n,v,p,H,count) {
    delete count
    gsub(/[^0-9a-fA-F]/, "", hex)
    n=int(length(hex)/2)
    if (n==0) return 0

    for (i=1; i<=n; i++) {
        v=tolower(substr(hex,2*i-1,2))
        count[v]++
    }

    H=0
    for (v in count) {
        p=count[v]/n
        H-=p*log(p)/log(2)
    }
    return H
}

# Shannon entropy over bits
function bit_entropy(bits,    i,n,z,o,p,H) {
    n=length(bits)
    if (n==0) return 0

    z=0
    for (i=1; i<=n; i++)
        if (substr(bits,i,1)=="0") z++

    o=n-z
    H=0

    if (z>0) {
        p=z/n
        H-=p*log(p)/log(2)
    }
    if (o>0) {
        p=o/n
        H-=p*log(p)/log(2)
    }
    return H
}

{
    packet=$1
    ipcs=$2
    tcpcs=$3
    plen=$4
    ipflags=$5
    tcpflags=$6
    payload=$7
    raw=$8

    if (ipcs=="" || tcpcs=="" || raw=="")
        next

    # Build binary combination
    combined = hex2bin(ipflags) \
             hex2bin(tcpflags) \
             hex2bin(ipcs) \
             hex2bin(tcpcs) \
             hex2bin(payload)

    # Divide into 15 approximately equal segments.
    # The final segment receives any remainder.
    n=length(combined)
    base=int(n/15)
    remainder=n%15

    start=1
    for (s=1; s<=15; s++) {
        seglen=base+(s==15 ? remainder : 0)
        segment=substr(combined,start,seglen)
        if (s<=3)
            E[s]=bit_entropy(segment)
        start+=seglen
    }

    print packet, \
          hex2dec(ipcs), \
          hex2dec(tcpcs), \
          plen, \
          sprintf("%.6f",byte_entropy(raw)), \
          sprintf("%.6f",byte_entropy(payload)), \
          sprintf("%.6f",E[1]), \
          sprintf("%.6f",E[2]), \
          sprintf("%.6f",E[3])
}
' "$TMP" > "$OUTPUT"

echo "Saved features to: $OUTPUT"
echo "Extracted packets: $(($(wc -l < "$OUTPUT") - 1))"
```

Note: we need to install : 
```bash
sudo apt update
sudo apt install tshark gawk -y
```
## Entropy Features Engineering 
```bash

#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./extract_16_entropy.sh input.pcap output.csv

INPUT="${1:-}"
OUTPUT="${2:-entropy_16.csv}"

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
    echo "Usage: $0 input.pcap output.csv"
    exit 1
fi

# Prevent accidental input/output overwrite
if [[ "$INPUT" == "$OUTPUT" ]]; then
    echo "Error: Input and output must be different."
    exit 1
fi

# Temporary file
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "Extracting raw packet bytes..."

# -x exports packet bytes as hexadecimal.
# -n disables name resolution.
# -l provides line-buffered output.
tshark -n -r "$INPUT" -x -l > "$TMP"

echo "Calculating entropy for 16 segments..."

gawk '
BEGIN {
    OFS=","

    printf "packet_number"
    for (i=1; i<=16; i++)
        printf ",seg%d_entropy",i
    printf "\n"

    packet=0
    hex=""
}

# Convert hexadecimal character to decimal
function hexval(c) {
    return index("0123456789abcdef",tolower(c))-1
}

# Convert hexadecimal string into binary
function hex2bin(h,    i,v,j,b) {
    b=""

    for (i=1; i<=length(h); i++) {
        v=hexval(substr(h,i,1))

        if (v<0) continue

        for (j=3; j>=0; j--)
            b=b int(v/(2^j))%2
    }

    return b
}

# Shannon entropy of binary segment
function entropy(bits,    n,i,z,o,p,H) {

    n=length(bits)

    if (n==0)
        return 0

    z=0

    for (i=1; i<=n; i++) {
        if (substr(bits,i,1)=="0")
            z++
    }

    o=n-z
    H=0

    if (z>0) {
        p=z/n
        H-=p*log(p)/log(2)
    }

    if (o>0) {
        p=o/n
        H-=p*log(p)/log(2)
    }

    return H
}

# Process one packet
function process_packet(    bits,n,base,rem,
                            i,start,len,segment) {

    if (hex=="")
        return

    packet++

    bits=hex2bin(hex)
    n=length(bits)

    base=int(n/16)
    rem=n%16

    printf "%d",packet

    start=1

    for (i=1; i<=16; i++) {

        # Distribute extra bits among first segments
        len=base+(i<=rem ? 1 : 0)

        segment=substr(bits,start,len)

        printf ",%.6f",entropy(segment)

        start+=len
    }

    printf "\n"
}

# Hex dump lines begin with a byte offset,
# followed by groups of hexadecimal bytes.
# Blank lines separate packets.

# Hex dump line
/^[[:space:]]*[0-9a-fA-F]{4,8}[[:space:]]/ {

    # Remove offset
    line=$0
    sub(/^[[:space:]]*[0-9a-fA-F]+[[:space:]]+/,"",line)

    # Extract only the hex byte portion,
    # excluding the ASCII representation
    split(line,parts,/  +/)
    data=parts[1]

    count=split(data,bytes,/[[:space:]]+/)

    for (i=1; i<=count; i++) {
        if (bytes[i] ~ /^[0-9a-fA-F][0-9a-fA-F]$/)
            hex=hex bytes[i]
    }

    next
}

# Blank line marks packet boundary
/^[[:space:]]*$/ {
    process_packet()
    hex=""
}

END {
    process_packet()
}
' "$TMP" > "$OUTPUT"

echo "Extraction completed."
echo "Output: $OUTPUT"
echo "Total packets: $(($(wc -l < "$OUTPUT") - 1))"

``` 

Note : to run this script we need to provide the input: 
```bash
./extract_16_entropy.sh doh.pcap doh_entropy_16.csv
```



















  
