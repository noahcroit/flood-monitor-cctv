#!/bin/bash
export STAFFGAUGE=/home/watergate/flood-monitor-cctv/staff-gauge-reader/
cd $STAFFGAUGE
python3 staffgauge-worker.py -s video -j config.json -d false

