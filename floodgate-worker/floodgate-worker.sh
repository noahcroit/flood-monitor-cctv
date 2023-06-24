#!/bin/bash
export GATE=/home/watergate/flood-monitor-cctv/floodgate-worker/
cd $GATE
python3 worker.py -j config.json
