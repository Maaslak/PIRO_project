#!/bin/bash
set -e
sudo apt-get install python3-venv
python3 -m venv ./venv
chmod 755 venv/bin/activate
./venv/bin/activate
pip3 install -r ./requirements.txt
python3 main.py $1 $2
