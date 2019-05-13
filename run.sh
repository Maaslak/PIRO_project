#!/bin/bash
set -e
#sudo apt-get install python3-venv
chmod 755 venv/bin/activate
source venv/bin/activate
pip3 install -r ./requirements.txt --quiet
python3 main.py $1 $2
deactivate

