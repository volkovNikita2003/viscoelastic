#!/bin/bash

if [ -f clean-all.sh ]; then
    bash clean-all.sh
fi

python3 generate_projects.py
bash calc-all.sh
