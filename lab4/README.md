1. Project Overview
This project implements a comprehensive stock analysis and mock trading system. It collects real-time data, generates trading signals based on technical indicators (MA, RSI, Hybrid), executes mock trades, and evaluates portfolio performance against benchmarks.

2. Project Structure
The project is organized into the following directory structure :

data/: Stores extracted datasets (e.g., Yahoo Finance data).

output/: Destination for generated results, CSV reports, and visualization plots.

script/: Contains source code and logic.

venv/: Virtual environment files.

Installation & Requirements

Python Version: > 3.12 (Required for pandas_ta compatibility).


Dependencies: Listed in requirements.txt.

To set up:

`pip install -r requirements.tx`


Command:


`python export_reports.py [symbol]`
