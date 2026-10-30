"""Recompute Experiment 2 aggregates and figures from published anonymous scores."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import experiment2_summary

if __name__=='__main__':
    experiment2_summary.OUT=Path(__file__).resolve().parent/'results/experiment2'
    experiment2_summary.main()
