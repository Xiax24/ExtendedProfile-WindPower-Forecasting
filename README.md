# ExtendedProfile-WindPower-Forecasting
# Extended Profile Including Surface-Layer Wind Speed for Farm-Scale Power Forecasting

This repository contains the code used to reproduce the results and figures in the associated GRL submission.

## Data

The dataset is archived at Science Data Bank (DOI: XXX).

After downloading, place the data files into:

    data/

## Requirements

Install dependencies:

    pip install -r requirements.txt

## Reproducing Figures

Figure 1:
    python figures/figure-1-a.py
    python figures/figure-1-b.py
    python figures/figure-1-c.py

Figure 2:
    python figures/figure-2-a-0.py
    ...

Figure 3:
    python figures/figure-3-a.py
    ...

Figure 4:
    python figures/figure-4-abcd.py

## Flow Regime Classification

The flow regime classification method is implemented in:

    methods/method_classify_regime.py
