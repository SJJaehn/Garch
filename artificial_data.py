"""
Generate the synthetic price datasets (monte_carlo, garch, dcc).

All the data-generating-process parameters live in garch/data/synthetic.py.

    python artificial_data.py
"""
from garch.data.synthetic import main


if __name__ == "__main__":
    main()
