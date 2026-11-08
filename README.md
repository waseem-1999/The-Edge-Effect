# The Edge Effect

Monte Carlo study of which customer problems to prioritize for immediate NPS gains.

Read the supplied PDF.

## Reproduce

Original environment: Python 3.13.0 and NumPy 2.1.2.

```powershell
python -m pip install numpy==2.1.2
python simulate.py
```

The experiment uses seed `20260809`, 12,000 fictional companies, and 20,000 ratings per company. Results are saved to `results.json`; the program also generates a browser copy, `results.js`.

These are simulated results, not business forecasts. Assumptions and limitations are explained in the paper.
