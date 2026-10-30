# Emergency Department Census and Length of Stay

This project examines how emergency department census at arrival relates to patients' length of stay. Using Python, the analysis reconstructs patient census, compares patterns across patient and physician characteristics, and looks at shift timing and workload. The findings are descriptive and do not establish a causal effect of crowding.

The analysis starts from 410,937 encounters and 22,564 physician shifts. It checks the original reference relationship and estimates comparisons that add patient severity, physician, arrival time, shift timing, and operational controls. It also checks alternative census definitions, physician workload within shifts, and selection into physician assignment.

## Code

- `analysis/prepare_data.py` joins the input files, builds census and timing measures, and prints data checks.
- `analysis/analysis.py` runs the models, sensitivity checks, figures, and validation.
- `analysis/technical_notes.md` explains the main choices and limitations.
- `analysis/run.sh` runs the analysis in order.

This work used AI assistance for analysis, coding, and presentation. The code's reference checks, independently recomputed census values for a sample of encounters, and model convergence checks are included in the workflow.

## Running the analysis

The original exercise data and reference values are not included in this repository. They are required to reproduce the results. Place `encounters.csv`, `physician_shifts.csv`, and `reference/census_los_reference_values.csv` in the project folder, then run:

```bash
python3 -m pip install -r requirements.txt
bash analysis/run.sh
```

The scripts write results and figures to `output/`. Without the original input files, the analysis cannot be rerun.
