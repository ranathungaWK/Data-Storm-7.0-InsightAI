# Budget Allocation Submission

This folder contains the Western Province trade spend allocation deliverable.

## Files

- `teamname_budget_allocations.csv` - final budget allocation file
- `run_pipeline.py` - pipeline copy for reference
- `README.md` - project overview copy for reference

## How the allocation is built

The allocation starts from the Model A peak sales output:

`opportunity = predicted_peak_sales - current_sales`

Then the score is adjusted using business signals from the outlet feature set:

`optimization_score = opportunity * (1 + poi_influence_score) * (1 + accessibility_weighted_score) * (1 + commercial_score) * (1 + 0.1 * cooler_count) / (1 + saturation_score)`

## Western Province filter

The repository does not include a province column, so Western Province outlets are derived from coordinates.

The current implementation uses this bounding box:

- Latitude: `6.0` to `7.5`
- Longitude: `79.7` to `80.3`

Only outlets inside that coordinate range are included in the budget allocation.

## Budget normalization

The allocation scores are normalized so they sum to the fixed marketing budget:

`Trade_Spend_Allocation = optimization_score / sum(optimization_score) * 5_000_000`

If all scores are zero, the budget is split evenly across the selected outlets.

## Output format

The final CSV contains exactly these columns:

- `Outlet_ID`
- `Trade_Spend_Allocation`

The values are written in LKR.
