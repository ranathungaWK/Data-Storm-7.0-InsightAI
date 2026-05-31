# Section D — Spend Optimization Logic

## Total budget

Total marketing budget used: `5,000,000 LKR`.

## Objective function

We compute a per-outlet score. Two variants considered:

1) Score_i = Opportunity_i

2) Score_i = Opportunity_i × BusinessStrength_i

In our pipeline we used a composite score that modifies `Opportunity` by POI, accessibility, commercial activity, cooler capacity, and penalizes saturation.

Formally in the code we compute an optimization score per outlet:

$$
Score_i = Opportunity_i \times (1 + poi_i) \times (1 + access_i) \times (1 + commercial_i) \times (1 + 0.1 \times cooler_i) / (1 + saturation_i)
$$

## Allocation formula

Allocations are proportional to the computed scores:

$$
Allocation_i = \frac{Score_i}{\sum_j Score_j} \times 5{,}000{,}000
$$

## Constraints

```
Allocation_i >= 0
sum_i Allocation_i = 5,000,000
```

## Optional top-N selection

We allow zeroing allocations for low-opportunity outlets (top-N selection) as an optional post-filter.

## Evidence

- `report_assets/top10_budget.csv` — Top 10 outlets by allocation
- `report_assets/budget_distribution.svg` — distribution histogram (generated)
- `report_assets/top100_allocations.csv` — Top 100 allocations CSV (generated)
