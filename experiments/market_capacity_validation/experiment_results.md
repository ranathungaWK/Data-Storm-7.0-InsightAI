# Market Capacity Experiment Report

## Setup
- Baseline model: market-capacity version using location, POI, competition, accessibility, and outlet static signals.
- Ablation model: same setup but without outlet_size and cooler_count.
- Ranking metric: market_gap = predicted_market_capacity - avg_monthly_volume.

## Hold-out comparison
- Full market-capacity blend RMSE: 197.21
- Full market-capacity blend R²: 0.8313
- Ablated blend RMSE: 502.46
- Ablated blend R²: -0.0952
- R² delta (full - ablated): +0.9265
- RMSE delta (full - ablated): -305.25

## Permutation importance on full market-capacity model
                     feature  rmse_increase_mean  rmse_increase_std
                 outlet_size          318.841091           2.797732
                cooler_count           72.703926           1.588266
   master_completeness_score            0.732961           0.279361
accessibility_weighted_score            0.078432           0.041407
        competitor_count_1km            0.036379           0.082652
            commercial_score            0.022380           0.028099
         poi_influence_score            0.019325           0.025576
                  restaurant            0.019081           0.033096
         accessibility_score            0.009774           0.014461
               train_station            0.000000           0.000000

## Top / bottom 50 hold-out inspection
    segment  count  predicted_market_capacity_mean  current_sales_mean  market_gap_mean  outlet_size  cooler_count  saturation_score  nearest_competitor_distance  poi_influence_score  accessibility_weighted_score  competitor_count_1km  competitor_count_500m  commercial_score  mobility_score  accessibility_score
     top_50     50                     1577.849800         1138.664522       439.185278      4.64000         3.820          1.876195                   291.971521             2.105594                      0.115581               13.8200                3.28000          4.720000        0.006552              0.36000
  bottom_50     50                      554.735405          554.734648         0.000757      0.66000         0.940          2.001458                   249.219475             2.837168                      0.190265               14.7400                3.80000          5.460000        0.010786              0.78000
holdout_all   4000                      342.855555          208.638381       134.217174      0.93775         1.274          1.314364                   333.803339             2.052861                      0.165164                8.5265                2.18275          4.165003        0.009363              0.52725

## Highest market-gap hold-out outlets
outlet_id  predicted_value  current_sales  market_gap
OUT_00367          1691.67     538.953703 1152.716297
OUT_00062          1692.88     574.308507 1118.571493
OUT_00050          1575.93     519.991310 1055.938690
OUT_00087          1547.28     523.219866 1024.060134
OUT_00339          1574.50     563.540974 1010.959026
OUT_08452          1703.42    1284.279954  419.140046
OUT_07080          1690.36    1284.760947  405.599053
OUT_05007          1697.33    1292.140637  405.189363
OUT_08159          1715.18    1310.025597  405.154403
OUT_02371          1702.91    1301.332138  401.577862

## Quick read
- Top 50 predicted capacity minus bottom 50: 1023.11.
- Top 50 POI influence delta: -0.73.
- Top 50 saturation delta: -0.13.