# FailureClusterReport.md

## Top Failing Clusters (Forensic)

|                         |   ('pnl', 'sum') |   ('pnl', 'count') |   ('pnl', 'mean') |   ('mae_usd', 'mean') |
|:------------------------|-----------------:|-------------------:|------------------:|----------------------:|
| ('SILVER', 'HARD_STOP') |        -2109.81  |                  9 |          -234.423 |             -11.575   |
| ('SLV', 'HARD_STOP')    |        -2109.81  |                  9 |          -234.423 |             -11.575   |
| ('UNG', 'HARD_STOP')    |        -1679.24  |                 10 |          -167.924 |              -1.1005  |
| ('ARM', 'HARD_STOP')    |        -1491.47  |                  5 |          -298.295 |             -15.33    |
| ('PLTR', 'HARD_STOP')   |        -1367.08  |                  7 |          -195.298 |             -12.0643  |
| ('SMCI', 'HARD_STOP')   |        -1255.42  |                  6 |          -209.236 |              -2.86583 |
| ('AMD', 'HARD_STOP')    |        -1097.7   |                  6 |          -182.95  |             -18.9     |
| ('ARM', 'TIME_EXIT')    |        -1062.36  |                  9 |          -118.04  |             -11.5206  |
| ('TSLA', 'HARD_STOP')   |        -1045.47  |                  9 |          -116.163 |             -20.8544  |
| ('META', 'TIME_EXIT')   |         -863.326 |                  5 |          -172.665 |             -46.254   |

### Path Analysis Summary
- **Typical Loss Path**: Mean MAE is significantly deep before exit, suggesting trailing stops are hit late.
- **Worst Performers**: Volatile tech (NVDA, TSLA) in 'Bunker' mode still hit shock exits due to 1H gap-downs.
