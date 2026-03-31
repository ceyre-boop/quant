# Liquidity Heatmaps

Understanding market liquidity through visual analysis.

## What are Liquidity Heatmaps?

Liquidity heatmaps show where resting orders are clustered in the order book:
- **Bright areas**: High liquidity (many orders)
- **Dark areas**: Low liquidity (few orders)
- **Gaps**: Potential price acceleration zones

## Data Collection

The system captures liquidity data automatically:

```python
from layer3.game_theory import LiquidityAnalyzer

analyzer = LiquidityAnalyzer()
heatmap = analyzer.capture_heatmap(symbol='NQ')

# Push to Firebase
broadcaster.broadcast_liquidity_heatmap('NQ', heatmap)
```

## Firebase Structure

```
/liquidity_heatmaps/{symbol}/
  ├── current/           # Latest heatmap
  ├── history/           # Historical heatmaps
  └── zones/             # Detected liquidity zones
```

## Heatmap Data Format

```json
{
  "timestamp": "2026-03-11T14:30:00Z",
  "symbol": "NQ",
  "price_range": [18400, 18500],
  "bids": [
    {"price": 18450, "size": 150, "intensity": 0.8},
    {"price": 18445, "size": 80, "intensity": 0.4}
  ],
  "asks": [
    {"price": 18455, "size": 200, "intensity": 0.9},
    {"price": 18460, "size": 50, "intensity": 0.2}
  ],
  "key_levels": {
    "support": [18450, 18430],
    "resistance": [18455, 18480],
    "vacuum_zones": [[18460, 18475]]
  }
}
```

## Trading Applications

### 1. Entry Timing

**Scenario**: You want to go long NQ

```python
heatmap = get_current_heatmap('NQ')

# Check bid liquidity at entry
if heatmap['bids'][0]['intensity'] > 0.7:
    # High liquidity = good fill
    execute_entry()
else:
    # Low liquidity = slippage risk
    wait_or_adjust_entry()
```

### 2. Stop Placement

```python
# Place stops below liquidity clusters
support_levels = heatmap['key_levels']['support']
stop_price = support_levels[0] - 5  # Just below support
```

### 3. Target Selection

```python
# Target resistance levels with high ask liquidity
resistance = heatmap['key_levels']['resistance']
target = resistance[0]  # Where sellers cluster
```

### 4. Vacuum Zone Detection

Vacuum zones = price areas with no liquidity:
- Price moves fast through vacuum
- Good for momentum trades
- Dangerous for mean-reversion

```python
vacuums = heatmap['key_levels']['vacuum_zones']

if vacuums:
    # Price may accelerate if it enters vacuum
    for zone in vacuums:
        if current_price near zone[0]:
            expect_acceleration()
```

## Visual Dashboard

Access liquidity heatmaps at:
```
https://your-firebase-app.web.app/liquidity
```

Features:
- Real-time heatmap view
- Historical playback
- Zone alerts
- Correlation with your signals

## Alerting

Set alerts for liquidity changes:

```python
# Alert when liquidity dries up
if heatmap['bids'][0]['intensity'] < 0.3:
    send_alert(f"Low liquidity on {symbol} - widen stops")

# Alert when vacuum zone forms
if detect_vacuum_formation(heatmap):
    send_alert(f"Vacuum zone on {symbol} - expect volatility")
```

## Historical Analysis

```python
from data.liquidity_archive import LiquidityArchive

archive = LiquidityArchive()

# Get heatmaps for a specific day
heatmaps = archive.get_for_date('2026-03-11', symbol='NQ')

# Analyze liquidity patterns
analyzer = LiquidityPatternAnalyzer()
patterns = analyzer.find_repeating_patterns(heatmaps)

# Use patterns to predict future liquidity
prediction = analyzer.predict_liquidity(
    current_heatmap,
    time_of_day=datetime.now().hour
)
```

## Integration with Trading

The game theory layer (Layer 3) uses liquidity data:

```python
# In layer3/game_theory.py

class GameTheoryEngine:
    def __init__(self):
        self.liquidity = LiquidityAnalyzer()
    
    def generate_output(self, symbol, context):
        heatmap = self.liquidity.get_heatmap(symbol)
        
        # Adjust sizing based on liquidity
        if heatmap['average_intensity'] < 0.4:
            context.position_size *= 0.5  # Reduce size
        
        # Find trapped positions
        trapped = self.liquidity.find_trapped_positions(heatmap)
        
        return GameOutput(
            liquidity_zone=self.liquidity.find_nearest_zone(heatmap, context.price),
            trapped_positions=len(trapped),
            optimal_entry=self.liquidity.find_optimal_entry(heatmap, context.direction)
        )
```

## Best Practices

1. **Don't trade in low liquidity** (< 0.3 intensity)
2. **Place stops outside liquidity clusters**
3. **Use vacuum zones for momentum entries**
4. **Monitor heatmap changes pre-news**
5. **Compare across symbols** for relative strength

## Next Steps

1. Enable liquidity data collection in `config.yaml`
2. Access heatmaps in Firebase console
3. Set up alerts for key liquidity changes
4. Backtest strategies with liquidity filters
