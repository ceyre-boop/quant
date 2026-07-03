"""
Quick test of ClawdBrain v3.1 with Chi-Squared Gate 6
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from clawd_brain_v31 import ClawdBrain, EntryGates, Layer1Output, Layer2Output

# Test Gate 6 directly
print("="*60)
print("Testing Gate 6: Chi-Squared Validation")
print("="*60)

gates = EntryGates(min_confidence=0.55, use_chi2=True)

# Mock layer outputs
layer1 = Layer1Output(
    symbol="SPY", direction="LONG", bias=0.3, 
    confidence=0.65, win_prob=0.65, features={}
)
layer2 = Layer2Output(
    symbol="SPY", direction="LONG", kelly_fraction=0.05,
    game_theory_score=0.7, edge=0.01, max_position_size=10000,
    recommended_shares=10, avg_win=0.015, avg_loss=0.010
)

result = gates.check(layer1, layer2, 100000)

print(f"\nTest Trade: SPY LONG")
print(f"  Confidence: {layer1.confidence}")
print(f"  Kelly Fraction: {layer2.kelly_fraction}")
print(f"  GT Score: {layer2.game_theory_score}")
print()
print(f"  Result: {'PASSED' if result.passed else 'REJECTED'}")
if result.reject_reason:
    print(f"  Reason: {result.reject_reason}")
print(f"  Chi2 Result: {result.chi2_result}")
print()

# Since we have no history, chi2 bucket won't be validated yet
print("Note: Chi-squared gate needs 30+ trade samples to validate buckets.")
print("      Initially uses confidence threshold until enough data collected.")
