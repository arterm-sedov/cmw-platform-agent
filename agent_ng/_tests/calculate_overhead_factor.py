"""Script to calculate overhead adjustment factor based on API vs estimate data.

This script analyzes the difference between API-reported tokens and our estimates
to determine a heuristic factor for adjusting overhead inflation.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_ng.token_budget import count_tokens, _calculate_avg_tool_size


def analyze_token_data(
    api_total: int,
    context_tokens: int,
    tool_tokens: int,
    overhead_estimate: int,
) -> dict:
    """Analyze token data to calculate overhead adjustment factor.

    Args:
        api_total: Actual API-reported total tokens
        context_tokens: Our estimate of context (conversation) tokens
        tool_tokens: Our estimate of tool result tokens
        overhead_estimate: Our estimate of overhead (tool schemas) tokens

    Returns:
        Dictionary with analysis results
    """
    # Our total estimate
    our_total = context_tokens + tool_tokens + overhead_estimate

    # Difference
    difference = our_total - api_total

    # If API total = context + tools + actual_overhead, then:
    # actual_overhead = api_total - context - tools
    inferred_actual_overhead = api_total - context_tokens - tool_tokens

    # Factor to adjust overhead
    if overhead_estimate > 0:
        overhead_factor = inferred_actual_overhead / overhead_estimate
    else:
        overhead_factor = 1.0

    # Percentage difference
    pct_diff = (difference / api_total * 100) if api_total > 0 else 0

    return {
        "api_total": api_total,
        "our_total_estimate": our_total,
        "difference": difference,
        "difference_pct": pct_diff,
        "context_tokens": context_tokens,
        "tool_tokens": tool_tokens,
        "overhead_estimate": overhead_estimate,
        "inferred_actual_overhead": inferred_actual_overhead,
        "overhead_factor": overhead_factor,
        "recommended_factor": round(overhead_factor, 3),
    }


def main():
    """Main analysis function."""
    # Data from user's example
    data = {
        "api_total": 39_772,
        "context_tokens": 5_448,
        "tool_tokens": 13_576,
        "overhead_estimate": 29_200,
    }

    print("=" * 60)
    print("Token Overhead Analysis")
    print("=" * 60)
    print()
    print(f"API Total (ground truth):     {data['api_total']:,}")
    print(f"Context tokens:                {data['context_tokens']:,}")
    print(f"Tool result tokens:            {data['tool_tokens']:,}")
    print(f"Overhead estimate (schemas):   {data['overhead_estimate']:,}")
    print()

    result = analyze_token_data(**data)

    print("Analysis Results:")
    print("-" * 60)
    print(f"Our total estimate:            {result['our_total_estimate']:,}")
    print(f"Difference:                     {result['difference']:,}")
    print(f"Difference percentage:         {result['difference_pct']:.2f}%")
    print()
    print(f"Inferred actual overhead:      {result['inferred_actual_overhead']:,}")
    print(f"Overhead adjustment factor:     {result['overhead_factor']:.4f}")
    print(f"Recommended factor (rounded):   {result['recommended_factor']}")
    print()

    # Calculate with factor applied
    adjusted_overhead = int(data['overhead_estimate'] * result['overhead_factor'])
    adjusted_total = data['context_tokens'] + data['tool_tokens'] + adjusted_overhead
    adjusted_diff = adjusted_total - data['api_total']
    adjusted_pct = (adjusted_diff / data['api_total'] * 100) if data['api_total'] > 0 else 0

    print("With Factor Applied:")
    print("-" * 60)
    print(f"Adjusted overhead:             {adjusted_overhead:,}")
    print(f"Adjusted total estimate:       {adjusted_total:,}")
    print(f"Adjusted difference:            {adjusted_diff:,}")
    print(f"Adjusted difference %:          {adjusted_pct:.2f}%")
    print()

    # Recommendation
    print("Recommendation:")
    print("-" * 60)
    if abs(adjusted_pct) <= 2.0:
        print(f"✅ Factor {result['recommended_factor']} brings estimate within 2% margin")
        print(f"   Add constant: OVERHEAD_ADJUSTMENT_FACTOR = {result['recommended_factor']}")
    else:
        print(f"⚠️  Factor {result['recommended_factor']} still has {adjusted_pct:.2f}% difference")
        print(f"   Consider refining the factor or investigating other sources of difference")
    print()

    return result


if __name__ == "__main__":
    main()
