# Implementation Guide: Qwen 3.6 Plus + Grok 4.20 Configuration

**Date:** April 23, 2026  
**Configuration:** Qwen 3.6 Plus (default) + Grok 4.20 (fallback)  
**Previous:** GLM 5.1 + Qwen 3.5 Plus

---

## Quick Start

### 1. Update Environment Configuration

Edit your `.env` file:

```bash
# Primary model (85% of tasks)
AGENT_DEFAULT_MODEL=qwen/qwen3.6-plus

# Fallback model (15% of tasks - complex/critical operations)
AGENT_FALLBACK_MODEL=x-ai/grok-4.20

# Provider
AGENT_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
```

### 2. Update llm_configs.py (Already Done)

Both models are already in your configuration:
- ✅ `qwen/qwen3.6-plus` - Added April 23, 2026
- ✅ `x-ai/grok-4.20` - Added April 23, 2026

### 3. Verify Pricing (Already Done)

Pricing already fetched and stored in `agent_ng/openrouter_pricing.json`:
- ✅ Qwen 3.6 Plus: $0.000325/$0.00195 per 1K tokens
- ✅ Grok 4.20: $0.002/$0.006 per 1K tokens

---

## Configuration Benefits

### vs Previous (GLM 5.1 + Qwen 3.5)

| Metric | Previous | New | Improvement |
|--------|----------|-----|-------------|
| **Default Cost** | $3.20/M | $1.30/M | ✅ -59% |
| **Default SWE-bench** | Not disclosed | 78.8% | ✅ Verified |
| **Default Error Recovery** | Poor (21-0%) | Good | ✅ Better |
| **Fallback Context** | 1M tokens | 2M tokens | ✅ +100% |
| **Fallback Architecture** | Single model | Multi-agent | ✅ Smarter |
| **Blended Cost (85/15)** | $2.82/M | $1.77/M | ✅ -37% |
| **Monthly Cost** | $141 | $88.25 | ✅ -$52.75 |

---

## When to Use Each Model

### Qwen 3.6 Plus (Default - 85% of tasks)

**Use for:**
- ✅ Routine CMW Platform operations
- ✅ Template and attribute management
- ✅ Record CRUD operations
- ✅ Form/dataset/toolbar configuration
- ✅ Multi-step API orchestration
- ✅ Exploratory queries
- ✅ Batch operations

**Strengths:**
- 78.8% SWE-bench Verified (best coding performance)
- $1.30/M (cost-effective)
- 1M context window (sufficient for most tasks)
- Native chain-of-thought reasoning
- Good error recovery

### Grok 4.20 (Fallback - 15% of tasks)

**Use for:**
- ✅ Complex multi-step workflows requiring verification
- ✅ Large context needs (>1M tokens)
- ✅ Critical operations where errors are costly
- ✅ Complex architectural decisions
- ✅ Debugging complex issues
- ✅ Parallel tool execution
- ✅ When Qwen fails or produces uncertain results

**Strengths:**
- Multi-agent architecture (4-16 specialized agents)
- 2M context window (largest available)
- 78% non-hallucination rate (lowest hallucination)
- Parallel tool execution
- Best for high-stakes operations

---

## Fallback Trigger Strategy

### Automatic Triggers (Recommended)

Implement smart routing in your agent logic:

```python
def select_model(context_size: int, complexity: str, is_critical: bool) -> str:
    """
    Smart model selection for CMW Platform Agent.
    
    Args:
        context_size: Total tokens in context
        complexity: "low", "medium", "high"
        is_critical: True for production-critical operations
    
    Returns:
        Model identifier string
    """
    # Trigger Grok 4.20 for large context
    if context_size > 1_000_000:
        return "x-ai/grok-4.20"
    
    # Trigger Grok 4.20 for critical operations
    if is_critical:
        return "x-ai/grok-4.20"
    
    # Trigger Grok 4.20 for high complexity
    if complexity == "high":
        return "x-ai/grok-4.20"
    
    # Default to Qwen 3.6 Plus
    return "qwen/qwen3.6-plus"
```

### Manual Triggers

Allow users to force Grok 4.20 via UI:
- Checkbox: "Use advanced reasoning (Grok 4.20)"
- Automatic for operations tagged as "critical"
- Automatic when context exceeds 1M tokens

---

## Cost Monitoring

### Target Metrics

**Daily Monitoring:**
- Qwen 3.6 Plus usage: Target 85% of conversations
- Grok 4.20 usage: Target 15% of conversations
- Average cost per conversation: Target <$0.10

**Weekly Monitoring:**
- Total cost: Target <$25/week
- Grok 4.20 trigger rate: Should be 10-20%
- Cost per 1M tokens: Should be $1.50-$2.00 blended

**Monthly Budget:**
- Target: $88/month (1,000 conversations)
- Alert threshold: $100/month
- Critical threshold: $150/month

### Cost Tracking Code

```python
def track_model_usage(model: str, input_tokens: int, output_tokens: int):
    """Track model usage and costs"""
    pricing = {
        "qwen/qwen3.6-plus": {"input": 0.000325, "output": 0.00195},
        "x-ai/grok-4.20": {"input": 0.002, "output": 0.006}
    }
    
    cost = (
        (input_tokens / 1000) * pricing[model]["input"] +
        (output_tokens / 1000) * pricing[model]["output"]
    )
    
    # Log to monitoring system
    log_metric("model_usage", {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "timestamp": datetime.now()
    })
    
    return cost
```

---

## Validation Checklist

### Week 1: Initial Validation

- [ ] **Day 1:** Update `.env` configuration
- [ ] **Day 1:** Restart agent and verify model loading
- [ ] **Day 2:** Test Qwen 3.6 Plus with 10 routine operations
- [ ] **Day 3:** Test Grok 4.20 with 3 complex workflows
- [ ] **Day 4:** Monitor cost per conversation
- [ ] **Day 5:** Compare error rates vs GLM 5.1 baseline
- [ ] **Day 6-7:** Collect user feedback

### Week 2: Performance Validation

- [ ] Run full test suite with Qwen 3.6 Plus
- [ ] Test all 49 CMW Platform tools
- [ ] Validate tool calling reliability
- [ ] Test error recovery scenarios
- [ ] Measure response quality vs previous models
- [ ] Validate streaming performance
- [ ] Test large context scenarios (>500K tokens)

### Week 3: Cost Optimization

- [ ] Analyze Grok 4.20 usage patterns
- [ ] Identify tasks that benefit most from Grok 4.20
- [ ] Fine-tune fallback triggers
- [ ] Optimize context window usage
- [ ] Validate cost projections vs actual usage
- [ ] Adjust budget alerts if needed

### Week 4: Production Readiness

- [ ] Document best practices for model selection
- [ ] Create user guide for when to use each model
- [ ] Set up automated cost monitoring
- [ ] Configure budget alerts
- [ ] Train team on new configuration
- [ ] Plan for monthly cost reviews

---

## Troubleshooting

### Issue: Grok 4.20 usage exceeds 20%

**Symptoms:** Monthly costs higher than projected

**Solutions:**
1. Review fallback trigger logic
2. Increase complexity threshold for Grok 4.20
3. Validate that Qwen 3.6 Plus is handling routine tasks
4. Check for unnecessary large context usage

### Issue: Qwen 3.6 Plus error rate higher than expected

**Symptoms:** Frequent failures on routine operations

**Solutions:**
1. Review error logs for patterns
2. Check if tasks are too complex for Qwen
3. Lower complexity threshold for Grok 4.20 fallback
4. Validate tool schemas and parameters

### Issue: Response quality degraded vs GLM 5.1

**Symptoms:** User complaints about response quality

**Solutions:**
1. Collect specific examples of quality issues
2. Test same queries with both models
3. Consider adjusting system prompts for Qwen 3.6 Plus
4. Increase Grok 4.20 usage for affected task types

### Issue: Context window exceeded on Qwen 3.6 Plus

**Symptoms:** Errors about context limit (1M tokens)

**Solutions:**
1. Implement automatic fallback to Grok 4.20 (2M context)
2. Enable history compression earlier
3. Review context usage patterns
4. Optimize prompt templates to reduce token usage

---

## Performance Benchmarks

### Expected Performance (Based on Research)

| Metric | Qwen 3.6 Plus | Grok 4.20 |
|--------|---------------|-----------|
| **SWE-bench Verified** | 78.8% | Not disclosed |
| **Tool Calling** | Native CoT | Multi-agent |
| **Context Window** | 1M tokens | 2M tokens |
| **Speed** | ~45 tok/s | 162-233 tok/s |
| **TTFT** | ~2s | 354ms (10.33s reasoning) |
| **Error Recovery** | Good | Excellent |
| **Hallucination Rate** | Low | Lowest (78% non-hallucination) |

### CMW Agent-Specific Benchmarks (To Measure)

Track these metrics during validation:

1. **Tool Calling Success Rate**
   - Target: >95% for both models
   - Measure: Successful tool calls / Total tool calls

2. **Multi-Step Workflow Success**
   - Target: >90% for Qwen, >95% for Grok
   - Measure: Completed workflows / Total workflows

3. **Error Recovery Rate**
   - Target: >80% for Qwen, >90% for Grok
   - Measure: Recovered errors / Total errors

4. **Response Quality Score**
   - Target: >4.0/5.0 for both models
   - Measure: User ratings (1-5 scale)

5. **Cost per Successful Operation**
   - Target: <$0.05 for Qwen, <$0.15 for Grok
   - Measure: Total cost / Successful operations

---

## Migration from GLM 5.1 + Qwen 3.5

### Step-by-Step Migration

**Phase 1: Preparation (Day 1)**
1. Backup current `.env` configuration
2. Document current GLM 5.1 performance metrics
3. Export conversation history for comparison
4. Set up cost monitoring dashboard

**Phase 2: Gradual Rollout (Week 1)**
1. Update configuration for 10% of users
2. Monitor performance and costs
3. Collect feedback from early adopters
4. Adjust fallback triggers if needed

**Phase 3: Full Rollout (Week 2)**
1. Update configuration for all users
2. Monitor system-wide metrics
3. Compare performance vs GLM 5.1 baseline
4. Document lessons learned

**Phase 4: Optimization (Week 3-4)**
1. Fine-tune model selection logic
2. Optimize cost efficiency
3. Update documentation
4. Train team on best practices

### Rollback Plan

If issues arise, rollback to previous configuration:

```bash
# Rollback to GLM 5.1 + Qwen 3.5
AGENT_DEFAULT_MODEL=z-ai/glm-5.1
AGENT_FALLBACK_MODEL=qwen/qwen-plus-2025-07-28
```

**Rollback Triggers:**
- Error rate increases >20% vs baseline
- Cost exceeds $150/month
- User satisfaction drops significantly
- Critical production issues

---

## Success Criteria

### Week 1 Success Metrics
- ✅ Configuration deployed without errors
- ✅ Qwen 3.6 Plus handles 80%+ of tasks
- ✅ No critical production issues
- ✅ Cost tracking operational

### Month 1 Success Metrics
- ✅ Cost reduction: -30% vs GLM 5.1 configuration
- ✅ Error rate: Equal or better than GLM 5.1
- ✅ User satisfaction: Equal or better than GLM 5.1
- ✅ Grok 4.20 usage: 10-20% of conversations

### Month 3 Success Metrics
- ✅ Stable cost: $80-$100/month
- ✅ Optimized fallback triggers
- ✅ Documented best practices
- ✅ Team trained on model selection

---

## Additional Resources

### Documentation
- Full research report: `docs/research/CMW_Agent_Model_Selection_Report_20260423.md`
- Model configurations: `agent_ng/llm_configs.py`
- Pricing data: `agent_ng/openrouter_pricing.json`

### OpenRouter Model Pages
- Qwen 3.6 Plus: https://openrouter.ai/qwen/qwen3.6-plus
- Grok 4.20: https://openrouter.ai/x-ai/grok-4.20

### Support
- OpenRouter Discord: https://discord.gg/fVyRaUDgxW
- OpenRouter Status: https://status.openrouter.ai
- OpenRouter Docs: https://openrouter.ai/docs

---

**Last Updated:** April 23, 2026  
**Configuration Status:** Ready for deployment  
**Estimated Savings:** -37% vs previous configuration ($52.75/month)
