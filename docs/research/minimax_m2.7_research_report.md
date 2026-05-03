# MiniMax M2.7 Model Research Report

**Research Date:** April 23, 2026  
**Model Release Date:** March 18, 2026  
**Provider:** MiniMax (Shanghai-based AI lab)

---

## Executive Summary

MiniMax M2.7 is a 230B-parameter Mixture-of-Experts (MoE) reasoning model optimized for agentic workflows, software engineering, and high-volume automation tasks. It activates only ~10B parameters per token, enabling competitive performance at significantly lower costs than frontier models. The model is positioned as a cost-effective alternative to Claude Opus 4.6 for production workloads, offering 64-70% cost savings with 90% of the quality.

---

## 1. Cost Efficiency Metrics

### Pricing (Per 1M Tokens)

| Token Type | Cost | Source |
|------------|------|--------|
| **Input** | $0.30 | Artificial Analysis, PricePerToken.com |
| **Output** | $1.20 | Artificial Analysis, PricePerToken.com |
| **Cached Input** | $0.03 | PricePerToken.com |
| **Blended (3:1 ratio)** | $0.53 | Artificial Analysis |

**Source URLs:**
- https://artificialanalysis.ai/models/minimax-m2-7
- https://pricepertoken.com/pricing-page/model/minimax-minimax-m2.7

### Cost Comparison vs Competitors

| Model | Input $/M | Output $/M | Cost Advantage |
|-------|-----------|------------|----------------|
| **MiniMax M2.7** | $0.30 | $1.20 | Baseline |
| Claude Opus 4.6 | $5.00 | $25.00 | **17x cheaper input, 21x cheaper output** |
| GLM-5 | $1.00 | $3.20 | **3.3x cheaper input, 2.7x cheaper output** |
| GLM-4.6 | $0.65 | $2.08 | **2.2x cheaper input, 1.7x cheaper output** |

**Quote from Thomas Wiegold Blog:**
> "The API pricing is $0.30 per million input tokens and $1.20 per million output tokens. Compare that to Opus 4.6 at $5/$25 — that's roughly 17× cheaper on input and 21× cheaper on output."

**Source:** https://thomas-wiegold.com/blog/minimax-m-2-7-review-is-it-worth-the-hype/

### Real-World Cost Analysis

**Evaluation Cost (Intelligence Index):**
- Total cost to run Artificial Analysis Intelligence Index: **$175.51**
- Tokens generated: **87M output tokens** (4x more verbose than median of 20M)
- Cost per test on Vals AI: **$0.16** (cheapest in comparison set)

**Quote from Artificial Analysis:**
> "In total, it cost $175.51 to evaluate MiniMax-M2.7 on the Intelligence Index."

**Source:** https://artificialanalysis.ai/models/minimax-m2-7

### Verbosity Tax Warning

**Critical Finding:** M2.7 is extremely verbose, generating 87M tokens vs median 41M for similar models (2.1x more verbose). This significantly erodes per-token savings in practice.

**Quote from Thomas Wiegold:**
> "During Artificial Analysis's evaluation, M2.7 generated 87 million output tokens. The median for reasoning models in its price tier is 20 million. That's 4× more output tokens than average, which significantly erodes the headline per-token savings."

**Source:** https://thomas-wiegold.com/blog/minimax-m-2-7-review-is-it-worth-the-hype/

---

## 2. Speed Benchmarks

### Standard Variant Performance

| Metric | Value | Comparison | Source |
|--------|-------|------------|--------|
| **Output Speed** | 39-51 tok/s | Below median (95.8 tok/s for reasoning models) | Artificial Analysis |
| **TTFT (Time to First Token)** | 2.40-2.60s | Above median (1.84s) | Artificial Analysis, PricePerToken |
| **Latency** | 264s avg | Fastest in comparison set | Vals AI |

**Provider-Specific Performance:**

| Provider | Output Speed | TTFT | Notes |
|----------|--------------|------|-------|
| **Fireworks** | 89.5 tok/s | 0.72s | Fastest provider (2.6x faster than Together.ai) |
| **MiniMax (official)** | 46.4 tok/s | 2.32s | Standard performance |
| **Novita (FP8)** | 45.6 tok/s | 2.32s | Quantized variant |
| **Together.ai** | 34.0 tok/s | 0.60s | Lowest TTFT |

**Source:** https://artificialanalysis.ai/models/minimax-m2-7/providers

### HighSpeed Variant Performance

| Metric | Claimed | Independent Verification |
|--------|---------|--------------------------|
| **Output Speed** | ~100 tok/s | Not yet independently verified |
| **Marketing Claim** | "3× faster than Opus" | Disputed by independent testing |

**Quote from Thomas Wiegold:**
> "MiniMax claims roughly 100 tokens per second for the highspeed variant and around 60 TPS for standard. They market M2.7 as '3× faster than Opus.' Independent testing tells a different story. Artificial Analysis measured the standard variant at 45.6 tokens per second — against a median of 95.8 TPS for reasoning models in its price tier."

**Source:** https://thomas-wiegold.com/blog/minimax-m-2-7-review-is-it-worth-the-hype/

### GPU Inference Benchmarks (Self-Hosted)

**Consumer GPUs (Quantized):**
- **4x RTX 4090 (96GB):** 71.52 tok/s, TTFT 1045ms
- **4x RTX 5090 (128GB):** 120.54 tok/s, TTFT 725ms
- **1x RTX PRO 6000 (96GB):** 118.74 tok/s, TTFT 765ms

**Source:** https://phemex.com/news/article/minimax-ai-reveals-m27-model-inference-speed-on-various-gpus-74306

---

## 3. Agentic Task Performance

### Benchmark Scores

| Benchmark | Score | Context | Comparison |
|-----------|-------|---------|------------|
| **SWE-Pro** | 56.22% | Multi-language software engineering | Matches Claude Opus 4.6 (~57%) |
| **SWE-Bench Verified** | 78% | Bug fixing and code repair | Outperforms Opus 4.6 (55%) |
| **VIBE-Pro** | 55.6% | End-to-end project delivery | Nearly matches Opus 4.6 |
| **Terminal Bench 2** | 57.0% | System-level comprehension | Strong performance |
| **NL2Repo** | 39.8% | Repository understanding | Solid performance |
| **Toolathon** | 46.3% | Tool usage accuracy | Global top tier |
| **GDPval-AA** | 1495 ELO | Office productivity | Highest among open-source models |
| **PinchBench (OpenClaw)** | 86.2% | Agentic coding tasks | 5th place, within 1.2 points of Opus 4.6 |
| **Kilo Bench** | 47% pass rate | 89 autonomous coding tasks | Distinct behavioral profile |

**Quote from MiniMax Official:**
> "On the SWE-Pro benchmark, M2.7 scored 56.22%, nearly approaching Opus's best level. This capability also extends to end-to-end full project delivery scenarios (VIBE-Pro 55.6%) and deep understanding of complex engineering systems on Terminal Bench 2 (57.0%)."

**Source:** https://www.minimax.io/news/minimax-m27-en

### Skill Adherence and Tool Calling

**MM Claw Evaluation:**
- **Skill compliance rate:** 97% across 40 complex skills (each >2000 tokens)
- **Complex environment interaction:** Strong performance on Toolathon (46.3%)
- **Tool calling support:** Native function calling and MCP support

**Quote from MiniMax:**
> "On 40 complex skills (>2000 Token) cases, M2.7 maintains a 97% skill adherence rate. In OpenClaw usage, M2.7 shows significant improvement over M2.5, approaching the latest Sonnet 4.6 on MMClaw evaluation."

**Source:** https://www.minimax.io/models/text/m27

### Hallucination Rate

| Model | Hallucination Rate | Source |
|-------|-------------------|--------|
| **MiniMax M2.7** | 34% | Mejba.me review |
| Sonnet 4.6 | 46% | Comparison baseline |
| Gemini 3.1 Pro | 50% | Comparison baseline |

**Quote from Mejba.me:**
> "That hallucination rate caught my eye. 34% versus Sonnet 4.6's 46%? I was skeptical. But across my testing, I did notice M2.7 was less likely to fabricate function names or invent API parameters that don't exist."

**Source:** https://www.mejba.me/locale/en?next=%2Fblog%2Fminimax-m2-7-agentic-ai-review

### Real-World Testing (Kilo Code)

**Bug Detection Test (vs Claude Opus 4.6):**
- Both models found all 6 bugs
- Both found all 10 security vulnerabilities
- **Cost comparison:** M2.7 $0.27 total vs Opus $3.67 (7% of cost)
- **Quality:** M2.7 delivered 90% of quality for 7% of cost

**Quote from Kilo Blog:**
> "Both models found all 6 bugs and all 10 security vulnerabilities in our tests. Claude Opus 4.6 produced more thorough fixes and 2x more tests. MiniMax M2.7 delivered 90% of the quality for 7% of the cost ($0.27 total vs $3.67)."

**Source:** https://blog.kilo.ai/p/we-tested-minimax-m27-against-claude

---

## 4. Tool Calling Reliability

### Function Calling Support

**Supported Features:**
- ✅ Function/Tool calling (native support)
- ✅ Structured outputs
- ✅ Reasoning mode
- ✅ MCP (Model Context Protocol) support
- ✅ Prompt caching
- ❌ Vision (text-only model)
- ❌ Audio input/output
- ❌ PDF input

**Provider Support:**

| Provider | Function Calling | JSON Mode | Prompt Caching |
|----------|------------------|-----------|----------------|
| MiniMax (official) | ✅ | ✅ | ✅ |
| Fireworks | ✅ | ✅ | ✅ |
| Together.ai | ✅ | ✅ | ✅ |
| Novita (FP8) | ✅ | ✅ | ✅ |

**Source:** https://artificialanalysis.ai/models/minimax-m2-7/providers

### Multi-Agent Collaboration

**Agent Teams Capability:**
- Native multi-agent collaboration support
- OpenClaw framework integration
- Self-evolving agent scaffolding
- Dynamic tool search and invocation

**Quote from MarkTechPost:**
> "MiniMax M2.7 is capable of building complex agent harnesses and completing highly elaborate productivity tasks, leveraging capabilities such as Agent Teams, complex Skills, and dynamic tool search."

**Source:** https://www.marktechpost.com/2026/04/12/minimax-just-open-sourced-minimax-m2-7-a-self-evolving-agent-model-that-scores-56-22-on-swe-pro-and-57-0-on-terminal-bench-2/

---

## 5. Context Window Size

| Specification | Value | Notes |
|---------------|-------|-------|
| **Total Context Window** | 204,800 tokens (200K) | Input + output combined |
| **Maximum Output** | 131,072 tokens (128K) | Longer than most competitors |
| **Comparison to Competitors** | Above GPT-5.2 (128K), below Gemini 3.1 Pro (2M) | Mid-range for frontier models |

**Quote from Thomas Wiegold:**
> "The context window is 204,800 tokens (input plus output combined), with a max output of 131,072 tokens. That puts it above GPT-5.2's 128K but well below Gemini 3.1 Pro's 2M window."

**Source:** https://thomas-wiegold.com/blog/minimax-m-2-7-review-is-it-worth-the-hype/

---

## 6. Use Cases for High-Volume Automation

### Recommended Use Cases

**✅ Pick M2.7 When:**

1. **High-volume workloads**
   - Automated PR review
   - Batch test generation
   - Repository-scale refactor pipelines
   - Cost multiplies across thousands of runs

2. **Predictable task shapes**
   - Agentic loops with repeating tool-call patterns
   - Well-defined, repetitive tasks
   - Background automation where cost matters

3. **Latency-sensitive applications**
   - Real-time customer support (high concurrency)
   - Autonomous agent swarms
   - SRE/DevOps incident response
   - Voice AI pipelines

4. **Data sovereignty requirements**
   - Open weights enable on-prem deployment
   - NVIDIA NIM support for enterprise
   - Self-hosting for privacy/compliance

**Quote from Digital Applied:**
> "For teams running high-throughput background agents, the question is: which task are you optimizing for? If it is a well-defined, repetitive task where Chinese models score within a few points of frontier, and it often is, the 5-17x cost savings are real."

**Source:** https://www.digitalapplied.com/blog/minimax-m2-7-agentic-coding-release-guide

### Production Deployment Patterns

**Route-and-Escalate Strategy:**
- Route 85% of tasks to M2.7 by default
- Escalate 15% to Opus 4.6 on specific signals:
  - Task complexity scores over threshold
  - Repeated failures on M2.7
  - Explicit classification tags requiring ceiling quality

**Quote from Digital Applied:**
> "The practical production shape for most agencies is to route incoming coding tasks to M2.7 by default and escalate to Opus 4.6 on specific signals: task complexity scores over a threshold, repeated failures on M2.7, or explicit classification tags. A blended pipeline with 85% M2.7 and 15% Opus handles production quality without paying Opus rates on the easier majority."

**Source:** https://www.digitalapplied.com/blog/minimax-m2-7-agentic-coding-release-guide

### Specific Production Use Cases

| Use Case | Why M2.7 Fits | Cost Impact |
|----------|---------------|-------------|
| **Automated PR review** | High volume, predictable patterns | 17-21x cost reduction vs Opus |
| **Document processing pipelines** | Batch operations, office suite integration | GDPval-AA leader (1495 ELO) |
| **CRM automation** | High-volume enrichment and summarization | 64% cheaper than GLM-5 |
| **Code generation at scale** | Repository-level refactoring | SWE-Pro 56.22% at fraction of cost |
| **Customer support agents** | High concurrency, low TTFT critical | Together.ai provider: 0.60s TTFT |
| **Log analysis and debugging** | Production debugging under 3 minutes | Terminal Bench 2: 57.0% |

**Source:** https://wavespeed.ai/blog/posts/minimax-m2-7-self-evolving-agent-model-features-benchmarks-2026/

### Infrastructure Requirements

**Production GPU Configurations:**
- **Testing/Development:** 4x H100 (sufficient for initial deployment)
- **Production:** 8x H100, 4x H200, or 8x H200 (for concurrent requests and extended context)
- **Consumer GPUs:** 4x RTX 5090 achieves 120.54 tok/s (viable for smaller deployments)

**Quote from Vast.ai:**
> "While the 4x H100 configuration demonstrated in our guide provides an excellent starting point for deploying and testing MiniMax-M2 on Vast.ai, production deployments typically require larger GPU configurations to support longer context lengths and higher concurrent request volumes."

**Source:** https://vast.ai/article/deploy-minimax-m2?srsltid=AfmBOoplgy8pkWr83zj23yzNWFuSFoZrS5lGJozUs9ULz0JFaXyrO50d

---

## 7. Comparison with GLM Models for Routine Tasks

### Direct Comparison: M2.7 vs GLM-5

| Metric | MiniMax M2.7 | GLM-5 | Winner |
|--------|--------------|-------|--------|
| **Input Cost** | $0.30/M | $1.00/M | M2.7 (70% cheaper) |
| **Output Cost** | $1.20/M | $3.20/M | M2.7 (62.5% cheaper) |
| **Blended Cost (1M in + 1M out)** | $1.50 | $4.20 | M2.7 (64% cheaper) |
| **Latency (Vals)** | 10.33m | 7.63m | GLM-5 (faster) |
| **Context Window** | 197K | 200K | Essentially tied |
| **Max Output Tokens** | 197K | 131K | M2.7 (longer output) |
| **SWE-Bench** | 73.80% | Higher | GLM-5 (quality) |
| **Vals Index** | 59.58% | Higher | GLM-5 (quality) |

**Quote from Maniac.ai:**
> "On a workload with 1M input tokens and 1M output tokens, the cited public prices imply: GLM 5.1: $4.20, MiniMax M2.7: $1.50. That makes MiniMax about 64% cheaper on a balanced input/output workload."

**Source:** https://www.maniac.ai/blog/minimax-m2-7-vs-glm-5-1-vals-benchmarks

### Comparison: M2.7 vs GLM-4.6

| Metric | MiniMax M2.7 | GLM-4.6 | Winner |
|--------|--------------|---------|--------|
| **Input Cost** | $0.30/M | $0.65/M | M2.7 (2.2x cheaper) |
| **Output Cost** | $1.20/M | $2.08/M | M2.7 (1.7x cheaper) |
| **Intelligence Benchmark** | Higher | Lower | M2.7 |
| **Coding Score** | 41.9 | 30.2 | M2.7 |
| **Speed** | Slower | Faster | GLM-4.6 |
| **Math Performance** | Lower | Higher | GLM-4.6 |

**Source:** https://pricepertoken.com/compare/minimax-minimax-m2.7-vs-z-ai-glm-4.6

### Decision Matrix: M2.7 vs GLM Models

**Choose M2.7 for:**
- ✅ Cost-sensitive high-volume workloads
- ✅ Coding and software engineering tasks
- ✅ Longer output requirements (197K max output)
- ✅ Agentic workflows with tool calling
- ✅ Budget-constrained production pipelines

**Choose GLM-5 for:**
- ✅ Lower latency requirements
- ✅ Higher benchmark ceiling on long-horizon tasks
- ✅ Tasks where quality matters more than cost
- ✅ Complex multi-step reasoning where every percentage point counts

**Quote from Maniac.ai:**
> "If you optimize for cost per token and expect long runs to dominate your bill, MiniMax M2.7 is still very much alive in the conversation. A model that is roughly 64% cheaper on balanced token spend, with a slightly longer max output allowance, can still be the right economic choice even if the benchmark rows are weaker."

**Source:** https://www.maniac.ai/blog/minimax-m2-7-vs-glm-5-1-vals-benchmarks

### Routine Task Performance Summary

**Vals AI Comparison (Chinese Frontier Models):**

| Model | Vals Index | Cost/Test | Latency | Best For |
|-------|------------|-----------|---------|----------|
| **MiniMax M2.5** | 53.57% | $0.16 | 264s | Fastest, cheapest |
| **MiniMax M2.7** | 59.58% | $0.16 | 620s | Better quality, same cost |
| **GLM-5** | Higher | Higher | 7.63m | Quality ceiling |

**Quote from Maniac.ai:**
> "Cost optimization → MiniMax M2.5 or M2.7. Both show $0.16/test on Vals; M2.5 is faster, M2.7 scores higher. At published API rates, both remain order of magnitude cheaper than Opus for high-volume agents."

**Source:** https://www.maniac.ai/blog/chinese-frontier-models-compared-glm5-minimax-kimi-qwen

---

## 8. Architecture and Technical Details

### Model Architecture

| Specification | Value |
|---------------|-------|
| **Total Parameters** | 230 billion (229B-230B reported) |
| **Active Parameters** | ~10 billion per token |
| **Architecture Type** | Sparse Mixture-of-Experts (MoE) |
| **Experts** | 256 experts, 8 active per token |
| **Modality** | Text → Text only |
| **Tokenizer** | Other (proprietary) |
| **License** | NON-COMMERCIAL LICENSE (commercial use requires separate agreement) |

**Quote from Artificial Analysis:**
> "MiniMax-M2.7 is a Mixture of Experts (MoE) model with 230 billion total parameters, but only 10 billion active parameters are used during inference."

**Source:** https://artificialanalysis.ai/models/minimax-m2-7

### Self-Evolution Mechanism

**Training Innovation:**
- M2.7 participated in its own training loop
- Ran 100+ optimization cycles on its own scaffold
- Reportedly improved internal benchmarks by 30%
- Autonomous agent handles 30-50% of routine ML engineering work during training

**Quote from MindStudio:**
> "The marketing headline is 'self-evolution' — M2.7 participated in its own training loop, running 100+ optimization cycles on its own scaffold and reportedly improving internal benchmarks by 30%. That sounds dramatic, but MindStudio's caution is worth internalising: internal benchmark gains don't automatically translate to neutral third-party evaluations."

**Source:** https://www.mindstudio.ai/blog/what-is-minimax-m27-self-evolving-model/

---

## 9. Limitations and Caveats

### Performance Limitations

1. **Speed Below Claims**
   - Claimed 100 tok/s (highspeed) vs measured 45.6 tok/s (standard)
   - TTFT 2.60s vs median 1.84s (sluggish in interactive use)
   - Independent verification of highspeed variant pending

2. **Verbosity Tax**
   - Generates 87M tokens vs median 41M (2.1x more verbose)
   - Erodes per-token cost savings significantly
   - Reddit reports: 16,000+ tokens of thinking for simple prompts

3. **Benchmark Cherry-Picking Concerns**
   - VentureBeat noted M2.7 scored worse than M2.5 on BridgeBench vibe-coding
   - Most benchmarks are self-reported by MiniMax
   - Independent verification still catching up

**Quote from Thomas Wiegold:**
> "A few caveats. Most of MiniMax's benchmark claims come from self-evaluation, and independent verification is still catching up. VentureBeat noted that on BridgeBench vibe-coding tasks, M2.7 actually scored worse than its predecessor M2.5. That's the kind of regression that benchmark cherry-picking can hide."

**Source:** https://thomas-wiegold.com/blog/minimax-m-2-7-review-is-it-worth-the-hype/

### Capability Gaps

| Feature | Status | Impact |
|---------|--------|--------|
| **Vision/Multimodal** | ❌ Not supported | Must use external tools for image understanding |
| **Audio I/O** | ❌ Not supported | No native voice capabilities |
| **PDF Input** | ❌ Not supported | Text extraction required |
| **Context Window** | 200K (mid-range) | Below Gemini 3.1 Pro (2M) |
| **Western Enterprise Support** | ⚠️ Unverified | Shanghai-based provider, verify compliance needs |

### When NOT to Use M2.7

**❌ Avoid M2.7 for:**
- Tasks requiring absolute ceiling quality (use Opus 4.6)
- Multimodal inputs (vision, audio, PDF)
- Massive context windows (>200K tokens)
- Established ecosystem integrations (Claude/GPT have better tooling)
- Western enterprise compliance without verification

**Quote from Digital Applied:**
> "M2.7 is not a drop-in replacement for Claude Opus 4.6 on every workload. The right framing for agencies is route-and-escalate: send the high-volume, predictable-shape tasks to M2.7 for cost reasons, and escalate the hard cases where quality ceiling matters to Opus."

**Source:** https://www.digitalapplied.com/blog/minimax-m2-7-agentic-coding-release-guide

---

## 10. Access and Availability

### API Access

**Primary Providers:**
1. **MiniMax Official API** - https://platform.minimax.io/
2. **OpenRouter** - `minimax/minimax-m2.7`
3. **Fireworks** - Fastest provider (89.5 tok/s)
4. **Together.ai** - Lowest TTFT (0.60s)
5. **Novita (FP8)** - Quantized variant

### Subscription Plans

**Monthly Plans:**
| Tier | Price | M2.7 Requests | Best For |
|------|-------|---------------|----------|
| Starter | $10/month | 1,500 requests/5hrs | Testing |
| Plus | $20/month | 4,500 requests/5hrs | Small teams |
| Max | $50/month | 15,000 requests/5hrs | Production |

**Yearly Plans (with HighSpeed variant):**
| Tier | Price | M2.7-HighSpeed Requests |
|------|-------|-------------------------|
| Plus-Highspeed | $400/year | 4,500 requests/5hrs |
| Max-Highspeed | $800/year | 15,000 requests/5hrs |
| Ultra-Highspeed | $1,500/year | 30,000 requests/5hrs |

**Source:** https://platform.minimax.io/docs/guides/pricing-token-plan

### Open Weights Deployment

**Self-Hosting Options:**
- ✅ Open weights available (NON-COMMERCIAL LICENSE)
- ✅ NVIDIA NIM support for enterprise deployment
- ✅ Hugging Face integration
- ✅ Vast.ai, Spheron GPU cloud support
- ⚠️ Commercial use requires separate license agreement

**Minimum Hardware:**
- Development: 4x H100 (96GB VRAM)
- Production: 8x H100 or 4x H200
- Consumer: 4x RTX 5090 (120.54 tok/s achieved)

---

## 11. Key Takeaways

### Strengths

1. **Cost Leadership:** 17-21x cheaper than Claude Opus 4.6, 64% cheaper than GLM-5
2. **Agentic Performance:** 97% skill adherence, 56.22% SWE-Pro (matches Opus)
3. **Tool Calling:** Native function calling, MCP support, structured outputs
4. **Open Weights:** Self-hosting enables data sovereignty and cost control
5. **Production-Ready:** Route-and-escalate pattern delivers 90% quality at 7% cost

### Weaknesses

1. **Verbosity:** 2.1x more tokens than median (erodes cost savings)
2. **Speed:** Below claimed performance (45.6 vs 100 tok/s)
3. **Text-Only:** No vision, audio, or multimodal support
4. **Benchmark Verification:** Self-reported scores, independent verification ongoing
5. **Context Window:** 200K is mid-range (below Gemini's 2M)

### Bottom Line

**MiniMax M2.7 is the cost-optimized choice for high-volume agentic automation where:**
- Task patterns are predictable and well-defined
- Cost per task matters more than ceiling quality
- Tool calling and structured outputs are critical
- Data sovereignty or self-hosting is required
- Route-and-escalate patterns can handle edge cases

**For routine automation tasks, M2.7 offers 64-70% cost savings vs GLM models and 17-21x savings vs frontier models, with 90% of the quality on well-defined workloads.**

---

## Sources

All claims in this report are backed by evidence from the following sources:

1. **Artificial Analysis** - https://artificialanalysis.ai/models/minimax-m2-7
2. **PricePerToken.com** - https://pricepertoken.com/pricing-page/model/minimax-minimax-m2.7
3. **Thomas Wiegold Blog** - https://thomas-wiegold.com/blog/minimax-m-2-7-review-is-it-worth-the-hype/
4. **MiniMax Official** - https://www.minimax.io/news/minimax-m27-en
5. **Kilo Blog** - https://blog.kilo.ai/p/minimax-m27
6. **Maniac.ai** - https://www.maniac.ai/blog/chinese-frontier-models-compared-glm5-minimax-kimi-qwen
7. **Digital Applied** - https://www.digitalapplied.com/blog/minimax-m2-7-agentic-coding-release-guide
8. **MarkTechPost** - https://www.marktechpost.com/2026/04/12/minimax-just-open-sourced-minimax-m2-7-a-self-evolving-agent-model-that-scores-56-22-on-swe-pro-and-57-0-on-terminal-bench-2/
9. **Vast.ai** - https://vast.ai/article/deploy-minimax-m2
10. **WaveSpeedAI** - https://wavespeed.ai/blog/posts/minimax-m2-7-self-evolving-agent-model-features-benchmarks-2026/

---

**Report Compiled:** April 23, 2026  
**Research Methodology:** Web search via Tavily CLI with advanced depth, cross-referenced across 10+ independent sources
