# GLM 4.7 (Z.ai) Model Research Report

**Research Date:** April 23, 2026  
**Model Version:** GLM-4.7 (including GLM-4.7-Flash variant)  
**Provider:** Z.ai (Zhipu AI)

---

## Executive Summary

GLM 4.7 is a Mixture-of-Experts (MoE) large language model with **355B total parameters** and **32B activated parameters**, positioned as a mid-tier coding-focused model between GLM 5.1 and smaller alternatives. It features hybrid reasoning capabilities with "preserved thinking" mode and demonstrates strong performance in terminal-based tasks, agentic coding, and tool calling.

---

## 1. Terminal-Based Task Execution Benchmarks

### Terminal Bench 2.0
- **GLM-4.7 Score:** 41.0%
- **Improvement over GLM-4.6:** +16.5% (from 24.5%)
- **Comparison:**
  - DeepSeek-V3.2: 46.4%
  - Gemini 3.0 Pro: 54.2%
  - GPT-5.1-High: 47.6%
  - Claude Sonnet 4.5: 42.8%

**Source:** [Hugging Face Model Card](https://huggingface.co/zai-org/GLM-4.7)

### Terminal Bench Hard
- **GLM-4.7 Score:** 33.3%
- **Improvement over GLM-4.6:** +9.7% (from 23.6%)
- **Comparison:**
  - DeepSeek-V3.2: 35.4%
  - Gemini 3.0 Pro: 39.0%
  - GPT-5.1-High: 43.0%
  - Claude Sonnet 4.5: 33.3%

**Evidence Quote:**
> "GLM-4.7 brings clear gains, compared to its predecessor GLM-4.6, in multilingual agentic coding and terminal-based tasks, including (73.8%, +5.8%) on SWE-bench, (66.7%, +12.9%) on SWE-bench Multilingual, and (41%, +16.5%) on Terminal Bench 2.0."

**Source:** [GitHub Repository](https://github.com/zai-org/GLM-4.5)

---

## 2. Coding Performance Metrics

### SWE-bench Verified
- **GLM-4.7 Score:** 73.8%
- **Improvement over GLM-4.6:** +5.8% (from 68.0%)
- **Ranking:** Competitive with top models
- **Comparison:**
  - GPT-5.1-High: 76.3%
  - Claude Sonnet 4.5: 77.2%
  - Gemini 3.0 Pro: 76.2%
  - DeepSeek-V3.2: 73.1%

### SWE-bench Multilingual
- **GLM-4.7 Score:** 66.7%
- **Improvement over GLM-4.6:** +12.9% (from 53.8%)
- **Comparison:**
  - DeepSeek-V3.2: 70.2%
  - Claude Sonnet 4.5: 68.0%
  - Kimi K2 Thinking: 61.1%

### LiveCodeBench v6
- **GLM-4.7 Score:** 84.9%
- **Achievement:** Open-source SOTA, surpassing Claude Sonnet 4.5 (64.0%)
- **Comparison:**
  - Gemini 3.0 Pro: 90.7%
  - GPT-5.1-High: 87.0%
  - GPT-5-High: 87.0%
  - DeepSeek-V3.2: 83.3%

**Evidence Quote:**
> "GLM-4.7 also supports thinking before acting, with significant improvements on complex tasks in mainstream agent frameworks such as Claude Code, Kilo Code, Cline, and Roo Code."

**Source:** [Z.ai Documentation](https://docs.z.ai/guides/llm/glm-4.7)

### Code Arena Ranking
- **Position:** #1 among open-source models and domestic models
- **Performance:** Outperforms GPT-5.2 in blind testing
- **Participants:** Millions of global users

---

## 3. Context Window and Pricing

### Context Window
- **Input Context:** 200K tokens (expanded from 128K in GLM-4.6)
- **Maximum Output:** 128K tokens
- **Modalities:** Text input/output only

**Evidence Quote:**
> "Context Length: 200K"  
> "Maximum Output Tokens: 128K"

**Source:** [Z.ai Documentation](https://docs.z.ai/guides/llm/glm-4.7)

### Pricing
**Note:** Specific pricing information was not publicly available in the documentation accessed. The Z.ai pricing page returned a 404 error. However, the documentation mentions:

> "Tired of limits? Get premium performance at a fraction of the cost, fully compatible with top coding tools like Claude Code and Cline. Starting from just $10/month."

**Inference:** Pricing appears to be subscription-based starting at $10/month, but detailed token-based pricing is not publicly documented.

---

## 4. Position as Mid-Tier Option

### Model Hierarchy

**GLM 5.1 (High-End):**
- 478B total parameters
- Top-tier performance across all benchmarks
- Higher computational requirements

**GLM 4.7 (Mid-Tier):**
- 355B total parameters, 32B activated
- Balanced performance and efficiency
- Strong coding and agentic capabilities
- **Positioning:** "Your new coding partner"

**GLM 4.7-Flash (Lightweight):**
- 30B-A3B MoE model
- "Strongest model in the 30B class"
- Optimized for lightweight deployment

**MiniMax (Lower-Tier):**
- Not directly compared in documentation
- Assumed to be smaller/faster alternative

**Evidence Quote:**
> "We also provide the lightweight 30B-A3B model GLM-4.7-Flash, offering a new option for lightweight deployment that balances performance and efficiency."

**Source:** [GitHub Repository](https://github.com/zai-org/GLM-4.5)

---

## 5. "Preserved Thinking" Mode Capabilities

### Three Thinking Modes

#### 1. Interleaved Thinking
- **Function:** Model thinks before every response and tool calling
- **Benefit:** Improves instruction following and generation quality
- **Introduced:** GLM-4.5 (enhanced in GLM-4.7)

#### 2. Preserved Thinking (NEW in GLM-4.7)
- **Function:** Automatically retains all thinking blocks across multi-turn conversations
- **Mechanism:** Reuses existing reasoning instead of re-deriving from scratch
- **Benefits:**
  - Reduces information loss
  - Minimizes inconsistencies
  - Improves cache hit rates
  - Reduces computational costs
- **Use Case:** Long-horizon, complex coding agent tasks

#### 3. Turn-Level Thinking (NEW in GLM-4.7)
- **Function:** Per-turn control over reasoning within a session
- **Flexibility:**
  - Disable thinking for lightweight requests (reduce latency/cost)
  - Enable thinking for complex tasks (improve accuracy/stability)

**Evidence Quote:**
> "GLM-4.7 further enhances Interleaved Thinking (a feature introduced since GLM-4.5) and introduces Preserved Thinking and Turn-level Thinking. By thinking between actions and staying consistent across turns, it makes complex tasks more stable and more controllable."

**Source:** [Hugging Face Model Card](https://huggingface.co/zai-org/GLM-4.7)

### Configuration (SGLang only)
```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "clear_thinking": false
  }
}
```

**More Details:** https://docs.z.ai/guides/capabilities/thinking-mode

---

## 6. Tool Calling Reliability

### τ²-Bench (Tool Calling Benchmark)
- **GLM-4.7 Score:** 87.4%
- **Achievement:** Open-source SOTA
- **Improvement over GLM-4.6:** +12.2% (from 75.2%)
- **Comparison:**
  - Gemini 3.0 Pro: 90.7%
  - Claude Sonnet 4.5: 87.2%
  - DeepSeek-V3.2: 85.3%
  - GPT-5.1-High: 82.7%

### BrowseComp (Web Browsing)
- **GLM-4.7 Score:** 52.0%
- **Improvement over GLM-4.6:** +6.9% (from 45.1%)
- **With Context Management:** 67.5% (vs 57.5% for GLM-4.6)
- **Comparison:**
  - GPT-5-High: 54.9%
  - DeepSeek-V3.2: 51.4%
  - Claude Sonnet 4.5: 24.1%

### BrowseComp-Zh (Chinese Web Browsing)
- **GLM-4.7 Score:** 66.6%
- **Improvement over GLM-4.6:** +17.1% (from 49.5%)
- **Comparison:**
  - DeepSeek-V3.2: 65.0%
  - GPT-5-High: 63.0%
  - Kimi K2 Thinking: 62.3%

**Evidence Quote:**
> "GLM-4.7 achieves significantly improvements in Tool using. Significant better performances can be seen on benchmarks such as τ²-Bench and on web browsing via BrowseComp."

**Source:** [GitHub Repository](https://github.com/zai-org/GLM-4.5)

### Tool Calling Features
- **Parser:** `glm47` (specific to GLM-4.7)
- **Format:** OpenAI-style tool description format
- **Auto Tool Choice:** Supported in vLLM (`--enable-auto-tool-choice`)
- **Integration:** Compatible with mainstream agent frameworks

---

## 7. Complex Reasoning Capabilities

### HLE (Humanity's Last Exam)
- **GLM-4.7 Score:** 24.8%
- **Improvement over GLM-4.6:** +7.6% (from 17.2%)
- **With Tools:** 42.8% (+12.4% improvement)
- **Comparison:**
  - Gemini 3.0 Pro: 37.5% (45.8% with tools)
  - GPT-5.1-High: 25.7% (42.7% with tools)
  - DeepSeek-V3.2: 25.1% (40.8% with tools)

### GPQA-Diamond
- **GLM-4.7 Score:** 85.7%
- **Improvement over GLM-4.6:** +4.7% (from 81.0%)
- **Comparison:**
  - Gemini 3.0 Pro: 91.9%
  - GPT-5.1-High: 88.1%
  - Claude Sonnet 4.5: 83.4%

### MMLU-Pro
- **GLM-4.7 Score:** 84.3%
- **Improvement over GLM-4.6:** +1.1% (from 83.2%)
- **Comparison:**
  - Gemini 3.0 Pro: 90.1%
  - Claude Sonnet 4.5: 88.2%
  - GPT-5-High: 87.5%

### Mathematical Reasoning

#### AIME 2025
- **GLM-4.7 Score:** 95.7%
- **Improvement over GLM-4.6:** +1.8% (from 93.9%)
- **Comparison:**
  - Gemini 3.0 Pro: 95.0%
  - GPT-5-High: 94.6%
  - DeepSeek-V3.2: 93.1%

#### HMMT Feb. 2025
- **GLM-4.7 Score:** 97.1%
- **Improvement over GLM-4.6:** +7.9% (from 89.2%)
- **Comparison:**
  - Gemini 3.0 Pro: 97.5%
  - DeepSeek-V3.2: 92.5%
  - GPT-5-High: 88.3%

#### HMMT Nov. 2025
- **GLM-4.7 Score:** 93.5%
- **Improvement over GLM-4.6:** +5.8% (from 87.7%)
- **Comparison:**
  - Gemini 3.0 Pro: 93.3%
  - DeepSeek-V3.2: 90.2%
  - GPT-5-High: 89.2%

**Evidence Quote:**
> "GLM-4.7 delivers a substantial boost in mathematical and reasoning capabilities, achieving (42.8%, +12.4%) on the HLE (Humanity's Last Exam) benchmark compared to GLM-4.6."

**Source:** [Hugging Face Model Card](https://huggingface.co/zai-org/GLM-4.7)

---

## 8. Additional Capabilities

### Frontend/UI Generation
- **Improvement:** "Big step forward in improving UI quality"
- **Features:**
  - Cleaner, more modern webpages
  - Better-looking slides with accurate layout and sizing
  - PPT 16:9 compatibility: 91% (up from 52%)
  - More flexible typography and color schemes

**Evidence Quote:**
> "Vibe Coding: GLM-4.7 takes a big step forward in improving UI quality. It produces cleaner, more modern webpages and generates better-looking slides with more accurate layout and sizing."

### General Capabilities
- Enhanced chat quality
- Improved creative writing
- Better role-playing scenarios
- More concise, intelligent, and empathetic conversations

---

## 9. Technical Specifications

### Model Architecture
- **Type:** Mixture-of-Experts (MoE)
- **Total Parameters:** 355B
- **Activated Parameters:** 32B
- **Precision Options:** BF16, FP8
- **License:** MIT (open-source)

### Hardware Requirements (Minimum)

| Model | Precision | GPU Configuration |
|-------|-----------|-------------------|
| GLM-4.7 | BF16 | H100 x 16 |
| GLM-4.7 | FP8 | H100 x 8 |
| GLM-4.7-Flash | BF16 | H100 x 1 |

### Hardware Requirements (Full 128K Context)

| Model | Precision | GPU Configuration |
|-------|-----------|-------------------|
| GLM-4.7 | BF16 | H100 x 32 |
| GLM-4.7 | FP8 | H100 x 16 |
| GLM-4.7-Flash | BF16 | H100 x 2 |

**Source:** [GitHub Repository](https://github.com/zai-org/GLM-4.5)

### Inference Frameworks
- **vLLM:** Supported (main branch)
- **SGLang:** Supported (main branch)
- **Transformers:** Supported (v4.57.3+)
- **Speculative Decoding:** MTP (Multi-Token Prediction)

### Evaluation Parameters

**Default Settings:**
- Temperature: 1.0
- Top-p: 0.95
- Max new tokens: 131,072

**Terminal Bench / SWE-bench:**
- Temperature: 0.7
- Top-p: 1.0
- Max new tokens: 16,384

**τ²-Bench:**
- Temperature: 0
- Max new tokens: 16,384

---

## 10. Competitive Positioning

### Strengths
1. **Terminal-based tasks:** Strong performance with 41% on Terminal Bench 2.0
2. **Tool calling:** 87.4% on τ²-Bench (open-source SOTA)
3. **Multilingual coding:** 66.7% on SWE-bench Multilingual
4. **Preserved thinking:** Unique feature for long-horizon tasks
5. **Cost-effectiveness:** Mid-tier pricing with competitive performance
6. **Open-source:** MIT license for commercial use

### Weaknesses
1. **Not top-tier:** Trails Gemini 3.0 Pro and GPT-5.1-High on most benchmarks
2. **Hardware requirements:** Requires significant GPU resources (8-16 H100s for FP8)
3. **Terminal Bench 2.0:** Behind DeepSeek-V3.2 (46.4%) and Gemini 3.0 Pro (54.2%)
4. **HLE reasoning:** 24.8% without tools (vs 37.5% for Gemini 3.0 Pro)

### Best Use Cases
- Agentic coding workflows (Claude Code, Cline, Roo Code)
- Terminal-based automation
- Multi-turn coding conversations with preserved context
- Tool-heavy applications (web browsing, API integration)
- Frontend/UI generation
- Cost-sensitive deployments requiring strong coding capabilities

---

## 11. Comparison with Competitors

### vs. GLM 5.1
- **GLM 5.1:** Higher-end model with more parameters
- **GLM 4.7:** More efficient, better cost/performance ratio
- **Trade-off:** Slightly lower performance for significantly lower resource requirements

### vs. DeepSeek-V3.2
- **Terminal Bench 2.0:** DeepSeek leads (46.4% vs 41.0%)
- **SWE-bench Verified:** GLM-4.7 leads (73.8% vs 73.1%)
- **Tool calling:** GLM-4.7 leads (87.4% vs 85.3%)
- **Positioning:** Similar tier, different strengths

### vs. Claude Sonnet 4.5
- **SWE-bench Verified:** Claude leads (77.2% vs 73.8%)
- **LiveCodeBench v6:** GLM-4.7 leads significantly (84.9% vs 64.0%)
- **Tool calling:** Similar (87.2% vs 87.4%)
- **BrowseComp:** GLM-4.7 leads significantly (52.0% vs 24.1%)

### vs. Gemini 3.0 Pro
- **Overall:** Gemini leads on most benchmarks
- **Terminal Bench 2.0:** Gemini leads (54.2% vs 41.0%)
- **HLE:** Gemini leads (37.5% vs 24.8%)
- **Cost:** GLM-4.7 likely more cost-effective

---

## 12. Sources and References

1. **Hugging Face Model Card:** https://huggingface.co/zai-org/GLM-4.7
2. **GitHub Repository:** https://github.com/zai-org/GLM-4.5
3. **Z.ai Documentation:** https://docs.z.ai/guides/llm/glm-4.7
4. **Technical Report (arXiv):** https://arxiv.org/abs/2508.06471
5. **Hugging Face GLM-4.7-Flash:** https://huggingface.co/zai-org/GLM-4.7-Flash

---

## 13. Conclusion

GLM 4.7 represents a strong mid-tier option for coding-focused applications, particularly excelling in:
- Terminal-based task execution (41% on Terminal Bench 2.0)
- Tool calling reliability (87.4% on τ²-Bench)
- Multilingual coding (66.7% on SWE-bench Multilingual)
- Preserved thinking for long-horizon tasks

While it doesn't match the absolute performance of Gemini 3.0 Pro or GPT-5.1-High, it offers competitive capabilities at a more accessible price point ($10/month subscription mentioned). The unique "preserved thinking" mode and strong integration with coding tools (Claude Code, Cline, Roo Code) make it particularly suitable for agentic coding workflows.

**Recommendation:** GLM 4.7 is well-positioned as a cost-effective alternative to top-tier models for coding-heavy applications, especially when preserved context across multi-turn conversations is valuable.

---

**Report Compiled:** April 23, 2026  
**Research Methodology:** Web scraping, documentation analysis, benchmark comparison  
**Data Sources:** Official documentation, Hugging Face, GitHub, arXiv
