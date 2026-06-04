# Publication Comparison (YOLO vs LLM)

Generated at (UTC): 2026-06-04T13:15:27.190738+00:00

## Overall Accuracy

| Model | Accuracy % | 95% CI (Wilson) | Correct / Total | Extra Tokens |
|---|---:|---:|---:|---:|
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | 98.38 | 97.87-98.76 | 3090 / 3141 | 7 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | 97.61 | 97.02-98.09 | 3066 / 3141 | 7 |
| LLM OpenRouter Qwen3-VL-30B-weak | 96.69 | 96.00-97.26 | 3037 / 3141 | 0 |
| LLM OpenRouter OpenAI GPT-5.2-weak | 95.57 | 94.80-96.24 | 3002 / 3141 | 6 |
| LLM OpenRouter OpenAI GPT-5.2-plain | 93.57 | 92.66-94.37 | 2939 / 3141 | 0 |
| LLM OpenRouter Qwen3-VL-30B-plain | 93.12 | 92.18-93.96 | 2925 / 3141 | 0 |
| YOLOv5 Medium-plain | 90.93 | 89.87-91.88 | 2856 / 3141 | 24 |
| LLM OpenRouter xAI Grok-4.3-weak | 90.10 | 89.00-91.09 | 2830 / 3141 | 12 |
| LLM OpenRouter xAI Grok-4.3-plain | 89.59 | 88.47-90.61 | 2814 / 3141 | 7 |
| YOLO26 Medium-plain | 89.18 | 88.04-90.21 | 2801 / 3141 | 6 |

## Pairwise Significance (McNemar, continuity-corrected)

| Model A | Model B | n10 (A right/B wrong) | n01 (A wrong/B right) | p-value | Delta pp (B-A) |
|---|---|---:|---:|---:|---:|
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter Gemini-3.1 Flash-Lite-weak | 17 | 41 | 2.527347e-03 | 0.7641 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter OpenAI GPT-5.2-plain | 155 | 28 | 1.228889e-20 | -4.0433 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter OpenAI GPT-5.2-weak | 107 | 43 | 2.690520e-07 | -2.0376 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter Qwen3-VL-30B-plain | 172 | 31 | 8.694256e-23 | -4.4890 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter Qwen3-VL-30B-weak | 81 | 52 | 1.518620e-02 | -0.9233 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter xAI Grok-4.3-plain | 282 | 30 | 7.933848e-46 | -8.0229 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter xAI Grok-4.3-weak | 284 | 48 | 4.661417e-38 | -7.5135 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | YOLO26 Medium-plain | 294 | 29 | 7.543064e-49 | -8.4368 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | YOLOv5 Medium-plain | 249 | 39 | 7.480319e-35 | -6.6858 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter OpenAI GPT-5.2-plain | 169 | 18 | 5.381340e-28 | -4.8074 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter OpenAI GPT-5.2-weak | 103 | 15 | 1.156457e-15 | -2.8017 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter Qwen3-VL-30B-plain | 182 | 17 | 3.052547e-31 | -5.2531 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter Qwen3-VL-30B-weak | 75 | 22 | 1.293249e-07 | -1.6874 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter xAI Grok-4.3-plain | 296 | 20 | 5.534260e-54 | -8.7870 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter xAI Grok-4.3-weak | 275 | 15 | 3.081979e-52 | -8.2776 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | YOLO26 Medium-plain | 305 | 16 | 3.845137e-58 | -9.2009 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | YOLOv5 Medium-plain | 260 | 26 | 3.477402e-43 | -7.4499 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter OpenAI GPT-5.2-weak | 65 | 128 | 8.087378e-06 | 2.0057 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter Qwen3-VL-30B-plain | 113 | 99 | 3.719409e-01 | -0.4457 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter Qwen3-VL-30B-weak | 44 | 142 | 1.140545e-12 | 3.1200 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter xAI Grok-4.3-plain | 224 | 99 | 5.216952e-12 | -3.9796 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter xAI Grok-4.3-weak | 253 | 144 | 5.947760e-08 | -3.4702 |
| LLM OpenRouter OpenAI GPT-5.2-plain | YOLO26 Medium-plain | 258 | 120 | 1.834482e-12 | -4.3935 |
| LLM OpenRouter OpenAI GPT-5.2-plain | YOLOv5 Medium-plain | 232 | 149 | 2.657535e-05 | -2.6425 |
| LLM OpenRouter OpenAI GPT-5.2-weak | LLM OpenRouter Qwen3-VL-30B-plain | 159 | 82 | 9.801104e-07 | -2.4514 |
| LLM OpenRouter OpenAI GPT-5.2-weak | LLM OpenRouter Qwen3-VL-30B-weak | 56 | 91 | 5.043023e-03 | 1.1143 |
| LLM OpenRouter OpenAI GPT-5.2-weak | LLM OpenRouter xAI Grok-4.3-plain | 273 | 85 | 4.920697e-23 | -5.9854 |
| LLM OpenRouter OpenAI GPT-5.2-weak | LLM OpenRouter xAI Grok-4.3-weak | 235 | 63 | 3.929549e-23 | -5.4760 |
| LLM OpenRouter OpenAI GPT-5.2-weak | YOLO26 Medium-plain | 286 | 85 | 2.948200e-25 | -6.3992 |
| LLM OpenRouter OpenAI GPT-5.2-weak | YOLOv5 Medium-plain | 245 | 99 | 5.372205e-15 | -4.6482 |
| LLM OpenRouter Qwen3-VL-30B-plain | LLM OpenRouter Qwen3-VL-30B-weak | 19 | 131 | 1.267807e-19 | 3.5657 |
| LLM OpenRouter Qwen3-VL-30B-plain | LLM OpenRouter xAI Grok-4.3-plain | 213 | 102 | 5.725861e-10 | -3.5339 |
| LLM OpenRouter Qwen3-VL-30B-plain | LLM OpenRouter xAI Grok-4.3-weak | 251 | 156 | 3.171216e-06 | -3.0245 |
| LLM OpenRouter Qwen3-VL-30B-plain | YOLO26 Medium-plain | 267 | 143 | 1.243459e-09 | -3.9478 |
| LLM OpenRouter Qwen3-VL-30B-plain | YOLOv5 Medium-plain | 230 | 161 | 5.840661e-04 | -2.1968 |
| LLM OpenRouter Qwen3-VL-30B-weak | LLM OpenRouter xAI Grok-4.3-plain | 269 | 46 | 6.726245e-36 | -7.0996 |
| LLM OpenRouter Qwen3-VL-30B-weak | LLM OpenRouter xAI Grok-4.3-weak | 257 | 50 | 6.497158e-32 | -6.5903 |
| LLM OpenRouter Qwen3-VL-30B-weak | YOLO26 Medium-plain | 292 | 56 | 2.184350e-36 | -7.5135 |
| LLM OpenRouter Qwen3-VL-30B-weak | YOLOv5 Medium-plain | 248 | 67 | 3.602083e-24 | -5.7625 |
| LLM OpenRouter xAI Grok-4.3-plain | LLM OpenRouter xAI Grok-4.3-weak | 214 | 230 | 4.765462e-01 | 0.5094 |
| LLM OpenRouter xAI Grok-4.3-plain | YOLO26 Medium-plain | 213 | 200 | 5.548680e-01 | -0.4139 |
| LLM OpenRouter xAI Grok-4.3-plain | YOLOv5 Medium-plain | 176 | 218 | 3.887097e-02 | 1.3372 |
| LLM OpenRouter xAI Grok-4.3-weak | YOLO26 Medium-plain | 267 | 238 | 2.127704e-01 | -0.9233 |
| LLM OpenRouter xAI Grok-4.3-weak | YOLOv5 Medium-plain | 215 | 241 | 2.417064e-01 | 0.8278 |
| YOLO26 Medium-plain | YOLOv5 Medium-plain | 110 | 165 | 1.128655e-03 | 1.7510 |

## Files

- `model_overall.csv`
- `model_per_dataset.csv`
- `model_per_page.csv`
- `model_error_types.csv`
- `model_latency_dataset.csv`
- `pairwise_mcnemar.csv`
