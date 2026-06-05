# Publication Comparison (YOLO vs LLM)

Generated at (UTC): 2026-06-05T20:05:32.250337+00:00

## Overall Accuracy

| Model | Accuracy % | 95% CI (Wilson) | Correct / Total | Extra Tokens |
|---|---:|---:|---:|---:|
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | 98.38 | 97.87-98.76 | 3090 / 3141 | 7 |
| LLM Google Gemini-3.5 Flash-weak | 98.22 | 97.69-98.62 | 3085 / 3141 | 0 |
| LLM Google Gemini-3.5 Flash-plain | 97.77 | 97.19-98.23 | 3071 / 3141 | 0 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | 97.61 | 97.02-98.09 | 3066 / 3141 | 7 |
| LLM OpenRouter Qwen3-VL-30B-weak | 96.69 | 96.00-97.26 | 3037 / 3141 | 0 |
| LLM OpenRouter OpenAI GPT-5.2-weak | 95.57 | 94.80-96.24 | 3002 / 3141 | 6 |
| LLM OpenRouter Qwen3-VL-30B-plain | 93.92 | 93.03-94.70 | 2950 / 3141 | 0 |
| LLM OpenRouter OpenAI GPT-5.2-plain | 93.57 | 92.66-94.37 | 2939 / 3141 | 0 |
| YOLOv5 Medium-plain | 90.93 | 89.87-91.88 | 2856 / 3141 | 24 |
| LLM OpenRouter xAI Grok-4.3-weak | 90.10 | 89.00-91.09 | 2830 / 3141 | 12 |
| LLM OpenRouter xAI Grok-4.3-plain | 89.59 | 88.47-90.61 | 2814 / 3141 | 7 |
| YOLO26 Medium-plain | 89.18 | 88.04-90.21 | 2801 / 3141 | 6 |

## Pairwise Significance (McNemar, continuity-corrected)

| Model A | Model B | n10 (A right/B wrong) | n01 (A wrong/B right) | p-value | Delta pp (B-A) |
|---|---|---:|---:|---:|---:|
| LLM Google Gemini-3.5 Flash-plain | LLM Google Gemini-3.5 Flash-weak | 36 | 50 | 1.609672e-01 | 0.4457 |
| LLM Google Gemini-3.5 Flash-plain | LLM OpenRouter Gemini-3.1 Flash-Lite-plain | 39 | 34 | 6.396669e-01 | -0.1592 |
| LLM Google Gemini-3.5 Flash-plain | LLM OpenRouter Gemini-3.1 Flash-Lite-weak | 30 | 49 | 4.285112e-02 | 0.6049 |
| LLM Google Gemini-3.5 Flash-plain | LLM OpenRouter OpenAI GPT-5.2-plain | 171 | 39 | 1.568743e-19 | -4.2025 |
| LLM Google Gemini-3.5 Flash-plain | LLM OpenRouter OpenAI GPT-5.2-weak | 119 | 50 | 1.688061e-07 | -2.1968 |
| LLM Google Gemini-3.5 Flash-plain | LLM OpenRouter Qwen3-VL-30B-plain | 158 | 37 | 8.444879e-18 | -3.8523 |
| LLM Google Gemini-3.5 Flash-plain | LLM OpenRouter Qwen3-VL-30B-weak | 85 | 51 | 4.658779e-03 | -1.0825 |
| LLM Google Gemini-3.5 Flash-plain | LLM OpenRouter xAI Grok-4.3-plain | 290 | 33 | 4.869474e-46 | -8.1821 |
| LLM Google Gemini-3.5 Flash-plain | LLM OpenRouter xAI Grok-4.3-weak | 291 | 50 | 1.276895e-38 | -7.6727 |
| LLM Google Gemini-3.5 Flash-plain | YOLO26 Medium-plain | 305 | 35 | 3.320696e-48 | -8.5960 |
| LLM Google Gemini-3.5 Flash-plain | YOLOv5 Medium-plain | 252 | 37 | 2.450915e-36 | -6.8450 |
| LLM Google Gemini-3.5 Flash-weak | LLM OpenRouter Gemini-3.1 Flash-Lite-plain | 54 | 35 | 5.639171e-02 | -0.6049 |
| LLM Google Gemini-3.5 Flash-weak | LLM OpenRouter Gemini-3.1 Flash-Lite-weak | 28 | 33 | 6.085478e-01 | 0.1592 |
| LLM Google Gemini-3.5 Flash-weak | LLM OpenRouter OpenAI GPT-5.2-plain | 181 | 35 | 5.844889e-23 | -4.6482 |
| LLM Google Gemini-3.5 Flash-weak | LLM OpenRouter OpenAI GPT-5.2-weak | 108 | 25 | 1.157895e-12 | -2.6425 |
| LLM Google Gemini-3.5 Flash-weak | LLM OpenRouter Qwen3-VL-30B-plain | 168 | 33 | 3.335908e-21 | -4.2980 |
| LLM Google Gemini-3.5 Flash-weak | LLM OpenRouter Qwen3-VL-30B-weak | 84 | 36 | 1.782766e-05 | -1.5282 |
| LLM Google Gemini-3.5 Flash-weak | LLM OpenRouter xAI Grok-4.3-plain | 302 | 31 | 1.556673e-49 | -8.6278 |
| LLM Google Gemini-3.5 Flash-weak | LLM OpenRouter xAI Grok-4.3-weak | 273 | 18 | 3.842396e-50 | -8.1184 |
| LLM Google Gemini-3.5 Flash-weak | YOLO26 Medium-plain | 314 | 30 | 1.448933e-52 | -9.0417 |
| LLM Google Gemini-3.5 Flash-weak | YOLOv5 Medium-plain | 263 | 34 | 5.896432e-40 | -7.2907 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter Gemini-3.1 Flash-Lite-weak | 17 | 41 | 2.527347e-03 | 0.7641 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter OpenAI GPT-5.2-plain | 155 | 28 | 1.228889e-20 | -4.0433 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter OpenAI GPT-5.2-weak | 107 | 43 | 2.690520e-07 | -2.0376 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter Qwen3-VL-30B-plain | 147 | 31 | 6.717994e-18 | -3.6931 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter Qwen3-VL-30B-weak | 81 | 52 | 1.518620e-02 | -0.9233 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter xAI Grok-4.3-plain | 282 | 30 | 7.933848e-46 | -8.0229 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | LLM OpenRouter xAI Grok-4.3-weak | 284 | 48 | 4.661417e-38 | -7.5135 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | YOLO26 Medium-plain | 294 | 29 | 7.543064e-49 | -8.4368 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | YOLOv5 Medium-plain | 249 | 39 | 7.480319e-35 | -6.6858 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter OpenAI GPT-5.2-plain | 169 | 18 | 5.381340e-28 | -4.8074 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter OpenAI GPT-5.2-weak | 103 | 15 | 1.156457e-15 | -2.8017 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter Qwen3-VL-30B-plain | 157 | 17 | 5.798205e-26 | -4.4572 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter Qwen3-VL-30B-weak | 75 | 22 | 1.293249e-07 | -1.6874 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter xAI Grok-4.3-plain | 296 | 20 | 5.534260e-54 | -8.7870 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | LLM OpenRouter xAI Grok-4.3-weak | 275 | 15 | 3.081979e-52 | -8.2776 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | YOLO26 Medium-plain | 305 | 16 | 3.845137e-58 | -9.2009 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | YOLOv5 Medium-plain | 260 | 26 | 3.477402e-43 | -7.4499 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter OpenAI GPT-5.2-weak | 65 | 128 | 8.087378e-06 | 2.0057 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter Qwen3-VL-30B-plain | 88 | 99 | 4.646128e-01 | 0.3502 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter Qwen3-VL-30B-weak | 44 | 142 | 1.140545e-12 | 3.1200 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter xAI Grok-4.3-plain | 224 | 99 | 5.216952e-12 | -3.9796 |
| LLM OpenRouter OpenAI GPT-5.2-plain | LLM OpenRouter xAI Grok-4.3-weak | 253 | 144 | 5.947760e-08 | -3.4702 |
| LLM OpenRouter OpenAI GPT-5.2-plain | YOLO26 Medium-plain | 258 | 120 | 1.834482e-12 | -4.3935 |
| LLM OpenRouter OpenAI GPT-5.2-plain | YOLOv5 Medium-plain | 232 | 149 | 2.657535e-05 | -2.6425 |
| LLM OpenRouter OpenAI GPT-5.2-weak | LLM OpenRouter Qwen3-VL-30B-plain | 134 | 82 | 5.202443e-04 | -1.6555 |
| LLM OpenRouter OpenAI GPT-5.2-weak | LLM OpenRouter Qwen3-VL-30B-weak | 56 | 91 | 5.043023e-03 | 1.1143 |
| LLM OpenRouter OpenAI GPT-5.2-weak | LLM OpenRouter xAI Grok-4.3-plain | 273 | 85 | 4.920697e-23 | -5.9854 |
| LLM OpenRouter OpenAI GPT-5.2-weak | LLM OpenRouter xAI Grok-4.3-weak | 235 | 63 | 3.929549e-23 | -5.4760 |
| LLM OpenRouter OpenAI GPT-5.2-weak | YOLO26 Medium-plain | 286 | 85 | 2.948200e-25 | -6.3992 |
| LLM OpenRouter OpenAI GPT-5.2-weak | YOLOv5 Medium-plain | 245 | 99 | 5.372205e-15 | -4.6482 |
| LLM OpenRouter Qwen3-VL-30B-plain | LLM OpenRouter Qwen3-VL-30B-weak | 19 | 106 | 1.447690e-14 | 2.7698 |
| LLM OpenRouter Qwen3-VL-30B-plain | LLM OpenRouter xAI Grok-4.3-plain | 212 | 76 | 1.792020e-15 | -4.3298 |
| LLM OpenRouter Qwen3-VL-30B-plain | LLM OpenRouter xAI Grok-4.3-weak | 251 | 131 | 1.139239e-09 | -3.8204 |
| LLM OpenRouter Qwen3-VL-30B-plain | YOLO26 Medium-plain | 266 | 117 | 3.955571e-14 | -4.7437 |
| LLM OpenRouter Qwen3-VL-30B-plain | YOLOv5 Medium-plain | 230 | 136 | 1.166834e-06 | -2.9927 |
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
