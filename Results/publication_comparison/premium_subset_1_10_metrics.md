# Premium Subset Metrics (Datasets 1..10)

| Model | Acc % | TP | FN | FP | TN | Recall | FNR | Precision | Specificity | Class-Acc | q_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLM Google Gemini-3.5 Flash-weak | 99.41 | 357 | 0 | 2 | 151 | 1.0000 | 0.0000 | 0.9944 | 0.9869 | 0.9961 | 9 |
| LLM OpenRouter Qwen3-VL-30B-weak | 96.47 | 355 | 2 | 11 | 142 | 0.9944 | 0.0056 | 0.9699 | 0.9281 | 0.9745 | 5 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | 98.82 | 354 | 3 | 2 | 151 | 0.9916 | 0.0084 | 0.9944 | 0.9869 | 0.9902 | 10 |
| LLM OpenRouter OpenAI GPT-5.5-weak-subset_1_10 | 90.78 | 353 | 4 | 35 | 118 | 0.9888 | 0.0112 | 0.9098 | 0.7712 | 0.9235 | 18 |
| LLM Google Gemini-3.5 Flash-plain | 98.63 | 352 | 5 | 0 | 153 | 0.9860 | 0.0140 | 1.0000 | 1.0000 | 0.9902 | 13 |
| LLM OpenRouter Anthropic Claude Opus 4.7-weak-subset_1_10 | 98.82 | 352 | 5 | 1 | 152 | 0.9860 | 0.0140 | 0.9972 | 0.9935 | 0.9882 | 9 |
| LLM OpenRouter OpenAI GPT-5.2-weak | 95.10 | 349 | 8 | 11 | 142 | 0.9776 | 0.0224 | 0.9694 | 0.9281 | 0.9627 | 10 |
| LLM OpenRouter xAI Grok-4.3-weak | 90.59 | 348 | 9 | 25 | 128 | 0.9748 | 0.0252 | 0.9330 | 0.8366 | 0.9333 | 22 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | 97.65 | 346 | 11 | 0 | 153 | 0.9692 | 0.0308 | 1.0000 | 1.0000 | 0.9784 | 10 |
| LLM OpenRouter Anthropic Claude Opus 4.7-plain-subset_1_10 | 97.06 | 343 | 14 | 0 | 153 | 0.9608 | 0.0392 | 1.0000 | 1.0000 | 0.9725 | 10 |
| LLM OpenRouter OpenAI GPT-5.5-plain-subset_1_10 | 96.08 | 340 | 17 | 0 | 153 | 0.9524 | 0.0476 | 1.0000 | 1.0000 | 0.9667 | 11 |
| LLM OpenRouter Qwen3-VL-30B-plain | 94.51 | 338 | 19 | 2 | 151 | 0.9468 | 0.0532 | 0.9941 | 0.9869 | 0.9588 | 11 |
| LLM OpenRouter OpenAI GPT-5.2-plain | 92.94 | 333 | 24 | 3 | 150 | 0.9328 | 0.0672 | 0.9911 | 0.9804 | 0.9471 | 12 |
| YOLOv5 Medium-plain | 91.18 | 332 | 25 | 0 | 153 | 0.9300 | 0.0700 | 1.0000 | 1.0000 | 0.9510 | 11 |
| YOLO26 Medium-plain | 91.18 | 321 | 36 | 0 | 153 | 0.8992 | 0.1008 | 1.0000 | 1.0000 | 0.9294 | 39 |
| LLM OpenRouter xAI Grok-4.3-plain | 90.39 | 319 | 38 | 0 | 153 | 0.8936 | 0.1064 | 1.0000 | 1.0000 | 0.9255 | 41 |

## Premium Models Only (Datasets 1..10)

| Model | Acc % | Recall | FNR | Precision | Specificity |
|---|---:|---:|---:|---:|---:|
| Gemini-3.1 Flash-Lite-plain | 97.65 | 0.9692 | 0.0308 | 1.0000 | 1.0000 |
| Gemini-3.1 Flash-Lite-weak | 98.82 | 0.9916 | 0.0084 | 0.9944 | 0.9869 |
| LLM Google Gemini-3.5 Flash-plain | 98.63 | 0.9860 | 0.0140 | 1.0000 | 1.0000 |
| LLM Google Gemini-3.5 Flash-weak | 99.41 | 1.0000 | 0.0000 | 0.9944 | 0.9869 |
| OpenAI GPT-5.2-plain | 92.94 | 0.9328 | 0.0672 | 0.9911 | 0.9804 |
| OpenAI GPT-5.2-weak | 95.10 | 0.9776 | 0.0224 | 0.9694 | 0.9281 |
| OpenAI GPT-5.5-plain-subset_1_10 | 96.08 | 0.9524 | 0.0476 | 1.0000 | 1.0000 |
| OpenAI GPT-5.5-weak-subset_1_10 | 90.78 | 0.9888 | 0.0112 | 0.9098 | 0.7712 |
| Anthropic Claude Opus 4.7-plain-subset_1_10 | 97.06 | 0.9608 | 0.0392 | 1.0000 | 1.0000 |
| Anthropic Claude Opus 4.7-weak-subset_1_10 | 98.82 | 0.9860 | 0.0140 | 0.9972 | 0.9935 |
| xAI Grok-4.3-plain | 90.39 | 0.8936 | 0.1064 | 1.0000 | 1.0000 |
| xAI Grok-4.3-weak | 90.59 | 0.9748 | 0.0252 | 0.9330 | 0.8366 |
| Qwen3-VL-30B-plain | 94.51 | 0.9468 | 0.0532 | 0.9941 | 0.9869 |
| Qwen3-VL-30B-weak | 96.47 | 0.9944 | 0.0056 | 0.9699 | 0.9281 |
