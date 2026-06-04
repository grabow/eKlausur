# Premium Subset Metrics (Datasets 1..10)

| Model | Acc % | TP | FN | FP | TN | Recall | FNR | Precision | Specificity | Class-Acc | q_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLM OpenRouter Qwen3-VL-30B-weak | 96.47 | 355 | 2 | 11 | 142 | 0.9944 | 0.0056 | 0.9699 | 0.9281 | 0.9745 | 5 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-weak | 98.82 | 354 | 3 | 2 | 151 | 0.9916 | 0.0084 | 0.9944 | 0.9869 | 0.9902 | 10 |
| LLM OpenRouter OpenAI GPT-5.2-weak | 95.10 | 349 | 8 | 11 | 142 | 0.9776 | 0.0224 | 0.9694 | 0.9281 | 0.9627 | 10 |
| LLM OpenRouter xAI Grok-4.3-weak | 90.59 | 348 | 9 | 25 | 128 | 0.9748 | 0.0252 | 0.9330 | 0.8366 | 0.9333 | 22 |
| LLM OpenRouter Gemini-3.1 Flash-Lite-plain | 97.65 | 346 | 11 | 0 | 153 | 0.9692 | 0.0308 | 1.0000 | 1.0000 | 0.9784 | 10 |
| LLM OpenRouter OpenAI GPT-5.5-weak-subset_1_10 | 88.04 | 343 | 14 | 33 | 120 | 0.9608 | 0.0392 | 0.9122 | 0.7843 | 0.9078 | 35 |
| LLM OpenRouter Anthropic Claude Opus 4.7-plain-subset_1_10 | 97.06 | 343 | 14 | 0 | 153 | 0.9608 | 0.0392 | 1.0000 | 1.0000 | 0.9725 | 10 |
| LLM OpenRouter OpenAI GPT-5.5-plain-subset_1_10 | 96.08 | 340 | 17 | 0 | 153 | 0.9524 | 0.0476 | 1.0000 | 1.0000 | 0.9667 | 11 |
| LLM OpenRouter OpenAI GPT-5.2-plain | 92.94 | 333 | 24 | 3 | 150 | 0.9328 | 0.0672 | 0.9911 | 0.9804 | 0.9471 | 12 |
| LLM OpenRouter Anthropic Claude Opus 4.7-weak-subset_1_10 | 93.73 | 333 | 24 | 1 | 152 | 0.9328 | 0.0672 | 0.9970 | 0.9935 | 0.9510 | 35 |
| YOLOv5 Medium-plain | 91.18 | 332 | 25 | 0 | 153 | 0.9300 | 0.0700 | 1.0000 | 1.0000 | 0.9510 | 11 |
| YOLO26 Medium-plain | 91.18 | 321 | 36 | 0 | 153 | 0.8992 | 0.1008 | 1.0000 | 1.0000 | 0.9294 | 39 |
| LLM OpenRouter Qwen3-VL-30B-plain | 89.61 | 320 | 37 | 2 | 151 | 0.8964 | 0.1036 | 0.9938 | 0.9869 | 0.9235 | 37 |
| LLM OpenRouter xAI Grok-4.3-plain | 90.39 | 319 | 38 | 0 | 153 | 0.8936 | 0.1064 | 1.0000 | 1.0000 | 0.9255 | 41 |

## Premium Models Only (Datasets 1..10)

| Model | Acc % | Recall | FNR | Precision | Specificity |
|---|---:|---:|---:|---:|---:|
| Gemini-3.1 Flash-Lite-plain | 97.65 | 0.9692 | 0.0308 | 1.0000 | 1.0000 |
| Gemini-3.1 Flash-Lite-weak | 98.82 | 0.9916 | 0.0084 | 0.9944 | 0.9869 |
| OpenAI GPT-5.2-plain | 92.94 | 0.9328 | 0.0672 | 0.9911 | 0.9804 |
| OpenAI GPT-5.2-weak | 95.10 | 0.9776 | 0.0224 | 0.9694 | 0.9281 |
| OpenAI GPT-5.5-plain-subset_1_10 | 96.08 | 0.9524 | 0.0476 | 1.0000 | 1.0000 |
| OpenAI GPT-5.5-weak-subset_1_10 | 88.04 | 0.9608 | 0.0392 | 0.9122 | 0.7843 |
| Anthropic Claude Opus 4.7-plain-subset_1_10 | 97.06 | 0.9608 | 0.0392 | 1.0000 | 1.0000 |
| Anthropic Claude Opus 4.7-weak-subset_1_10 | 93.73 | 0.9328 | 0.0672 | 0.9970 | 0.9935 |
| xAI Grok-4.3-plain | 90.39 | 0.8936 | 0.1064 | 1.0000 | 1.0000 |
| xAI Grok-4.3-weak | 90.59 | 0.9748 | 0.0252 | 0.9330 | 0.8366 |
| Qwen3-VL-30B-plain | 89.61 | 0.8964 | 0.1036 | 0.9938 | 0.9869 |
| Qwen3-VL-30B-weak | 96.47 | 0.9944 | 0.0056 | 0.9699 | 0.9281 |
