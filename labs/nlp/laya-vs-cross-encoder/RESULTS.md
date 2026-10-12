# BANKING77 payment intent comparison

The [experiment protocol](README.md) fixes eight related payment intents, 50 training and 20 validation requests per intent, and all 40 official test requests per intent. This is an eight-intent subset of BANKING77, not the full 77-intent task.

Dataset source commit: `57ec275d8078af65b7731c2a98be812d844a6d6b`. Test examples: 320.

| Model | Accuracy | Macro-F1 | Log loss ↓ | Brier ↓ | ECE ↓ | ms/request ↓ | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya, original | 0.775 | 0.770 | 0.742 | 0.309 | 0.075 | 98.6 | 421,293,827 |
| Laya, fine-tuned | 0.906 | 0.906 | 0.342 | 0.139 | 0.023 | 98.8 | 421,293,827 |
| CrossEncoder, ModernBERT-base | 0.897 | 0.900 | 0.310 | 0.150 | 0.031 | 76.3 | 149,605,633 |
| CrossEncoder, ModernBERT-large | 0.934 | 0.935 | 0.248 | 0.117 | 0.042 | 208.0 | 395,832,321 |

## F1 by intent

| Intent | Laya, original | Laya, fine-tuned | CrossEncoder, ModernBERT-base | CrossEncoder, ModernBERT-large |
| --- | ---: | ---: | ---: | ---: |
| `card_payment_fee_charged` | 0.935 | 0.935 | 0.904 | 0.975 |
| `card_payment_not_recognised` | 0.622 | 0.828 | 0.780 | 0.861 |
| `card_payment_wrong_exchange_rate` | 0.919 | 0.987 | 0.987 | 0.987 |
| `declined_card_payment` | 0.755 | 0.914 | 0.935 | 0.929 |
| `pending_card_payment` | 0.923 | 0.951 | 0.925 | 0.962 |
| `reverted_card_payment?` | 0.561 | 0.854 | 0.949 | 0.933 |
| `transaction_charged_twice` | 0.776 | 0.950 | 0.881 | 0.975 |
| `direct_debit_payment_not_recognised` | 0.667 | 0.833 | 0.841 | 0.857 |

## Observations

Laya classified 248/320 test requests correctly before fine-tuning and 290/320 afterward, an increase of 42 correct requests. CrossEncoder, ModernBERT-base classified 287/320 correctly. CrossEncoder, ModernBERT-large classified 299/320 correctly.

With similar-sized ModernBERT encoders, the large CrossEncoder was correct on 9 more of the 320 test requests than fine-tuned Laya (299 versus 290); its measured latency was 208.0 versus 98.8 ms/request. One test split and one timing run do not establish a reliable ranking.

The two CrossEncoder rows separate a practical base-sized baseline from a large-sized comparison with Laya. The models still differ in their starting checkpoints, heads, and training objectives, so differences in accuracy cannot be attributed to the Decision Head alone.

These are single local CPU timing runs, not hardware-independent throughput claims.

The most frequent Laya, fine-tuned confusion was `direct_debit_payment_not_recognised` → `card_payment_not_recognised` (8 requests).
The most frequent CrossEncoder, ModernBERT-base confusion was `direct_debit_payment_not_recognised` → `card_payment_not_recognised` (8 requests).
The most frequent CrossEncoder, ModernBERT-large confusion was `card_payment_not_recognised` → `direct_debit_payment_not_recognised` (9 requests).

## Reading these results

All four runs used `cpu`, batch size 4, the same question wording, option descriptions, and official test rows. Each CrossEncoder scores eight pairs per request; Laya scores one eight-option question. Laya and the large CrossEncoder use ModernBERT-large-family encoders, but their exact initialization and task-specific training history are not matched. This remains a system comparison, not an architecture ablation.

Laya reserved 40 of its 400 allocated training requests for probability calibration, leaving 360 gradient-training items. CrossEncoder used all 400 for gradient training and fitted temperature on validation logits. Compare both quality and probability metrics with this protocol difference in mind. The benchmark does not measure out-of-scope detection or generalization to all 77 BANKING77 intents.

The model checkpoints and raw data are excluded from Git; rerun the commands in [README.md](README.md) to regenerate them.
