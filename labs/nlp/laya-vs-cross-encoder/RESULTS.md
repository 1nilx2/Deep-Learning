# BANKING77 payment intent comparison

The [experiment protocol](README.md) fixes eight related payment intents, 50 training and 20 validation requests per intent, and all 40 official test requests per intent. This is an eight-intent subset of BANKING77, not the full 77-intent task.

Dataset source commit: `57ec275d8078af65b7731c2a98be812d844a6d6b`. Test examples: 320.

| Model | Accuracy | Macro-F1 | Log loss ↓ | Brier ↓ | ECE ↓ | ms/request ↓ | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya, original | 0.775 | 0.770 | 0.742 | 0.309 | 0.075 | 98.6 | 421,293,827 |
| Laya, fine-tuned | 0.906 | 0.906 | 0.342 | 0.139 | 0.023 | 98.8 | 421,293,827 |
| CrossEncoder, fine-tuned | 0.897 | 0.900 | 0.310 | 0.150 | 0.031 | 76.3 | 149,605,633 |

## F1 by intent

| Intent | Laya original | Laya fine-tuned | CrossEncoder |
| --- | ---: | ---: | ---: |
| `card_payment_fee_charged` | 0.935 | 0.935 | 0.904 |
| `card_payment_not_recognised` | 0.622 | 0.828 | 0.780 |
| `card_payment_wrong_exchange_rate` | 0.919 | 0.987 | 0.987 |
| `declined_card_payment` | 0.755 | 0.914 | 0.935 |
| `pending_card_payment` | 0.923 | 0.951 | 0.925 |
| `reverted_card_payment?` | 0.561 | 0.854 | 0.949 |
| `transaction_charged_twice` | 0.776 | 0.950 | 0.881 |
| `direct_debit_payment_not_recognised` | 0.667 | 0.833 | 0.841 |

## Observations

Laya classified 248/320 test requests correctly before fine-tuning and 290/320 afterward, an increase of 42 correct requests. CrossEncoder classified 287/320 correctly. The 3-request gap between the fine-tuned models is small on this test set and does not establish a reliable ranking.

CrossEncoder used 149,605,633 parameters and 76.3 ms/request; fine-tuned Laya used 421,293,827 parameters and 98.8 ms/request. These are single local CPU timing runs, not a hardware-independent throughput claim.

The most frequent Laya, fine-tuned confusion was `direct_debit_payment_not_recognised` → `card_payment_not_recognised` (8 requests).
The most frequent CrossEncoder, fine-tuned confusion was `direct_debit_payment_not_recognised` → `card_payment_not_recognised` (8 requests).

## Reading these results

All three runs used `cpu`, batch size 4, the same question wording, option descriptions, and official test rows. The CrossEncoder scores eight pairs per request; Laya scores one eight-option question. The starting checkpoints differ in size and training history, so this is a system comparison.

Laya reserved 40 of its 400 allocated training requests for probability calibration, leaving 360 gradient-training items. CrossEncoder used all 400 for gradient training and fitted temperature on validation logits. Compare both quality and probability metrics with this protocol difference in mind. The benchmark does not measure out-of-scope detection or generalization to all 77 BANKING77 intents.

The model checkpoints and raw data are excluded from Git; rerun the commands in [README.md](README.md) to regenerate them.
