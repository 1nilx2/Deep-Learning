# Learning labs

These notebooks record hands-on study and experiments. Some follow external tutorials or require hosted environments and datasets; check each notebook's first cells and cited references before running it.

| Area | Notebook | Focus |
| --- | --- | --- |
| LLM | [Llama 3.2 11B fine-tuning](llm/Finetune_Llama_3.2_11B.ipynb) | Fine-tuning workflow |
| LLM | [Llama 3.1 8B with LoRA and Unsloth](llm/Llama_3_1_8B_LoRA_Unsloth_Colab.ipynb) | Adapter training in Colab |
| Multimodal | [Idefics2 with LoRA](llm/VLM_Idefics2_lora_in_sagemaker.ipynb) | Vision-language fine-tuning in SageMaker |
| NLP | [BERT binary classification](nlp/BERT_BinaryClassification.ipynb) | Classification workflow |
| NLP | [KLUE BERT tagging](nlp/KLUE_BERT_Tagging.ipynb) | Token tagging |
| NLP | [Transformer chatbot](nlp/Transformer_ChatBot.ipynb) | Transformer-based question answering |
| CV | [Neural style transfer](cv/style-transfer/Neural_Style_Transfer.ipynb) | Style transfer, with [AdaIN code](cv/style-transfer/AdaIn.py) |
| CV | [Variational autoencoder](cv/Variational_AutoEncoders,VAE.ipynb) | Image generation with a VAE |
| CV | [MMSegmentation customization](cv/mmseg_custom.ipynb) | Segmentation framework exploration |

Notebook outputs are retained as learning records. They are not presented as independently reproduced benchmark comparisons.

The KLUE BERT tagging and Transformer chatbot notebooks refer to CSV/text data files that are not included here. Their original data sources need to be supplied before those notebooks can run end to end.
