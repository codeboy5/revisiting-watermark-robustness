# Revisiting the Robustness of Watermarking to Paraphrasing Attacks [EMNLP 2024 Main]
Saksham Rastogi, Danish Pruthi

[EMNLP](https://aclanthology.org/2024.emnlp-main.1005/) | [ArXiv](https://arxiv.org/abs/2411.05277) | [PDF](https://arxiv.org/abs/2411.05277)

## TL;DR: Reverse engineer LLM watermarks to improve the effectiveness of paraphrasing attacks.

# Abstract

Amidst rising concerns about the internet being proliferated with content generated from language models (LMs), watermarking is seen as a principled way to certify whether text was generated from a model. Many recent watermarking techniques slightly modify the output probabilities of LMs to embed a signal in the generated output that can later be detected. Since early proposals for text watermarking, questions about their robustness to paraphrasing have been prominently discussed. Lately, some techniques are deliberately designed and claimed to be robust to paraphrasing. Particularly, a recent approach trains a model to produce a watermarking signal that is invariant to semantically-similar inputs. However, such watermarking schemes do not adequately account for the ease with which they can be reverse-engineered. We show that with limited access to model generations, we can undo the effects of watermarking and drastically improve the effectiveness of paraphrasing attacks.

## Example commands:

### Generate a corpus of watermarked text

```bash
python generate_watermarked_text.py --model_name "meta-llama/Llama-2-7b-hf" --dataset_name "armanc/scientific_papers" --dataset_config_name "arxiv" --dataset_split "test" --num_samples 3200 --min_new_tokens 200 --seed 104729 --watermark_configs_file config/watermark_config_scaling_kgw_simple0.json --output_file final_watermarked_text/test/Llama-2-7b-hf_simple_0_seed104729_arxiv.json  --data_field article
```

### Paraphrase test set using dipper, specify the path to the reverse engineered green list in the file

```bash
python generate_paraphrases_with_dipper.py
```

## Experiments

All the python notebooks corresponding to the experiments are inside the folder `experiments`.

## Data

All the text corpus generated to reverse engineer the green list is upload to this [drive folder](https://drive.google.com/drive/folders/1exppLmGA-igwJfuRhOZ_h-4ohb9d0Zwl?usp=sharing).


# BibTeX
```
@inproceedings{rastogi-pruthi-2024-revisiting,
    title = "Revisiting the Robustness of Watermarking to Paraphrasing Attacks",
    author = "Rastogi, Saksham  and
      Pruthi, Danish",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1005/",
    doi = "10.18653/v1/2024.emnlp-main.1005",
    pages = "18100--18110"
}
```

## References
- The `kgw_watermarking` folder is from the [original repo](https://github.com/jwkirchenbauer/lm-watermarking).
- The `paraphrastic_representations_at_scale` is from the [original repo](https://github.com/jwieting/paraphrastic-representations-at-scale).


## Bugs or Questions?
If you have any questions relating to the paper or repository, please don't hesitate to reach out to iitdsaksham@gmail.com.
