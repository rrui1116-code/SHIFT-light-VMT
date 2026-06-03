# SHIFT
The code implementation of the EMNLP 2025 main conference paper *[SHIFT: Selected Helpful Informative Frame for Video-guided Machine Translation](https://aclanthology.org/2025.emnlp-main.161/)*


All code in this repository is for **academic research purposes only, and any form of commercial use is strictly prohibited**.

This is the refactored version of my code. If you have any questions or suggestions, please feel free to discuss them with me.

If this resource is useful to your work, we kindly encourage you to cite it.
```text
@inproceedings{guan-etal-2025-shift,
    title = "{SHIFT}: Selected Helpful Informative Frame for Video-guided Machine Translation",
    author = "Guan, Boyu  and
      Han, Chuang  and
      Zhang, Yining  and
      Liang, Yupu  and
      Zhang, Zhiyang  and
      Zhao, Yang  and
      Zong, Chengqing",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.161/",
    doi = "10.18653/v1/2025.emnlp-main.161",
    pages = "3249--3267",
    ISBN = "979-8-89176-332-6",
    abstract = "Video-guided Machine Translation (VMT) aims to improve translation quality by integrating contextual information from paired short video clips. Mainstream VMT approaches typically incorporate multimodal information by uniformly sampling frames from the input videos. However, this paradigm frequently incurs significant computational overhead and introduces redundant multimodal content, which degrades both efficiency and translation quality. To tackle these challenges, we propose SHIFT (Selected Helpful Informative Frame for Translation). It is a lightweight, plug-and-play framework designed for VMT with Multimodal Large Language Models (MLLMs). SHIFT adaptively selects a single informative key frame when visual context is necessary; otherwise, it relies solely on textual input. This process is guided by a dedicated clustering module and a selector module. Experimental results demonstrate that SHIFT enhances the performance of MLLMs on the VMT task while simultaneously reducing computational cost, without sacrificing generalization ability. The code will be released upon acceptance."
}
```
