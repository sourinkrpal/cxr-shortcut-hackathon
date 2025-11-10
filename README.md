# Pneumothorax-chest drain shortcut learning

Project @ [UBRA AI Toolbox Hackathon](https://www.bremen-research.de/en/ai-toolbox-hackathon)

## Getting set up
- [ ] Create a GitHub account (if you do not have one yet)
- [ ] Fork this repository and give all group members access to the fork; I suggest using this as the primary code synchronization method
- [ ] Create a [Weights & Biases](https://wandb.ai/) account (if you do not have one yet - one per group is sufficient)
- [ ] Take notice of the CheXpert dataset research use agreement and create an individual account [here](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)
- [ ] [Get onto the VM](VM.md) and verify you can run cxp_pneu.py
- [ ] Go through [the baseline code](cxp_pneu.py); make sure you understand it
- [ ] Implement [the Serna et al. approach](https://www.sciencedirect.com/science/article/pii/S0004370222000224?via%3Dihub) as a potential shortcut learning mitigation method
- [ ] Go through the list of open issues; address what you find interesting or explore your own ideas freely

## Reading materials (optional)
General shortcut learning:
- [Shortcut learning in deep neural networks](https://www.nature.com/articles/s42256-020-00257-z)
- [The risk of shortcutting in deep learning algorithms for medical imaging research](https://www.nature.com/articles/s41598-024-79838-6)

Papers that cover pneumothorax/chest drain shortcut learning:
- [Hidden stratification causes clinically meaningful failures in machine learning for medical imaging](https://dl.acm.org/doi/10.1145/3368555.3384468)
- [DETECTING SHORTCUTS IN MEDICAL IMAGES - A CASE STUDY IN CHEST X-RAYS](https://arxiv.org/pdf/2211.04279)
- [Slicing Through Bias: Explaining Performance Gaps in Medical Image Analysis using Slice Discovery Methods](https://arxiv.org/html/2406.12142v2)

Drawbacks of DANN / CDANN (baseline method for 'domain invariance' which can also be used to address shortcut learning):
- [Fundamental Limits and Tradeoffs in Invariant Representation Learning](https://www.jmlr.org/papers/v23/21-1078.html)
- [10 Years of Fair Representations: Challenges and Opportunities](https://arxiv.org/abs/2407.03834)
- [Are demographically invariant models and representations in medical imaging fair?](https://arxiv.org/html/2305.01397v3)
- [The limits of fair medical imaging AI in real-world generalization](https://www.nature.com/articles/s41591-024-03113-4)
- [MEDFAIR: Benchmarking Fairness for Medical Imaging](https://arxiv.org/abs/2210.01725)

Alternative approach pursued here: [Sensitive loss: Improving accuracy and fairness of face representations with discrimination-aware deep learning](https://www.sciencedirect.com/science/article/pii/S0004370222000224)