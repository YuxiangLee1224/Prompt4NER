# A Hybrid Prompt Method for Few-shot NER

### Requirement
```
python >= 3.6.13
torch >= 1.10.2
transformers >= 4.10.2
```

We have provided the `config.json` file in the `premodel` folder. Please use the provided config file instead of the one downloaded from Hugging Face.

### Ablation study

![](https://github.com/YuxiangLee1224/Prompt4NER/blob/master/as.png)

To analyze the effectiveness of hard prompts and delayed soft prompts, we conducted the follow-
ing experiments separately: 1):baseline w/ MRC prompts; 2):baseline w/ soft prompts; 3):baseline w/ late soft prompts;
4):baseline w/ (MRC prompts, late soft prompts).

As our paper focuses on few-shot scenarios, and our experimental results demonstrate more significant
advantages with fewer samples, we chose K = 10 and K = 20 for the ablation study. From the experimental results in the table, it can be observed that soft prompts, especially delayed soft prompts, significantly
enhance the model’s performance. In contrast, the benefits from using MRC as a hard prompt are less
consistent.

For the results of the ablation study, we provide the following explanations: Firstly, according to the
No Free Lunch Theorem in machine learning, the model’s performance in this way is expected. Specifically, as the number of samples available for training increases, the so-called accessible prior knowledge
(hard prompts) becomes less important and can even have a counterproductive effect, especially in more
complex datasets. This also indicates that hard prompts designed for human interpretability (discrete
prompts) may not necessarily benefit the model. On the contrary, this aligns with the claims made when
introducing soft prompts. Soft prompts, by continuously adjusting themselves during the training pro-
cess, demonstrate better performance.

In conclusion, the above experiments demonstrate the effectiveness of our delayed soft prompts and
MRC prompts in the context of few-shot tasks.
   