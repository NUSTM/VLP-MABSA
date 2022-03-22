# Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis(VLP-MABSA)
Codes and datasets for our ACL'2022 paper:[Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis]()

Author

Yan Ling

ylin@njust.edu.cn

## Data Process
The pre-training dataset we use is MVSA-Multi. You can get from this [git](https://github.com/xunan0812/MultiSentiNet).
- For texts in MVSA-Multi dataset, we first use [twitter_nlp](https://github.com/aritter/twitter_nlp) to perform Named Entity Recognition in order to find the aspects. Then, we use the sentiment lexicon [sentiwordnet](https://github.com/zeeeyang/lexicon_rnn/tree/master/lexicons) to matching the opinion words.
- For images in MVSA-Multi dataset, we first perform [Faster-RCNN](https://github.com/jiasenlu/bottom-up-attention) to extract the region feature(only retain 36 regions with highest Confidence) as the input feature. Then we use [ANPs extractor](https://github.com/stephen-pilli/DeepSentiBank) to extract the ANPs distribution of each image.
## Data Download
Because the pre-training dataset after processing is very large, so we only provide the downstream datasets. You can download the downstream datasets and our pre-training model via the link [Baidu Netdist](https://pan.baidu.com/s/11INRcFpoBR-6iggukx1VtA) with code:d0tn.
## Main Task
To Train the downstream JMASA task on two twitter datasets, you can just run the following code. 
```
sh 15_pretrain_full.sh
sh 17_pretrain_full.sh
```
The following is the description of parameters of the above shell
```

```
