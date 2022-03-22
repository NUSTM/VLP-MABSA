# Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis(VLP-MABSA)
Codes and datasets for our ACL'2022 paper:[Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis]()

Author

Yan Ling

ylin@njust.edu.cn

## Data Process
- Download the datasets including the pre-training dataset MVSA-Multi and downstream task datasets TWITTER-15/17 via the link 
- For texts in MVSA-Multi dataset, we first use [twitter_nlp](https://github.com/aritter/twitter_nlp) to perform Named Entity Recognition in order to find the aspects. The results are saved in file aspect_spans.json. Then, we use the sentiment lexicon [sentiwordnet](https://github.com/zeeeyang/lexicon_rnn/tree/master/lexicons) to matching the opinion words and the results are saved in opinion_spans_sentiwordnet.json.
- For images in MVSA-Multi dataset, we first perform [Faster-RCNN](https://github.com/jiasenlu/bottom-up-attention) to extract the region feature(only retain 36 regions with highest Confidence) as the input feature. Then we use [ANPs extractor](https://github.com/stephen-pilli/DeepSentiBank) to extract the ANPs distribution of each image which is saved as ANP.json.
- For the sentiments of image-text pairs, we save them in file sentiment.json.
- 
## Main Task
To Train the JMASA task on two twitter datasets, you can just run the following code.
'''python
sh 15_pretrain_full.sh
sh 17_pretrain_full.sh
'''
