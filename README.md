# Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis(VLP-MABSA)
Codes and datasets for our ACL'2022 paper:[Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis](https://aclanthology.org/2022.acl-long.152/)

Author

Yan Ling

ylin@njust.edu.cn

## Data Processing
The pre-training dataset we use is MVSA-Multi. You can get from this [git](https://github.com/xunan0812/MultiSentiNet). At first, you need to use the judgement rules provided by the git to remove the samples with inconsistent labels.
### Text Processing
For texts in MVSA-Multi dataset, we first use NLTK to perform the tokenization.
- How to obtain aspects

We use [twitter_nlp](https://github.com/aritter/twitter_nlp) to perform Named Entity Recognition in order to find the aspects.
For example, given the text "It is unbelievable ! Stephen Curry won the game !". The result of NER through twitter_nlp is
```
It/O is/O unbelievable/O !/O Stephen/B-ENTITY Curry/I-ENTITY won/O the/O game/O !/O
```
We save the result in a dict with the following format
```
{"text_id":{"aspect_spans":[list of aspect spans],"aspect_texts":[list of aspect texts]},...}
{"100":{"aspect_spans":[[4,5]],"aspect_texts":[["Stephen","Curry"]},...}
```
- How to obtain opinions

We use the sentiment lexicon [sentiwordnet](https://github.com/zeeeyang/lexicon_rnn/tree/master/lexicons) to matching the opinion words. The lexicon is adopted as a dictionary. The words in the text which belongs to the lexicon are considered as the opinion terms.
Using the text above as an example, the word "nice" belongs to the lexicon. We save the infomation with the same format of aspect spans.
```
{"text_id":{"opinion_spans":[list of opinion spans],"opinion_texts":[list of opinion texts]},...}
{"100":{"opinion_spans":[[2,2]],"opinion_texts":[["unbelievable"]},...}
```

The dics of aspect spans and opinion spans are used for the AOE pre-training task.
### Image Processing
- How to obtain the features of images

For images in MVSA-Multi dataset, we perform [Faster-RCNN](https://github.com/jiasenlu/bottom-up-attention) to extract the region feature(only retain 36 regions with highest Confidence) as the input feature and the dimension of each region feature is 2048. For the details of how to perform Faster-RCNN, you can refer to the [Faster-RCNN](https://github.com/jiasenlu/bottom-up-attention).
- How to obtain the ANP of each image

We employ [ANPs extractor](https://github.com/stephen-pilli/DeepSentiBank) to predict the ANPs distribution of each image.
To run the code, you need to provide the list of image paths like
```
/home/MVSA/data/2499.jpg
/home/MVSA/data/2500.jpg
...
```
The result of [ANPs extractor](https://github.com/stephen-pilli/DeepSentiBank) is a dict
```
{
    "numbers":100             #the number of images
    "images":[
        {
             "bi-concepts": {
                   handsome_guy: 0.13, # probability of each ANP in descending order 
                   cute_boy: 0.08,
                   ...
              },
              "features": [ 
              ...
              ]
         },
         ...
}
```
The ANP with the highest probability is chosen as the output text of the AOG pre-training task.
### Sentiment Processing
As introduced in the [git](https://github.com/xunan0812/MultiSentiNet), there are many tweets, in which the labels of text and image are inconsistent. Firstly, you need to adopt the judgement rule defined by the author to remove the incosistent data.
Then we save the sentiment labels with the following format
```
{"data_id": sentiment label}
{"13357": 2, "13356": 0,...} # 0,1,2 denote negtive, neutral and positive, respectively
```
For more details, we provide the description of our pre-training data files in **src/data/jsons/MVSA_descriptions.txt** which explains the files defined in **src/data/jsons/MVSA.json**.
## Data Download
Because the pre-training dataset after processing is very large, we only provide the downstream datasets. You can download the downstream datasets and our pre-training model via [Baidu Netdist](https://pan.baidu.com/s/11INRcFpoBR-6iggukx1VtA) with code:d0tn or [Google Drive](https://drive.google.com/drive/folders/1rm0FtHOTMUfZfRjWIE9Ukn_1D5MDXQy3?usp=sharing)
## Pre-Training
If you have done all the processing above, you can perform the pre-training by running the code as follows.
```
sh MVSA_pretrain.sh
```
## Downstream Task Training
To Train the downstream JMASA task on two twitter datasets, you can just run the following code. Note that you need to change all the file path in file **src\data\jsons\twitter15_info.json** and **src\data\jsons\twitter17_info.json** to your own path.
```
sh 15_pretrain_full.sh
sh 17_pretrain_full.sh
```
The following is the description of some parameters of the above shell
```
--dataset           include dataset name and the path of info json file.
--checkpoint_dir    path to save your training model
--log_dir           path to save the training log
--checkpoint        path of the pre-training model
```
We also provide our training logs on two datasets in folder **./log**.  
## Acknowledgements
- Some codes are based on the codes of [BARTABSA](https://github.com/yhcc/BARTABSA) and [KM-BART](https://github.com/FomalhautB/KM-BART), many thanks!
