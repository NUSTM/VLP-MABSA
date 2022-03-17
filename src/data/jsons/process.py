import json
a = {
    'text_dir': '/home/ylin/MABSA/data/MVSA/data',
    'img_region_dir': '/home/ylin/MABSA/data/MVSA/photo_region',
    'senti_dir': '/home/ylin/MABSA/data/MVSA/sentiment.json',
    'BIO_dir': '/home/ylin/MABSA/data/MVSA/BIO_data',
    'ANP_dir': '/home/ylin/MABSA/DeepSentiBank/result/mvsa_photo_path.json',
    'ANP_clss_dir': '/home/ylin/MABSA/DeepSentiBank/classes.json'
}
# js = json.dump(a)
f = open('MVSA_info.json', 'w')
json.dump(a, f)
