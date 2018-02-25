import text_utils
tu = text_utils.TextUtility(min_df=5, max_df=0.9)
textf = '/home/kd/data/AmazonCat-13K/text_feat.txt'
tu.fit(textf)
