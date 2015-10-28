__author__ = 'hs'
import jieba

seg_list_cn = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list_cn))

seg_list_tw = jieba.cut("中文斷詞前言自然語言處理的其中一個重要環節就是中文斷詞的")  # 默认是精确模式
print(", ".join(seg_list_tw))