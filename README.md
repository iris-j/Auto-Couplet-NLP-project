# Auto-Couplet-NLP-project
Text analysis of Chinese couplets. HMM and Seq2Seq model for couplet generation.

## Python packages dependency
- BeautifulSoup: html网页解析
- pypinyin：获取汉字的读音和音调
- jieba：中文分词，词性标注
- nltk：获取bigram和频率分布
- tensorflow：神经网络的搭建和训练
- wordcloud/harvest/networkx/pyecharts：数据分析，网络关系图的绘制


## File structure
- auto-couplet/couplet analysis.py：code for text analysis
- auto-couplet/HMM_couplet.py：HMM model
- auto-couplet/images：images generated from text analysis
- seq2seq_couplet/seq2seq_couplet.py：Seq2Seq model
- seq2seq_couplet/data：saved models
- couplet：couplets data

## Data access and cleaning
Many thanks to https://github.com/htw2012/seq2seq_couplet

## Examples
HMM model：
```
春风轻拂千山绿，江山彩润万里和
春风轻拂千山绿，瑞气细润万事丰
春风款款温雨润，旭日冉冉富花香
雪里云山似，花间风景如
九州大地松尤绿，四海神州柏更苍
九州激荡舒清景，四海顿生展宏图
九州曙色含绿水，四海春光有青山
九州大展登云志，四海新开奋月心
九州俊彦振中兴，四海忠人迎富裕
梅报春百花齐放，鹊开道万马奔腾
飞雪片片醉，画图声声歌
江山双归合民意，社稷一震腾春风
```

Seq2Seq model:
```
春风轻拂千山绿，旭日长歌满院春
九州大展登云志，万里长歌展国猷
红梅开岁首，紫气展人家
```
