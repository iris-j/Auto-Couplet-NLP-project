# -*- coding: utf-8 -*-
from HMM_couplet import *
import matplotlib


def gen_word_cloud(fd):  # 输入频率词典，生成词云图，最多选取100个词
    background = imread('images/timg.jpg')
    wc = WordCloud(background_color='white', max_words=100, mask=background, max_font_size=100, stopwords=STOPWORDS,
                   font_path="C:/Windows/Fonts/simfang.ttf", random_state=42)
    wc.generate_from_frequencies(fd)
    image_colors = ImageColorGenerator(background)
    plt.imshow(wc)
    plt.axis('off')
    plt.figure()
    # plt.imshow(wc.recolor(color_func=image_colors))
    # plt.axis('off')
    plt.show()
    wc.to_file('images/duilian.png')


def plot_zhexian(len, zishu, shanglian_yinlv, xialian_yinlv):  # 输入对联长度，画出该长度对联的音律折线图
    plt.title(u'%s字对联的平仄' % zishu)
    x_axix = [i+1 for i in range(0, len)]
    plt.plot(x_axix, [shanglian_yinlv[i+1] for i in range(len)], color='skyblue', label=u'上联平仄')
    plt.plot(x_axix, [xialian_yinlv[i+1] for i in range(len)], color='gray', label=u'下联平仄')
    plt.legend()
    plt.xlabel(u'位置')
    plt.ylabel(u'音律')
    plt.xticks([i+1 for i in range(len)])
    plt.savefig('images/%s字对联的平仄.jpg'%len)
    plt.show()


def gen_network_graph(fd, docs):  # 输入频率词典和原始文本，将最高频的100个名词意象作为网络节点，构建共现网络关系
    ht = HarvestText()
    for i in fd.most_common(100):
        ht.add_new_entity(i[0], i[0], "名词")
    ht.set_linking_strategy("freq")
    G = ht.build_entity_graph(docs, used_types=["名词"])
    important_nodes = [node for node in G.nodes if G.degree[node] >= 5]  # 只保留重要的节点
    G_sub = G.subgraph(important_nodes).copy()
    nodes = [{"name": "结点1", "value": 0, "symbolSize": 10} for _ in range(G_sub.number_of_nodes())]
    for i, name0 in enumerate(G_sub.nodes):
        nodes[i]["name"] = name0
        nodes[i]["value"] = G_sub.degree[name0]
        nodes[i]["symbolSize"] = G_sub.degree[name0] / 10.0
    links = [{"source": "", "target": ""} for i in range(G_sub.number_of_edges())]
    for i, (u, v) in enumerate(G_sub.edges):
        links[i]["source"] = u
        links[i]["target"] = v
        links[i]["value"] = G_sub[u][v]["weight"]

    eg = Graph('意象关系图')
    eg.add('意象', nodes, links)
    eg.render("images/意象关系图.html")
    return eg


test = Process()
shanglian, xialian = test.get_data()
all_word = []
shanglian_ping = defaultdict(int)
shanglian_ze = defaultdict(int)
xialian_ping = defaultdict(int)
xialian_ze = defaultdict(int)
noun = []
for i in range(len(shanglian)):
    sent1 = shanglian[i]
    sent2 = xialian[i]
    sent1_pos, sent1_pure_pos, pure_word1, reverse_pos1 = test.cut_and_pos(sent1)
    sent2_pos, sent2_pure_pos, pure_word2, reverse_pos2 = test.cut_and_pos(sent2)
    all_word = all_word + pure_word1 + pure_word2
    for j in range(len(reverse_pos1)):
        if reverse_pos1[j][0].startswith('n'):
            noun.append(reverse_pos1[j][1])
    for j in range(len(reverse_pos2)):
        if reverse_pos2[j][0].startswith('n'):
            noun.append(reverse_pos2[j][1])

fd = FreqDist(noun)
print(fd.most_common(50))


for i in range(len(shanglian)):
    sent1 = shanglian[i]
    sent2 = xialian[i]
    for j in range(len(sent1)):
        if len(sent1) != 7:
            continue
        if tone(sent1[j]):
            shanglian_ping[j+1] = shanglian_ping[j+1]+1
        else:
            shanglian_ze[j+1] = shanglian_ze[j+1]+1
        if tone(sent2[j]):
            xialian_ping[j+1] = xialian_ping[j+1]+1
        else:
            xialian_ze[j+1] = xialian_ze[j+1]+1

print(shanglian_ping)
print(shanglian_ze)
print(xialian_ping)
print(xialian_ze)
shanglian_yinlv = defaultdict(float)
xialian_yinlv = defaultdict(float)
for i in range(len(shanglian_ping)):
    shanglian_yinlv[i+1] = shanglian_ping[i+1]/(shanglian_ping[i+1]+shanglian_ze[i+1])
    xialian_yinlv[i+1] = xialian_ping[i+1]/(xialian_ping[i+1]+xialian_ze[i+1])
print(shanglian_yinlv)
print(xialian_yinlv)
font = {'family':'SimHei'}
matplotlib.rc('font',**font)
plot_zhexian(7, '七', shanglian_yinlv, xialian_yinlv)

