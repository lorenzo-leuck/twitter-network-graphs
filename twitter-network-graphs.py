import pandas as pd
import numpy as np
from itertools import chain
import os
import csv
import regex as re
from csv import DictReader
import itertools
import networkx as nx
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import greedy_modularity_communities
import json
from networkx.readwrite import json_graph

arq = "twitter-coletas/bolsonaro-junho-2021/bolsonaro-07-06-21"

arq_json = arq + ".json"
arq_csv = arq + ".csv"
arq_grafo = arq + "-grafo.csv"
arq_rt = arq + "-grafo-rt.csv"
arq_mt = arq + "-grafo-mt.csv"
arq_semantico = arq + "-grafo-semantico.csv"
arq_hashtag = arq + "-grafo-hashtags.csv"

pd_obj = pd.read_json(arq_json)
pd_obj.to_csv(arq_csv, index=False)
file = pd.read_csv(arq_csv)
keep_col = ['screen_name','full_text']
new_file = file[keep_col]
new_file.to_csv(arq_csv, index=False)

outrow = []
with open(arq_csv, "r",encoding='utf-8') as fileIn:  # input file location
    with open(arq_grafo, "w",encoding='utf-8') as fileOut:  # output file location
        writer = csv.writer(fileOut)
        reader = csv.reader(fileIn, delimiter=',')
        writer.writerow(['source','target', "full_text"])
        for row in reader:
            screen_name = row[0]    
            text = row[1]      
            target = ''
            for cell in row:
                target = re.findall(r"@([^\s]+)", cell)
                target = re.sub(r"\[|\'|\:|\]", '', str(target))
                outrow 
            writer.writerow([screen_name, target, text])

df = pd.read_csv(arq_grafo)
df['target'].replace('', np.nan, inplace=True)
df.dropna(subset=['target'], inplace=True)
df.to_csv(arq_grafo, index=False)

df = pd.read_csv(arq_grafo)
rt = df['full_text'].str.startswith('RT @')
df[rt].to_csv(arq_rt, index=False)
df2 = pd.read_csv(arq_rt)
df2['target'].replace(r",[\s\S]*$", "", regex=True, inplace= True)
df2.to_csv(arq_rt, index=False)


def chainer(s):
    return list(chain.from_iterable(s.str.split(',')))

lens = df['target'].str.split(',').map(len)

res = pd.DataFrame({'source': np.repeat(df['source'], lens),
                    'target': chainer(df['target']),
                    'text': np.repeat(df['full_text'], lens)
                    })

res['target'].replace('', np.nan, inplace=True)
res.dropna(subset=['target'], inplace=True)

mt = ~res['text'].str.startswith('RT @')
res[mt].to_csv(arq_mt, index=False)


f = open("stopwords.txt", 'r', encoding='utf-8')
stopwords = [name.rstrip().lower() for name in f]

with open(arq_csv, encoding="utf-8") as f:
    vTweets = [row["full_text"] for row in DictReader(f)]

vFrases = []

for idx,tweet in enumerate(vTweets):
    tweet = re.sub(r"https?:\/\/(\S+)", "", tweet)
    # tira toda a pontuação exceto arroba e jogo da velha
    tweet = re.sub(r"[^\P{P}@#]+", "", tweet)
    # tira toda a pontuação
    # frase = frase.translate(str.maketrans('','',string.punctuation))
    tweet = " ".join([x for x in tweet.split(' ') if x.lower() not in stopwords])
    tweet = tweet.rstrip() 
    tweet = tweet.strip() 
    tweet = tweet.lower()   
    vFrases.append(tweet) 

col1=[]
col2=[]

for frase in vFrases:
    vPalavras = frase.split(' ')
    nPalavras = len(vPalavras)

    if nPalavras > 2:
        nLinhas = 0
        vNumeros = []
        for i in range(nPalavras):
            b = nPalavras-(i+1)
            if b>0:
                vNumeros.append(b)
            nLinhas = nLinhas + b

        nNumeros= len(vNumeros)
        invNumeros = vNumeros[::-1]
        
        c1 = []
        for i in range(nNumeros):
            for j in range(vNumeros[i]):
                c1.append(vPalavras[i])  
        col1.extend(c1)


        ordemC2 = []
        for i in range(nNumeros):
            for j in invNumeros:
                ordemC2.append(j)
            invNumeros.pop(0)

        Mposicao = []
        posicaoC2 = []
        for i in range(int(nLinhas/nPalavras)):
            for j in range(nPalavras):
                posicaoC2.append(vPalavras[j])
                Mposicao.append(j)

        c2 =[]
        c2[:] = [posicaoC2[i] for i in ordemC2]
        col2.extend(c2)

matriz = np.c_[col1,col2]
df = pd.DataFrame(columns=["source","target"], data=matriz)
df['source'].replace('', np.nan, inplace=True)
df.dropna(subset=['source'], inplace=True)
df['target'].replace('', np.nan, inplace=True)
df.dropna(subset=['target'], inplace=True)
df.to_csv(arq_semantico, index=False)


outrow = []
with open(arq_csv, "r",encoding='utf-8') as fileIn:  # input file location
    with open(arq_grafo, "w", encoding='utf-8') as fileOut:  # output file location
        writer = csv.writer(fileOut)
        reader = csv.reader(fileIn, delimiter=',')
        writer.writerow(['usuario','hashtags'])
        for row in reader:
            screen_name = row[0]      
            hashtags = ''
            for cell in row:
                hashtags = re.findall(r"#([^\s]+)", cell)
                hashtags = re.sub(r"\[|\'|\:|\]", '', str(hashtags))
                # hashtags = re.sub('['+string.punctuation+']', '', hashtags)
                hashtags = re.sub(r"[^\P{P},]+", "", hashtags)
                hashtags = re.sub(r"^\s|\s|\s\s","",hashtags)
                outrow 
            writer.writerow([screen_name, hashtags])

dfRW = pd.read_csv(arq_grafo)
dfRW['hashtags'].replace('', np.nan, inplace=True)
dfRW.dropna(subset=['hashtags'], inplace=True)
dfRW.to_csv(arq_grafo, index=False)

with open(arq_grafo, encoding="utf-8") as f:
    vTweets = [row["hashtags"] for row in DictReader(f)]

col1=[]
col2=[]

for tweet in vTweets:
    # print(tweet)
    vPalavras = tweet.split(',')
    nPalavras = len(vPalavras)

    if nPalavras > 2:
        nLinhas = 0
        vNumeros = []
        for i in range(nPalavras):
            b = nPalavras-(i+1)
            if b>0:
                vNumeros.append(b)
            nLinhas = nLinhas + b

        nNumeros= len(vNumeros)
        invNumeros = vNumeros[::-1]
        
        c1 = []
        for i in range(nNumeros):
            for j in range(vNumeros[i]):
                c1.append(vPalavras[i])  
        col1.extend(c1)


        ordemC2 = []
        for i in range(nNumeros):
            for j in invNumeros:
                ordemC2.append(j)
            invNumeros.pop(0)

        Mposicao = []
        posicaoC2 = []
        for i in range(int(nLinhas/nPalavras)):
            for j in range(nPalavras):
                posicaoC2.append(vPalavras[j])
                Mposicao.append(j)

        c2 =[]
        c2[:] = [posicaoC2[i] for i in ordemC2]
        col2.extend(c2)

# usr = pd.DataFrame({'source': np.repeat(df['source'], lens),
usr = dfRW['usuario'].values[0]
n = len(col1)
usrs = [usr]*n

matriz = np.c_[col1,col2,usrs]
df = pd.DataFrame(columns=["source","target","user"], data=matriz)
df['source'].replace('', np.nan, inplace=True)
df.dropna(subset=['source'], inplace=True)
df['target'].replace('', np.nan, inplace=True)
df.dropna(subset=['target'], inplace=True)

df.to_csv(arq_hashtag, index=False)


os.remove(arq_grafo)
os.remove(arq_csv)

quadriGrafo = [arq_rt, arq_mt, arq_semantico,arq_hashtag]

for ix,grafo in enumerate(quadriGrafo): 
    df = pd.read_csv(grafo)
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, create_using=Graphtype)

    for node, in_deg in dict(G.in_degree).items(): 
        G.nodes[node]['in_degree'] = in_deg
        # G.nodes[node]['paridade'] = (1-in_deg % 2)

    indeg = G.in_degree()

    # Mantém somente n perfis mais retuitados, mencionados
    
    if (ix == 2):
        top5 = sorted(indeg, key=lambda x: x[1], reverse=True)[:15]
    else:
        top5 = sorted(indeg, key=lambda x: x[1], reverse=True)[:35]
    
    to_keep = [n for (n, deg) in top5]
    G = G.subgraph(to_keep)

    # Remover pelo grau de entrada(quantidade de rt ou mentions)
    # to_remove = [n for (n, deg) in indeg if deg < 10]
    # G.remove_nodes_from(to_remove)

    # Remove nós isolados 
    G = nx.Graph(G)
    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.Graph(G)

    c = list(greedy_modularity_communities(G))

    nodes = list(G.nodes)

    group = []
    for x in nodes:
        group.extend([i for i, lst in enumerate(c) if x in lst])

    nx.set_node_attributes(G, 1, 'group')
    for i,node in enumerate(G.nodes):
        # print(node)
        G.nodes[node]['group'] = group[i]

    grafoJson = grafo.replace(".csv",".json")
    data = json_graph.node_link_data(G)
    with open(grafoJson, 'w') as f:
        json.dump(data, f, indent=4)

    nomeGrafo = grafo.replace(".csv","")
    html_page_name = grafo.replace(".csv",".html")

    html_page = open(html_page_name, 'w')

    html_page.write('<!DOCTYPE html><html>')

    head = f"""
    <meta charset="UTF-8"> 
    <head>
        <title>{nomeGrafo}</title>
        <script src='https://d3js.org/d3.v5.min.js'></script>
    """

    style = """
        <style>
            #viz {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            }
        </style>
    </head>
    """

    body = """
    <body>
        <svg id='viz'></svg>
    </body>
    """
    script = f"""
    <script>
        var width = 1200;
        var height = 800;
        var color = d3.scaleOrdinal(d3.schemeCategory10);

        d3.json("{grafoJson}").then(function (graph) {{

            var label = {{
                'nodes': [],
                'links': []
            }};

            graph.nodes.forEach(function (d, i) {{
                label.nodes.push({{ node: d }});
                label.nodes.push({{ node: d }});
                label.links.push({{
                    source: i * 2,
                    target: i * 2 + 1
                }});
            }});

            var labelLayout = d3.forceSimulation(label.nodes)
                .force("charge", d3.forceManyBody().strength(-50))
                .force("link", d3.forceLink(label.links).distance(0).strength(2));

            var graphLayout = d3.forceSimulation(graph.nodes)
                .force("charge", d3.forceManyBody().strength(-3000))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("x", d3.forceX(width / 2).strength(1))
                .force("y", d3.forceY(height / 2).strength(1))
                .force("link", d3.forceLink(graph.links).id(function (d) {{ return d.id; }}).distance(50).strength(1))
                .on("tick", ticked);

            var adjlist = [];

            graph.links.forEach(function (d) {{
                adjlist[d.source.index + "-" + d.target.index] = true;
                adjlist[d.target.index + "-" + d.source.index] = true;
            }});

            function neigh(a, b) {{
                return a == b || adjlist[a + "-" + b];
            }}


            var svg = d3.select("#viz").attr("width", width).attr("height", height);
            var container = svg.append("g");

            svg.call(
                d3.zoom()
                    .scaleExtent([.1, 4])
                    .on("zoom", function () {{ container.attr("transform", d3.event.transform); }})
            );

            var link = container.append("g").attr("class", "links")
                .selectAll("line")
                .data(graph.links)
                .enter()
                .append("line")
                .attr("stroke", "#aaa")
                .attr("stroke-width", "1px");

            var node = container.append("g").attr("class", "nodes")
                .selectAll("g")
                .data(graph.nodes)
                .enter()
                .append("circle")
                .attr("r", 5)
                .attr("fill", function (d) {{ return color(d.group); }})

            node.on("mouseover", focus).on("mouseout", unfocus);

            node.call(
                d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended)
            );

            var labelNode = container.append("g").attr("class", "labelNodes")
                .selectAll("text")
                .data(label.nodes)
                .enter()
                .append("text")
                .text(function (d, i) {{ return i % 2 == 0 ? "" : d.node.id; }})
                .style("fill", "#555")
                .style("font-family", "Arial")
                .style("font-size", d => d.node.in_degree/3)
                .style("pointer-events", "none"); // to prevent mouseover/drag capture

            node.on("mouseover", focus).on("mouseout", unfocus);

            function ticked() {{

                node.call(updateNode);
                link.call(updateLink);

                labelLayout.alphaTarget(0.3).restart();
                labelNode.each(function (d, i) {{
                    if (i % 2 == 0) {{
                        d.x = d.node.x;
                        d.y = d.node.y;
                    }} else {{
                        var b = this.getBBox();

                        var diffX = d.x - d.node.x;
                        var diffY = d.y - d.node.y;

                        var dist = Math.sqrt(diffX * diffX + diffY * diffY);

                        var shiftX = b.width * (diffX - dist) / (dist * 2);
                        shiftX = Math.max(-b.width, Math.min(0, shiftX));
                        var shiftY = 16;
                        this.setAttribute("transform", "translate(" + shiftX + "," + shiftY + ")");
                    }}
                }});
                labelNode.call(updateNode);

            }}

            function fixna(x) {{
                if (isFinite(x)) return x;
                return 0;
            }}

            function focus(d) {{
                var index = d3.select(d3.event.target).datum().index;
                node.style("opacity", function (o) {{
                    return neigh(index, o.index) ? 1 : 0.1;
                }});
                labelNode.attr("display", function (o) {{
                    return neigh(index, o.node.index) ? "block" : "none";
                }});
                link.style("opacity", function (o) {{
                    return o.source.index == index || o.target.index == index ? 1 : 0.1;
                }});
            }}

            function unfocus() {{
                labelNode.attr("display", "block");
                node.style("opacity", 1);
                link.style("opacity", 1);
            }}

            function updateLink(link) {{
                link.attr("x1", function (d) {{ return fixna(d.source.x); }})
                    .attr("y1", function (d) {{ return fixna(d.source.y); }})
                    .attr("x2", function (d) {{ return fixna(d.target.x); }})
                    .attr("y2", function (d) {{ return fixna(d.target.y); }});
            }}

            function updateNode(node) {{
                node.attr("transform", function (d) {{
                    return "translate(" + fixna(d.x) + "," + fixna(d.y) + ")";
                }});
            }}

            function dragstarted(d) {{
                d3.event.sourceEvent.stopPropagation();
                if (!d3.event.active) graphLayout.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}

            function dragged(d) {{
                d.fx = d3.event.x;
                d.fy = d3.event.y;
            }}

            function dragended(d) {{
                if (!d3.event.active) graphLayout.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}

        }});
    </script>
    </html>
    """

    html_page.write(head)
    html_page.write(style)
    html_page.write(body)
    html_page.write(script)
    html_page.close()
