# 图Embedding的介绍

相比item2vec，图Embedding的方法在于生成的训练集不同：

* DeepWalk:随机起点，然后对于无权图进行深度探索，从而生成大量的物品序列（即训练集）；对于有权图，可以计算跳转到下一个节点的概率，然后进行探索。
* Node2Vec:两个概念（同质性：DFS，设置参数q，结构性：BFS，设置参数p）,通过调整p,q，使生成embedding偏向同质性或结构性，$\omega_{vx}:节点v与节点x之间的权值，d(t,x)节点t与节点x之间的距离$。

$$
p_{vx}=\alpha _{pq}(t,x) * \omega _{vx} ,t表示v的上一节点
$$

$$
\alpha _{pq}(t,x)=\begin{array}{l}
\left\{\begin{matrix}
\frac{1}{p},if\space d(t,x)=0 \\
1,if\space d(t,x)=1\\
\frac{1}{q},if\space d(t,x)=2
\end{matrix}\right.
\end{array}
$$

$$

$$

$$

$$

* EGES:引入补充信息，对补充信息还是使用相关Embedding的方式，最后利用注意力机制的方法，为每一个Embedding训练一个权重，对多个embedding进行组合（代替传统的‘sum-pooling’ or ‘average-pooling’）,最后对求取的权值数可以使用对数进行映射 ----  $e^{w_i}$. 该方式可以一定程度解决‘冷启动’的问题

