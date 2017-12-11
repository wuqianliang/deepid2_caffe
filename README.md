# deepid2_caffe
deepid2 face verification  base on caffe.
下最新版的caffe,把代码合进去就行了。

在lfw数据集上面 训练和测试：

train:
I1209 00:28:19.336725 20145 sgd_solver.cpp:106] Iteration 69800, lr = 0.0001
I1209 00:28:23.152557 20145 solver.cpp:228] Iteration 69900, loss = 0.121022
I1209 00:28:23.152586 20145 solver.cpp:244]     Train net output #0: sim_loss = 0.403397 (* 0.3 = 0.121019 loss)
I1209 00:28:23.152595 20145 solver.cpp:244]     Train net output #1: softmax_loss = 4.43887e-06 (* 0.7 = 3.10721e-06 loss)

test:
I1209 00:33:53.272675 20145 solver.cpp:337] Iteration 76000, Testing net (#0)
I1209 00:34:09.513937 20145 solver.cpp:404]     Test net output #0: accuracy1 = 0.213
I1209 00:34:09.513994 20145 solver.cpp:404]     Test net output #1: sim_loss = 0.0579158 (* 0.3 = 0.0173747 loss)
I1209 00:34:09.514001 20145 solver.cpp:404]     Test net output #2: softmax_loss = 10.4477 (* 0.7 = 7.3

后面补上在youtube face 和MSCELEBv1上面的测试结果。


reference:
https://github.com/wjgaas/DeepID2
http://www.miaoerduo.com/deep-learning/%E5%9F%BA%E4%BA%8Ecaffe%E7%9A%84deepid2%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%AD%EF%BC%89.html

deepid 第一代:
http://blog.csdn.net/real_myth/article/details/51241854

https://github.com/joyhuang9473/deepid-implementation
