# deepid2_caffe
deepid2 face verification  base on caffe.
下最新版的caffe,把代码合进去就行了。
reference:

https://github.com/wjgaas/DeepID2

http://www.miaoerduo.com/deep-learning/%E5%9F%BA%E4%BA%8Ecaffe%E7%9A%84deepid2%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%AD%EF%BC%89.html


在lfw数据集上面 训练和测试：

I1209 00:28:19.336725 20145 sgd_solver.cpp:106] Iteration 69800, lr = 0.0001
I1209 00:28:23.152557 20145 solver.cpp:228] Iteration 69900, loss = 0.121022
I1209 00:28:23.152586 20145 solver.cpp:244]     Train net output #0: sim_loss = 0.403397 (* 0.3 = 0.121019 loss)
I1209 00:28:23.152595 20145 solver.cpp:244]     Train net output #1: softmax_loss = 4.43887e-06 (* 0.7 = 3.10721e-06 loss)

后面补上在youtube face 和MSCELEBv1上面的测试结果。
