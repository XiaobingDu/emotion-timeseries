# emotion-timeseries


Code Repository for the paper: "Affect2MM: Affective Analysis of Multimedia Content Using Emotion Causality".

Training and Testing Codes for all three datasets. 
1. SENDv1 Dataset
2. MovieGraphs Dataset
3. LIRIS-ACCEDE Dataset

# PositiveEmotion

1. current
- GC 输出全部是0
- co-attention 权重输出全部是average weights
- 模型最后一层输出如下，模型似乎没有学习能力
- - predict_last......
  - tensor(-0.0411,  0.1370, -0.1754, -0.0580, -0.0578,  0.1431,  0.1562, -0.0996,0.1616
- - dec_out .....
  -  1.3158e-01, -2.6054e-02,  2.1075e-01,  1.3412e-01,  7.6706e-02,
          1.9719e-01,  2.0428e-01,  5.6338e-02,  4.5805e-02,  6.5245e-02,

- KLDivLoss 没有随着epoch而减小，并没有什么变化

2. 未完成
- criterion 未添加
- GCN + label correlation未添加
- dominant emotion loss founction 未添加
- dominant emotion criterion 未添加