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
- KLDivLoss 没有随着epoch而减小，并没有什么变化

2. 未完成
- criterion 未添加
- GCN + label correlation未添加
