setwd("C:/Users/Zhou1314/Desktop/07.ML_table")

## 自定义函数
source("./MLresult.R")
source('./ML_classfiers_Table.R')

### 数据准备
data <- read.table('Matrix.txt',header = T,sep = '\t',check.names = F)
rownames(data) <- paste0('sample',rownames(data))
info <- data[,'status',drop=F] ## 预测结局
colnames(info) <- 'group'
## 删除预测结局
data <- as.data.frame(t(data[,2:ncol(data)])) # 行为基因列为样本

## 预测变量
Selected.Genes <- rownames(data) ## 特征变量

### 运行
# data 是除了分组信息之外的其他表达矩阵
# info 是分组信息，也就是预测结局
# Selected.Genes 是特征变量
# table.name 是输出文件的名字
# N 是随机训练的次数
# N.split 是划分训练集和测试机的比例
## 测试一下
ML_classfiers_Table(data=data, info=info, 
                    Selected.Genes = Selected.Genes, 
                    table.name = 'ML_predict', 
                    N=1,N.split=0.6)

## 循环迭代并且画图
Num <- 10 ## 迭代次数
for (i in seq(Num)) {
  ML_classfiers_Table(data=data, info=info, 
                      Selected.Genes = Selected.Genes, 
                      table.name = paste(i,'ML_predict',sep = '_'), 
                      N=i,N.split=0.6)
}

### 可视化
## plot training ROC
### 重新绘制
library(ggplot2)
library(paletteer)
col <- paletteer_d("basetheme::deepblue")

df_trainROC <- read.csv('ML_predict.csv',header = T)
colnames(df_trainROC)[1] <- 'Model'

## 循环读取文件
data <- list()
for(j in 1:Num){ 
  path <- paste0(i, "_ML_predict.csv")
  tmp <- read.csv(file = path, header = TRUE)
  colnames(tmp)[1] <- 'Model'
  tmp$Random <- j
  data[[j]] <- tmp
}
data <- do.call(rbind,data)

## Training 
p1 <- ggplot(data, aes(x = Random, y = Train.Accu, color = Model,
                 group = Model)) +
  geom_line(lwd = .8) +
  geom_point()+
  scale_color_manual(values = col) + 
  scale_y_continuous(limits = c(0, 1))+
  labs(x = "Number of Random", y = "Average AUC") +
  theme_bw() +#去掉背景灰色
  theme(panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
        legend.position = c(.85,.3),#更改图例的位置，放至图内部的右下角
        legend.box.background = element_rect(color="black"))+#为图例田间边框线
  scale_x_continuous(limits = c(1,10),breaks = seq(1,10,1)) +#更改横坐标刻度值
  ggtitle('Training Dataset')
ggsave('AUCPlot_train.pdf',p1, width = 8,height = 6)

## Testing 
p2 <- ggplot(data, aes(x = Random, y = Test.Accu, color = Model,
                       group = Model)) +
  geom_line(lwd = .8) +
  geom_point()+
  scale_color_manual(values = col) + 
  scale_y_continuous(limits = c(0, 1))+
  labs(x = "Number of Random", y = "Average AUC") +
  theme_bw() +#去掉背景灰色
  theme(panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
        legend.position = c(.85,.3),#更改图例的位置，放至图内部的右下角
        legend.box.background = element_rect(color="black"))+#为图例田间边框线
  scale_x_continuous(limits = c(1,10),breaks = seq(1,10,1)) +#更改横坐标刻度值
  ggtitle('Testing Dataset')
ggsave('AUCPlot_test.pdf',p2, width = 8,height = 6)

