#### 多种机器学习训练，绘制ROC曲线

library(randomForest)
library(pROC)
library(e1071)
library(nnet)
library(MASS)
library(mda)
library(xgboost)
library(catboost)
# devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.20/catboost-R-Windows-0.20.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
library(caret)
library(PRROC)
library(openxlsx)
library(Boruta)
library(tidyverse)
library(tableone)
library(gbm)
library(MLmetrics)
library(C50)
library(kernlab)
library(naivebayes)
library(mda)
library(earth)
library(mlbench)
names(getModelInfo())

## 一些自定义函数
MySummary  <- function(data, lev = NULL, model = NULL){
  a1 <- defaultSummary(data, lev, model)
  b1 <- twoClassSummary(data, lev, model)
  c1 <- prSummary(data, lev, model)
  out <- c(a1, b1,c1)
  out}
myControl = trainControl(method = "cv", number = 5, verboseIter = FALSE)
fitControl <- trainControl(#   
  method = "cv",#   
  number = 5,
  summaryFunction=MySummary,
  classProbs=T,#   
  savePredictions = T,#   
  verboseIter = F)# 
myControl=fitControl

set.seed(12)
#Logistic regression
# install.packages('MLmetrics')
model_lm =train(Type ~ ., 
                data = traindata, 
                method = "glm",
                family = "binomial",
                trControl = fitControl)
#RandomForest
traindata$Type = as.factor(traindata$Type)
model_rf = train(Type ~ ., 
                 data = traindata, 
                 tuneLength = 1,
                 method = "ranger",
                 importance = 'impurity',
                 num.trees=500,
                 keep.forest=TRUE,
                 trControl = fitControl)
#SVM
model_svm = train(Type ~ ., 
                  data = traindata,
                  method='svmRadial' , 
                  verbose=F,
                  metric = "ROC",   
                  tuneLength = 10,
                  trControl = myControl,
                  importance = TRUE)
#NaiveBayes
model_bayes = train(Type ~ ., 
                    data = traindata,
                    method = "naive_bayes", 
                    linout = TRUE,
                    metric = "AUC",
                    trControl = myControl)

#Neural Network 
if(nrow(traindata)>1000){
  model_nnet = train(Type ~ ., 
                     data = traindata,
                     method='nnet',
                     trControl = myControl) 
}

#Linear Discriminant Analysis
model_lda = train(Type ~ ., 
                  data = traindata,
                  method='lda',
                  trControl = myControl) 

#Mixture Discriminant Analysis
model_mda = train(Type ~ ., 
                  data = traindata,
                  method='mda',
                  trControl = myControl)

#Flexible Discriminant Analysis
model_fda = train(Type ~ ., 
                  data = traindata,
                  method='fda',
                  trControl = myControl) 

# Gradient boosting machine
model_gbm = train(Type ~ ., 
                  data = traindata,
                  tuneLength = 2,
                  method = "gbm",
                  metric = "AUC",
                  trControl = myControl)

#XGboost
model_xgboost = train(Type ~ ., 
                      data = traindata,
                      method='xgbTree',
                      trControl = myControl)
#catboost
# 使用catboost.caret方法	
model_catboost <- train(traindata[,-1], traindata$Type,
               method = catboost.caret,	
               logging_level = 'Silent', 
               preProc = NULL,trControl = myControl)	


####################################################################################
### Out put the Training ROC
####################################################################################
library(MLeval)
pdf('Model_plot.pdf',width = 10,height = 8)
df_trainlist = evalm(list(model_lm, model_rf,model_svm,
                          model_bayes,model_lda,model_mda,
                          model_fda,model_gbm,model_xgboost,
                          model_catboost), 
                     gnames=c("Logistic Regression","Random Forest", "SVM", 
                              "Naive Bayes",
                              "Linear Discriminant Analysis", 
                              "Mixture Discriminant Analysis", 
                              "Flexible Discriminant Analysis",
                              "Gradient Boosting Machine",
                              "XGBoost","CatBoost"))
dev.off()

## plot training ROC
### 重新绘制
df_trainROC =df_trainlist$roc$data 
colnames(df_trainROC)[3] <- 'Model'
# colnames(df_trainROC)=colnames(dfROC)
col <- paletteer_d("basetheme::deepblue")

ggplot() + 
  geom_line(data = df_trainROC,aes(x = FPR, y = SENS, color = Model),size=1.5) + 
  geom_line(aes(x=c(0,1),y=c(0,1)),color = "grey",size = 1.5,linetype=6 )+
  theme_bw()+
  # annotate("text",x = .75, y = .25,
  #          label = paste("AUC of min = ",round(auc_min,2)),color = "blue",size=10)+
  scale_x_continuous(name  = "1-Specificity")+
  scale_y_continuous(name = "Sensitivity")+ 
  theme(plot.title = element_text(hjust = 0.5,size = 24),
        axis.text=element_text(size=24),
        axis.title.x = element_text(size = 24),
        axis.title.y = element_text(size = 24),
        legend.title = element_text(size=24),
        legend.text = element_text(size=18)) + 
  scale_color_manual(values = col)
ggsave("ROC_train.pdf",width = 10,height = 8)




