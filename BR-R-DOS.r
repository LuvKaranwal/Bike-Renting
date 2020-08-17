rm(list=ls())
brp=read.csv("day.csv",header = T,na.strings = c(" ","","NA"))
str(brp)

x=c("ggplot2","corrgram","DMwR","caret","randomForest","unbalanced","C50","inTrees",
    "dummies","e1071","Information","MASS","rpart","gbm","ROSE","sampling","RRF","AICcmodavg","Metrics","glmnet")
lapply(x,require,character.only = TRUE)
rm(x)

# sample case values: 2012-12-26	1	1	12	0	3	1	3	0.243333	0.220333	0.823333	0.316546	9	432	441

print("sample case values: 2012-12-26 1  1  12  0  3  1  3  0.243333  0.220333  0.823333  0.316546  9  432  441")
print("Enter dteday: ")
dteday = as.integer(readLines("stdin",n=1))
print("Enter season: ")
season = as.integer(readLines("stdin",n=1))
print("Enter yr: ")
yr = as.integer(readLines("stdin",n=1))
print("Enter mnth: ")
mnth = as.integer(readLines("stdin",n=1))
print("Enter holiday: ")
holiday = as.integer(readLines("stdin",n=1))
print("Enter weekday: ")
weekday = as.integer(readLines("stdin",n=1))
print("Enter workingday: ")
workingday = as.integer(readLines("stdin",n=1))
print("Enter weathersit: ")
weathersit = as.integer(readLines("stdin",n=1))
print("Enter temp: ")
temp = as.double(readLines("stdin",n=1))
print("Enter atemp: ")
atemp = as.double(readLines("stdin",n=1))
print("Enter hum: ")
hum = as.double(readLines("stdin",n=1))
print("Enter windspeed: ")
windspeed = as.double(readLines("stdin",n=1))


names(brp)[names(brp) == "mnth"] <- "month"
names(brp)[names(brp) == "cnt"] <- "count"
str(brp)

# (1).Box plot
options(warn=-1)
# brp$casual = toString(brp$casual)
# brp$registered = toString(brp$registered)

numeric_index = sapply(brp ,is.numeric)
numeric_data = brp[,numeric_index]
cnames = colnames(numeric_data)
cnames
for(i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "count"), data = subset(brp))+
           stat_boxplot(geom = "errorbar" , width = 0.5)+
           geom_boxplot(outlier.color="red",fill ="grey",outlier.shape = 18,
                        outlier.size=1,notch=FALSE)+
           theme(legend.position="bottom")+
           labs(y=cnames[i],x= "count")+
           ggtitle(paste("Box plot for", cnames[i])))        
}

#Plotting plots together
#gridExtra is library name

gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)

options(warn=0)

# brp$casual = as.integer(brp$casual)
# brp$registered = as.integer(brp$registered)
# head(brp)

brp = subset(brp, select = -c(dteday))
temp_brp = subset(brp, select = -c(holiday, casual, registered, count))


numeric_index = sapply(temp_brp ,is.numeric)
numeric_data = temp_brp[,numeric_index]
cnames = colnames(numeric_data)
 
for(i in cnames)
{
#   print(i)
  val = temp_brp[,i][brp[,i] %in% boxplot.stats(temp_brp[,i])$out]
  
#   Put NA in the place of outliers and Impute.    
#   brp[,i][brp[,i] %in% val] = NA    
  
#   Remove the outliers.
  temp_brp = temp_brp[which(!temp_brp[,i] %in% val),]
  
} 

head(temp_brp, 5)


brp <- brp[brp$instant %in% temp_brp$instant,]

# brp = knnImputation(brp, k = 3)
head(brp, 5)
sum(is.na(brp))


str(brp)

# Histograms to check how the data is spreaded.

cols=c('instant','season','yr','month','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual',
       'registered','count')

for(i in 1:(length(cols)))
{
    hist(brp[,i], xlab = cols[i],main = cols[i])
}    


# head(brp)
# str(brp)


###Correlation plot
#Extreme Blue:highly positively correlated.
#Extreme Red :highly negatively correlated. 
# Check co-orelation between variables
temp_brp = subset(brp, select = -c(registered, casual, count))
numeric_index = sapply(temp_brp ,is.numeric)

numeric_data = temp_brp[,numeric_index]
cnames = colnames(numeric_data)


corrgram(temp_brp[,numeric_index],order = F,upper.panel=panel.pie,text.panel = panel.txt,
         main = "Correlation Plot")


instant=brp[nrow(brp),'instant'] + 1
input_index = as.integer(row.names(brp)[nrow(brp)])

de <- list(instant=instant ,season=season, yr=yr, month=mnth, holiday=holiday, weekday=weekday, workingday=workingday, weathersit=weathersit, 
           temp=temp, atemp=atemp, hum=hum, windspeed=windspeed, casual=as.integer(9), registered=as.integer(432), count=as.integer(441))
brp = rbind(brp,de, stringsAsFactors=FALSE)
str(brp)


brp = subset(brp, select = -c(instant, month, yr))
hola=brp




cols=c('instant','season','yr','month','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual',
       'registered','count')

# Saving the continuous variable names.
# Normalisation
# for(i in cols)
# {
#   print(i)
#   bld[,i] = (bld[,i] - min(bld[,i]))/
#                                 (max(bld[,i]) - min(bld[,i]))
# }
# Above- Converting of variables in normalized variables.

# Below-Standardisation(Apply it when, the data is following the normality pattern)
for(i in names(brp))
{
#   print(i)
  brp[,i] = (brp[,i] - mean(brp[,i]))/
                                sd(brp[,i])  
}
# brp[,'holiday'] = (brp[,'holiday'] - mean(brp[,'holiday'])) / sd(brp[,'holiday'])
str(brp)
# summary(brp)

# Dividing data into train and test using stratified sampling method.(Since, our target variable "default" is binary classification.(Yes or no)).

# library(DataCombine)

set.seed(1234)
train.index = createDataPartition(c(brp$count), p = .80, list = FALSE)
train.index = train.index[-nrow(train.index),]
# Above- list= false(we don't want repetitive observations in the training data,
# responded =0.80(80% of observations in marketing_train),the func createData... will return indexes of the observations)
train = brp[train.index,]
test = brp[-train.index,]


hola = hola[-train.index,]

# acv = actual_count_value
acv = as.data.frame(hola[,'count'], col.names = c('count'))
names(acv)[names(acv) == 'hola[, "count"]'] <- "count"
# A- Storing count variable original values and renaming its column name
tail(acv, 5)


# temp_brp = subset(brp, select=-c(count))

LR_model <- lm(count ~ season + holiday + weekday + workingday + weathersit 
                   + temp + atemp + hum + windspeed, data=brp)  

# summary(LR_model)

pred <- predict(LR_model, newdata = test)

cat(sprintf ("Mean Square Error = %0.2f", MAE(pred, test$count)))
cat(sprintf ("\nRoot mean square error = %0.2f",RMSE(pred, test$count)))
cat(sprintf ("\nR-Square value = %0.2f",R2(pred, test$count)))


pred<- as.data.frame(pred)
names(pred)[names(pred) == "1"] <- "pred"
ole = data.frame(observed=test$count, predicted=pred$pred, stringsAsFactors = FALSE)

plot(ole$observed, ole$predicted,main = "Scatter plot for Multiple Linear regression Model",
     xlab = "Observed", ylab = "predicted",
     pch = 19, frame = FALSE)
abline(lm(predicted ~ observed, data=ole), col = "blue")



dcv = pred$pred*(sd(acv$count)) + mean(acv$count) 

dcv = as.data.frame(dcv)


cat("\nActual count value\n", acv[nrow(acv),1])
cat("\nPredicted count value\n", as.integer(dcv[nrow(dcv),1]))


library(rpart)

DT_model <- rpart(count ~ season + holiday + weekday + workingday + weathersit 
                   + temp + atemp + hum + windspeed,  method="anova", data=brp )

pred <- predict(DT_model, newdata = test)

cat(sprintf ("Mean Square Error = %0.2f", MAE(pred, test$count)))
cat(sprintf ("\nRoot mean square error = %0.2f",RMSE(pred, test$count)))
cat(sprintf ("\nR-Square value = %0.2f",R2(pred, test$count)))

pred<- as.data.frame(pred)
names(pred)[names(pred) == "1"] <- "pred"
ole = data.frame(observed=test$count, predicted=pred$pred, stringsAsFactors = FALSE)

plot(ole$observed, ole$predicted,main = "Scatter plot for Decision tree Regression Model",
     xlab = "Observed", ylab = "predicted",
     pch = 19, frame = FALSE)
abline(lm(predicted ~ observed, data=ole), col = "blue")



dcv = pred$pred*(sd(acv$count)) + mean(acv$count) 

dcv = as.data.frame(dcv)


cat("\nActual count value\n", acv[nrow(acv),1])
cat("\nPredicted count value\n", as.integer(dcv[nrow(dcv),1]))


library(rpart)

RF_model <- randomForest(count ~ season + holiday + weekday + workingday + weathersit 
                   + temp + atemp + hum + windspeed, data=brp, prox=TRUE)


pred <- predict(RF_model, newdata = test)

cat(sprintf ("Mean Square Error = %0.2f", MAE(pred, test$count)))
cat(sprintf ("\nRoot mean square error = %0.2f",RMSE(pred, test$count)))
cat(sprintf ("\nR-Square value = %0.2f",R2(pred, test$count)))

pred<- as.data.frame(pred)
names(pred)[names(pred) == "1"] <- "pred"
ole = data.frame(observed=test$count, predicted=pred$pred, stringsAsFactors = FALSE)

plot(ole$observed, ole$predicted,main = "Scatter plot for Random Forest Model",
     xlab = "Observed", ylab = "predicted",
     pch = 19, frame = FALSE)
abline(lm(predicted ~ observed, data=ole), col = "blue")



dcv = pred$pred*(sd(acv$count)) + mean(acv$count) 

dcv = as.data.frame(dcv)


cat("\nActual count value\n", acv[nrow(acv),1])
cat("\nPredicted count value\n", as.integer(dcv[nrow(dcv),1]))


GB_model <- gbm(count ~ season + holiday + weekday + workingday + weathersit 
                   + temp + atemp + hum + windspeed, data=brp,distribution = "gaussian",n.trees = 10000,
                  shrinkage = 0.01, interaction.depth = 4)


pred <- predict(GB_model, newdata=test, n.trees=10000, type = "link", single.tree = FALSE)
cat(sprintf ("Mean Square Error = %0.2f", MAE(pred, test$count)))
cat(sprintf ("\nRoot mean square error = %0.2f",RMSE(pred, test$count)))
cat(sprintf ("\nR-Square value = %0.2f",R2(pred, test$count)))

pred<- as.data.frame(pred)
names(pred)[names(pred) == "1"] <- "pred"
ole = data.frame(observed=test$count, predicted=pred$pred, stringsAsFactors = FALSE)

plot(ole$observed, ole$predicted,main = "Scatter plot for Gradient Boosting Model",
     xlab = "Observed", ylab = "predicted",
     pch = 19, frame = FALSE)
abline(lm(predicted ~ observed, data=ole), col = "blue")


dcv = pred$pred*(sd(acv$count)) + mean(acv$count) 

dcv = as.data.frame(dcv)


cat("\nActual count value\n", acv[nrow(acv),1])
cat("\nPredicted count value\n", as.integer(dcv[nrow(dcv),1]))


library(xgboost) 
XGB_model <- xgboost(data = as.matrix(train), # training data as matrix
                      label = train$count,  # column of outcomes
                      nrounds = 250,       # number of trees to build
                      objective = "reg:linear", # objective
                      eta = 0.02,
                      depth = 3,
                      verbose = 0  # silent
)

pred <- predict(XGB_model, newdata=as.matrix(test))

cat(sprintf ("Mean Square Error = %0.2f", MAE(pred, test$count)))
cat(sprintf ("\nRoot mean square error = %0.2f",RMSE(pred, test$count)))
cat(sprintf ("\nR-Square value = %0.2f",R2(pred, test$count)))

pred<- as.data.frame(pred)
names(pred)[names(pred) == "1"] <- "pred"
ole = data.frame(observed=test$count, predicted=pred$pred, stringsAsFactors = FALSE)

plot(ole$observed, ole$predicted,main = "Scatter plot for Extreme Gradient Boosting Model",
     xlab = "Observed", ylab = "predicted",
     pch = 19, frame = FALSE)
abline(lm(predicted ~ observed, data=ole), col = "blue")

dcv = pred$pred*(sd(acv$count)) + mean(acv$count) 

dcv = as.data.frame(dcv)


cat("\nActual count value\n", acv[nrow(acv),1])
cat("\nPredicted count value\n", as.integer(dcv[nrow(dcv),1]))


y <- train$count
x <- as.matrix(train[,1:9])
lambdas <- 10^seq(3, -2, by = -.1)

RR_model <- cv.glmnet(x, y, alpha = 0, lambda = lambdas)

opt_lambda <- RR_model$lambda.min
# opt_lambda

pred <- predict(RR_model, s = opt_lambda, newx = as.matrix(test[,1:9]))

sst <- sum(test$count - mean(test$count^2))
sse <- sum((pred -test$count^2))
MSE=min(RR_model$cvm)

cat(sprintf ("Mean Square Error = %0.2f", MSE))
cat(sprintf ("\nRoot mean square error = %0.2f", sqrt(MSE)))
cat(sprintf ("\nR-Square value = %0.2f",1 - sse / sst))




pred<- as.data.frame(pred)
names(pred)[names(pred) == "1"] <- "pred"
ole = data.frame(observed=test$count, predicted=pred$pred, stringsAsFactors = FALSE)

plot(ole$observed, ole$predicted,main = "Scatter plot for Ridge Regression",
     xlab = "Observed", ylab = "predicted",
     pch = 19, frame = FALSE)
abline(lm(predicted ~ observed, data=ole), col = "blue")



dcv = pred$pred*(sd(acv$count)) + mean(acv$count) 

dcv = as.data.frame(dcv)


cat("\nActual count value\n", acv[nrow(acv),1])
cat("\nPredicted count value\n", as.integer(dcv[nrow(dcv),1]))


SVR_model = svm(count ~ season + holiday + weekday + workingday + weathersit 
                   + temp + atemp + hum + windspeed, data=brp, importance=TRUE)
pred = predict(SVR_model, test)


sst <- sum((y - mean(y))^2)
sse <- sum((pred - y)^2)


cat(sprintf ("Mean Square Error = %0.2f", MAE(pred, test$count)))
cat(sprintf ("\nRoot mean square error = %0.2f",RMSE(pred, test$count)))
cat(sprintf ("\nR-Square value = %0.2f", -(1 - sse / sst)))

pred <- as.data.frame(pred)

plot(test$count, pred$pred,main = "Scatter plot for SVR model",
     xlab = "Observed", ylab = "predicted",
     pch = 19, frame = FALSE)
abline(lm(pred$pred ~ test$count), col = "blue")


dcv = pred$pred*(sd(acv$count)) + mean(acv$count) 

dcv = as.data.frame(dcv)


cat("\nActual count value\n", acv[nrow(acv),1])
cat("\nPredicted count value\n", as.integer(dcv[nrow(dcv),1]))

