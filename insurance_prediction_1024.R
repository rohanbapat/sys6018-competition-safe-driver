library(tidyverse)
library(e1071)
library(randomForest)

train_master <- read.csv('train.csv')
test_master <- read.csv('test.csv')

# Converting categorical variables 

for(colnm in colnames(train_master)){
  if((substr(colnm, nchar(colnm)-2, nchar(colnm))=="cat")||(substr(colnm, nchar(colnm)-2, nchar(colnm))=="bin")){

    train_master[colnm] <- as.factor(train_master[[colnm]])
  }
}

# Get correlation matrix
train_master_demo1 <- sapply(train_master, as.numeric)
cor_train_master_demo1 <- cor(train_master_demo1)
cor_train_master_demo1 <- as.data.frame(cor_train_master_demo1)
cor_train_master_demo1$corr_column <- rownames(cor_train_master_demo1)
cor_train_master_demo2 <- cor_train_master_demo1 %>% gather(varname, corr, id:ps_calc_20_bin)

# Check correlated variables
cor_train_master_demo2[(cor_train_master_demo2$corr>0.75)&(cor_train_master_demo2$corr<1),]

#  ps_ind_14 and ps_ind_12_bin are correlated. Remove one variable
train_master <- subset(train_master, select = -c(ps_ind_14))

# Create function to merge one-hot encoded variables
MultChoiceCondense<-function(vars,indata){
  tempvar<-matrix(NaN,ncol=1,nrow=length(indata[,1]))
  dat<-indata[,vars]
  for (i in 1:length(vars)){
    for (j in 1:length(indata[,1])){
      if (dat[j,i]==1) tempvar[j]=i
    }
  }
  return(tempvar)
}

# Variables 'ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin' are
# one-hot encoded from the same variable. Merging them into a single variable
train_master$ps_ind_custom <- MultChoiceCondense(c('ps_ind_06_bin' , 'ps_ind_07_bin'  ,'ps_ind_08_bin',  'ps_ind_09_bin'  ,'ps_ind_10_bin' ),train_master)

# Remove the 4 merged variables
train_master <- subset(train_master, select = -c(ps_ind_06_bin,ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin))

# Check variables for skewness
train_master_numeric <- train_master[sapply(train_master,is.numeric)]
sapply(train_master_numeric, skewness)

# Transform skewed variables
train_master$ps_reg_02_log <- log(train_master$ps_reg_02+1)
train_master$ps_car_15_log <- log(train_master$ps_car_15+1)

# Other data cleaning 
# remove id variable
train_master <- subset(train_master, select = -c(id))

# convert response to categorical
train_master$target <- as.factor(train_master$target)

# Drop ps_car_11_cat
train_master <- subset(train_master, select = -c(ps_car_11_cat))


# Do variable selection for logistic regression ---------------------------

# create null and full models for stepwise variable selection

# null model
null=glm(target ~ 1, data = train_master, family=binomial(link="logit"))
null

# create formula for full term set:
#
col_names <- c("ps_car_13","ps_car_06_cat","ps_reg_03","ps_car_14","ps_calc_10","ps_calc_14","ps_ind_15","ps_car_01_cat","ps_ind_03","ps_calc_11") 
# combine into formula
terms_init <-  paste(col_names, collapse="+")
long_formula <- as.formula(sprintf("target ~ (%s)^2", terms_init))

# full model:
full=glm(formula = long_formula, data = training_set, family=binomial(link="logit"))
full


# fit step-wise
my_step <- step(null, scope=list(lower=null, upper=full), direction="both")
# my_step <- step(full, direction="backward")
summary(my_step)
anova(my_step)

# make predictions on validation set
preds_init <- predict(my_step, newdata = final_validation_set)
# use predictions of change to make actual prediction values
preds_final <- final_validation_set$PRICE*(1 + preds_init)

# calculate MSE
n <- length(final_validation_set$DATE)
MSE_lm <- sum((preds_final - final_validation_set$NextDayPrice)^2)/n
MSE_lm
# 1.637032

# Build random forest
rf1 <- randomForest(target~., data = train_master, importance = TRUE, ntree = 100)
