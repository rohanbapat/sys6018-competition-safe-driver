library(tidyverse)
library(e1071)
library(randomForest)

train_master <- read.csv('train.csv')
test_master <- read.csv('test.csv')
test_master$target <- 0
combined_df <- rbind(train_master,test_master)

# Converting categorical variables 

for(colnm in colnames(combined_df)){
  if((substr(colnm, nchar(colnm)-2, nchar(colnm))=="cat")||(substr(colnm, nchar(colnm)-2, nchar(colnm))=="bin")){

    combined_df[colnm] <- as.factor(combined_df[[colnm]])
  }
}

# Get correlation matrix
combined_df_demo1 <- sapply(combined_df, as.numeric)
cor_combined_df_demo1 <- cor(combined_df_demo1)
cor_combined_df_demo1 <- as.data.frame(cor_combined_df_demo1)
cor_combined_df_demo1$corr_column <- rownames(cor_combined_df_demo1)
cor_combined_df_demo2 <- cor_combined_df_demo1 %>% gather(varname, corr, id:ps_calc_20_bin)

# Check correlated variables
cor_combined_df_demo2[(cor_combined_df_demo2$corr>0.75)&(cor_combined_df_demo2$corr<1),]

#  ps_ind_14 and ps_ind_12_bin are correlated. Remove one variable
combined_df <- subset(combined_df, select = -c(ps_ind_14))

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
combined_df$ps_ind_custom <- MultChoiceCondense(c('ps_ind_06_bin' , 'ps_ind_07_bin'  ,'ps_ind_08_bin',  'ps_ind_09_bin'  ,'ps_ind_10_bin' ),combined_df)

# Remove the 4 merged variables
combined_df <- subset(combined_df, select = -c(ps_ind_06_bin,ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin))

# Check variables for skewness
combined_df_numeric <- combined_df[sapply(combined_df,is.numeric)]
sapply(combined_df_numeric, skewness)

# Transform skewed variables
combined_df$ps_reg_02_log <- log(combined_df$ps_reg_02+1)
combined_df$ps_car_15_log <- log(combined_df$ps_car_15+1)

# Other data cleaning 
#
# convert response to categorical
combined_df$target <- as.factor(combined_df$target)

# Drop ps_car_11_cat
combined_df <- subset(combined_df, select = -c(ps_car_11_cat))


# Downsample Training -----------------------------------------------------
set.seed(23)
id_vec0 <- as_tibble(train_master) %>% # create vector of all id's in training set where target == 0
  filter(target == 0) %>% 
  select(id) %>% 
  pull()
id_rand_smpl0 <- sample(id_vec0,21694) # sample randomly from id's where target == 0

id_vec1 <- as_tibble(train_master) %>% # create vector of all id's in training set where target == 1
  filter(target == 1) %>% 
  select(id) %>% 
  pull()

total_sample_id_vec <- c(id_rand_smpl0,id_vec1)

downsampled_train <- combined_df[(combined_df$id %in% total_sample_id_vec),] # downsample training

# Do variable selection for logistic regression ---------------------------

# create formula for full term set:
#
col_names <- c("ps_car_13","ps_car_06_cat","ps_reg_03","ps_car_14","ps_calc_10","ps_calc_14","ps_ind_15","ps_car_01_cat","ps_ind_03","ps_calc_11") 
# combine into formula
terms_init <-  paste(col_names, collapse="+")
long_formula <- as.formula(sprintf("target ~ (%s)^2", terms_init))

library(glmnet)
y <- downsampled_train$target
x <- model.matrix(long_formula, downsampled_train)[,-1]

# Cross-validate lasso regression to find optimal lambda value
set.seed(123)
cv.logistic.lasso <- cv.glmnet(x, y, alpha=1, family="binomial") # 
plot(cv.logistic.lasso) # plot lambda vs error
cv.logistic.lasso$lambda.min # minimum lambda: 0.00368678

# Fit lasso regression on entire training
#
train_ids <- train_master$id
y_alltrain <- combined_df[(combined_df$id %in% train_ids),]$target
x_alltrain <- model.matrix(long_formula, combined_df[(combined_df$id %in% train_ids),])[,-1]
logistic.lasso <- glmnet(x_alltrain, y_alltrain, alpha = 1, family = "binomial")

# Make predictions on test set and export to csv
test_ids <- test_master$id
x_test <- model.matrix(long_formula, combined_df[(combined_df$id %in% test_ids),])[,-1]
preds <- predict(logistic.lasso, s = 0.00368678, newx = x_test, type = "response")

# create tibble for output
test_ids <- as.integer(test_ids)
output <- as_tibble(cbind(test_ids, preds))
colnames(output) <- c("id","target") # rename cols

# write final output to file
write_csv(output, path = "khg3je_logistic.csv")


# Build random forest
rf1 <- randomForest(target~., data = downsampled_train, mtry = 5, importance = TRUE, ntree = 1000)

# Make predictions on test set and export to csv
preds.rf = predict(rf1,newdata=combined_df[(combined_df$id %in% test_ids),], type = "prob")

# create tibble for output
test_ids <- as.integer(test_ids)
output <- as_tibble(cbind(test_ids, preds.rf))
colnames(output) <- c("id","target") # rename cols

# write final output to file
write_csv(output, path = "khg3je_rf.csv")

