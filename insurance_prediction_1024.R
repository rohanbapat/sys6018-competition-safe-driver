library(tidyverse)
library(e1071)
library(randomForest)
source("gini_functions.R")

# Read In Data ------------------------------------------------------------

train_master <- read.csv('train.csv') # read in training
test_master <- read.csv('test.csv') # read in test
test_master$target <- 0 # create placeholder column for test target (just to make using the dataframe more seamless in later functions)
combined_df <- rbind(train_master,test_master) # combine training and test


# Clean Data --------------------------------------------------------------

# Convert categorical variables to factor 
for(colnm in colnames(combined_df)){
  if((substr(colnm, nchar(colnm)-2, nchar(colnm))=="cat")||(substr(colnm, nchar(colnm)-2, nchar(colnm))=="bin")){

    combined_df[colnm] <- as.factor(combined_df[[colnm]])
  }
}

# First remove variables with lot of missing values
sapply(combined_df, function(x) sum(x==-1))

# Variables ps_car_03_cat and ps_car_05_cat have lot of missing observations (over 50%)
# Drop both these columns
combined_df <- subset(combined_df, select = -c(ps_car_03_cat, ps_car_05_cat))

# Custom function to get mode of vector
Mode <- function(x) {
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

# 
impute_missing <- function(x){
  if(sum(x==-1)==0){
    return(x)
  }
  else{
    non_msg <- x!=-1
    if(class(x) == "factor"){
      imputed <- Mode(x[non_msg])
    }
    else{
      imputed <- median(x[non_msg])
    }
    x[!non_msg] <- imputed
    return(x)
  }
}

combined_df <- as.data.frame(lapply(combined_df, impute_missing))

# Get correlation matrix
combined_df_demo1 <- sapply(combined_df, as.numeric)
cor_combined_df_demo1 <- cor(combined_df_demo1)
cor_combined_df_demo1 <- as.data.frame(cor_combined_df_demo1)
cor_combined_df_demo1$corr_column <- rownames(cor_combined_df_demo1)
cor_combined_df_demo2 <- cor_combined_df_demo1 %>% gather(varname, corr, id:ps_calc_20_bin)

# Check orrelated variables
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
combined_df$ps_reg_02 <- log(combined_df$ps_reg_02+1)
combined_df$ps_car_15 <- log(combined_df$ps_car_15+1)

# Other data cleaning 
#
# convert response to factor
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

total_sample_id_vec <- c(id_rand_smpl0,id_vec1) # combine all ids into single vector for downsampling

downsampled_full <- combined_df[(combined_df$id %in% total_sample_id_vec),] # downsample training
set.seed(567)
ds_train_row_nums <- sample(1:nrow(downsampled_full),as.integer(nrow(downsampled_full)/2)) # create vector of row numbers to sample
downsampled_train <- downsampled_full[ds_train_row_nums,]
downsampled_test <- downsampled_full[-ds_train_row_nums,]


# Select Variables And Create Model Formula -------------------------------

## NOTE: The following variables were selected by fitting a random forest and selecting out the 10 most important variables according to 'importance' function

# create formula for full term set:
#
col_names <- c("ps_car_13","ps_car_06_cat","ps_reg_03","ps_car_14","ps_calc_10","ps_calc_14","ps_ind_15","ps_car_01_cat","ps_ind_03","ps_calc_11") 
# combine into formula
terms_init <-  paste(col_names, collapse="+")
long_formula <- as.formula(sprintf("target ~ (%s)^2", terms_init)) # create formula using chosen variables AND all pairwise interaction terms

# Fit lasso logistic regression  ----------------------------------------

library(glmnet)
# Cross-validate lasso regression to find optimal lambda value
y <- downsampled_full$target # create vector of target 
x <- model.matrix(long_formula, downsampled_full)[,-1] # create model matrix
set.seed(123) # set seed
cv.logistic.lasso <- cv.glmnet(x, y, alpha=1, family="binomial") # cross-validate logistic lasso
plot(cv.logistic.lasso) # plot lambda vs error
cv.logistic.lasso$lambda.min # extract minimum lambda: 0.00368678

# Make predictions on test set
test_ids <- test_master$id # extract ids to identify test set rows
x_test <- model.matrix(long_formula, combined_df[(combined_df$id %in% test_ids),])[,-1] # create model matrix
preds <- predict(cv.logistic.lasso$glmnet.fit, s = 0.00368678, newx = x_test, type = "response") # make predictions using lambda value chosen with cross-validation

# create tibble for output
test_ids <- as.integer(test_ids)
output <- as_tibble(cbind(test_ids, preds)) # create tibble for export
colnames(output) <- c("id","target") # rename tibble cols

# write final output to file
write_csv(output, path = "khg3je_logistic.csv")



# Optimize Random Forest Hyperparameters ----------------------------------

mtry_vals <- c(3,4,5,6,7,8) # create vector of variable subset numbers
ntrees <- c(2000,5000,10000) # set number of trees to try in hyperparameter grid. NOTE: We tried many other, smaller numbers of trees, saw an upward trend, and then tried larger numbers of trees.  This set is only the last set we tried.
hyperparam_grid <- expand.grid(mtry_vals,ntrees)
colnames(hyperparam_grid)[1:2] <- c("mtry","ntree")

rf_func <- function(mtryX,ntreeX){
  set.seed(1) 
  my_rf=randomForest(formula = long_formula, data = downsampled_train, mtry = mtryX, ntree= ntreeX) # fit random forest to downsampled train
  rf.preds = predict(my_rf,newdata=downsampled_test,type = "prob")[,2] # make predictions on test
  normalized.gini.index(as.numeric(downsampled_test$target), rf.preds)
  
}

params_and_mse <- hyperparam_grid
params_and_mse$gini_score <- mapply(rf_func,hyperparam_grid[[1]],hyperparam_grid[[2]]) # apply function to hyperparam grid and add as new column

library(tidyverse)
params_and_mse %>%  # plot all
  ggplot(aes(x=ntree,y=gini_score,color=as.factor(mtry))) +
  geom_point() +
  geom_line()


# Fit Random Forest Using Chosen Hyperparameters --------------------------

# Fit forest
rf1 <- randomForest(long_formula, data = downsampled_full, mtry = 3, importance = TRUE, ntree = 5000)

# Make predictions on test set and export to csv
preds.rf = predict(rf1,newdata=combined_df[(combined_df$id %in% test_ids),], type = "prob")

# create tibble for output
test_ids <- as.integer(test_ids)
output <- as_tibble(cbind(test_ids, preds.rf))
colnames(output) <- c("id","target") # rename cols

# write final output to file
write_csv(output, path = "khg3je_rf.csv") # Note that this must be manually converted out of scientific notation.  Use excel.

