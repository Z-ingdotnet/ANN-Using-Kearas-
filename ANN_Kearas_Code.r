# library(bannerCommenter)
# banner("Title:", "R code","Author: Iverson ZHOU","Date: 2020-02-01","Version 1","Topic: Churn Modeling With Artificial Neural Networks (Keras)"
#         +        ,"Dataset: Telco-Customer-Churn", emph = TRUE)

###########################################################################
###########################################################################
###                                                                     ###
###                                TITLE:                               ###
###                                R CODE                               ###
###                                IVERSON ZHOU                         ###
###                          DATE: 2020-02-01                           ###
###                              VERSION 1                              ###
###    TOPIC: CHURN MODELING WITH ARTIFICIAL NEURAL NETWORKS (KERAS)    ###
###                    DATASET: TELCO-CUSTOMER-CHURN                    ###
###                                                                     ###
###########################################################################
###########################################################################

getwd()
setwd("C:/Users/22323/OneDrive/GitHub/R/Churn Modeling With Artificial Neural Networks (Keras)")


Packages <- c(
  "keras", "lime", "rsample", "recipes", "yardstick", "corrr","funModeling","tidyverse","readxl", "tidyquant"
)
#install.packages(Packages)
lapply(Packages, library, character.only = TRUE)

WA_Fn_UseC_Telco_Customer_Churn <- read_excel("C:/Users/Zing/OneDrive/GitHub/R/Churn Modeling With Artificial Neural Networks (Keras)/WA_Fn-UseC_-Telco-Customer-Churn.xlsx")


head(WA_Fn_UseC_Telco_Customer_Churn,10)

glimpse(WA_Fn_UseC_Telco_Customer_Churn)



#Observations: 7,043
#Variables: 21
#$ customerID       <chr> "7590-VHVEG", "5575-GNVDE", "3668-QPYBK", "7795-CFOCW", "9237-HQITU", "9305-CDSKC", "1452-KIOVK", "6713-OKOMC", "7892-POOKP", "6388-TABGU", "9763-GRSKD", "7469-LKBCI", "8
#$ gender           <chr> "Female", "Male", "Male", "Male", "Female", "Female", "Male", "Female", "Female", "Male", "Male", "Male", "Male", "Male", "Male", 


Telco_Customer_Churn_pruned<-na.omit(WA_Fn_UseC_Telco_Customer_Churn[,-c(1)])
#Telco_Customer_Churn_pruned <- WA_Fn_UseC_Telco_Customer_Churn %>%
#  select(-customerID) %>%
#  drop_na() %>%
#  select(Churn, everything())

str(Telco_Customer_Churn_pruned)
summary(Telco_Customer_Churn_pruned)

#some na's
df_status(Telco_Customer_Churn_pruned)

freq(Telco_Customer_Churn_pruned)
Telco_data_prof=profiling_num(Telco_Customer_Churn_pruned)

utils::View(Telco_data_prof)
Hmisc::describe(Telco_Customer_Churn_pruned)
freq(Telco_Customer_Churn_pruned$tenure)


#split into six cohorts 

a<-ggplot(WA_Fn_UseC_Telco_Customer_Churn[,c(1,6)], aes(x = tenure))
a + geom_histogram(bins = 6, color = "black", fill = "gray") +
  geom_vline(aes(xintercept = mean(tenure)), 
             linetype = "dashed", size = 0.6)


hist(Telco_Customer_Churn_pruned$TotalCharges)
hist(log(Telco_Customer_Churn_pruned$TotalCharges))

correlate(Telco_Customer_Churn_pruned)


set.seed(100)
train_test_split <- initial_split(Telco_Customer_Churn_pruned, prop = 0.8)
train_test_split

train_data <- training(train_test_split)
test_data  <- testing(train_test_split)

train_data %>%
  select(Churn, TotalCharges) %>%
  mutate(
      Churn = Churn %>% as.factor() %>% as.numeric(),
      LogTotalCharges = log(TotalCharges)
      ) %>%
  corrr::correlate() %>%  #tidy correlations
  corrr::focus(Churn) %>%
  corrr::fashion()   #format aesthetically 



#four features that are multi-category: Contract, Internet Service, Multiple Lines, and Payment Method.

# Create recipe
rec_train_data<-  recipes::recipe(Churn ~ ., data = train_data) %>%
  step_discretize(tenure, options = list(cuts = 6)) %>%   # cut the continuous variable for “tenure” & group customers into six cohorts
  step_log(TotalCharges) %>%   #log transform “TotalCharges”.
  step_dummy(all_nominal(), -all_outcomes()) %>% #one-hot encoding 
  step_center(all_predictors(), -all_outcomes()) %>% #mean-centering 
  step_scale(all_predictors(), -all_outcomes()) %>% #rescaling
  prep(data = train_data) #estimate the required parameters from a training set


#apply the above recipe to the train and test data test processes following the recipe steps to made them ML-ready datasets
train_data_ml <- recipes::bake(rec_train_data, new_data = train_data) %>% select(-Churn)
test_data_ml  <- recipes::bake(rec_train_data, new_data = test_data) %>% select(-Churn)

glimpse(train_data_ml)

#recoding Response variables
train_target_vec <- ifelse(pull(train_data, Churn) == "Yes", 1, 0)
test_target_vec  <- ifelse(pull(test_data, Churn) == "Yes", 1, 0)




#Building Multi-Layer Perceptron (MLP) ANN 
model_keras <-keras_model_sequential()   #Initializing a sequential model, composed of a linear stack of layers


model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = 16,  #number of nodes
    kernel_initializer = "uniform",  #initialising the weights
    activation         = "relu",   #rectified linear activation function
    input_shape        = ncol(train_data_ml)) %>%   #number of columns in the train_data_ml


  layer_dropout(rate = 0.1) %>%   # Dropout to control overfitting,eliminates weights below 10% from overfitting the layers

  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 


  layer_dropout(rate = 0.1) %>% # Dropout to control overfitting,eliminates weights below 10% from overfitting the layers


  # Output layer
  layer_dense(
    units              = 1,  # binary classification
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compile ANN model
  compile(
    optimizer = 'adam', #optimization algorithms adam
    loss      = 'binary_crossentropy', #binary classification problem
    metrics   = c('accuracy') #metrics used to evaluat during training and testing
  )

# Fit the keras model
# run the ANN on training data
keras_ann_train <- fit(
  object           = model_keras, 
  x                = as.matrix(train_data_ml), 
  y                = train_target_vec,
  batch_size       = 50, 
  epochs           = 35,
  validation_split = 0.30
)

print(keras_ann_train)

# Predicted Class
target_yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(test_data_ml)) %>%
                         as.vector() #Binary Classification & one column output

# Predicted Class Probability
target_yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(test_data_ml)) %>%
                         as.vector() #Binary Classification & one column output



#Inspect Model Performance

# Format test data and predictions for yardstick metrics
estimates_keras_yardstick <- tibble(  #create a simple data frame 
  truth      = as.factor(test_target_vec) %>% fct_recode(yes = "1", no = "0"),  #with truth/actual values 
  estimate   = as.factor(target_yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"), #with estimate/predicted values 
  class_prob = target_yhat_keras_prob_vec  #Predicted class probability
)

options(yardstick.event_first = FALSE) #change default setting to classify 1 as the positive class instead of 0


# Confusion Table
estimates_keras_yardstick %>% conf_mat(truth, estimate)

# Accuracy measurement
estimates_keras_yardstick %>% metrics(truth, estimate)

# Area Under the Curve (AUC) measurement
estimates_keras_yardstick %>% roc_auc(truth, class_prob)


#How often is the model correct
tibble(
  precision = estimates_keras_yardstick %>% precision(truth, estimate), #Precision
  recall    = estimates_keras_yardstick %>% recall(truth, estimate)
)

# F1-Statistic--weighted average score between the precision and recall.
estimates_keras_yardstick %>% f_meas(truth, estimate, beta = 1)



#Model Explaination With LIME

class(model_keras)

# Create model_type() function
model_type.keras.engine.sequential.Sequential <- function(x, ...) {
  "classification"
}


# Setup lime::predict_model() function for keras
predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  data.frame(Yes = pred, No = 1 - pred)
}


predict_model(x = model_keras, newdata = test_data_ml, type = 'raw') %>%
  tibble::as_tibble()


# Run lime() on training set
system.time (
explainer <- lime::lime(
  x              = as.data.frame(train_data_ml), 
  model          = model_keras, 
  bin_continuous = FALSE
))


system.time (
explanation <- lime::explain(
  x = as.data.frame(test_data_ml[1:5, ]),  #limit the dataset size to reduce running time
  explainer    = explainer, 
  n_labels     = 1,  #explaining a single class
  n_features   = 5,  #the number of top and critical features to return
  kernel_width = 0.5 # shrinking the localized evaluation to increate R square
))




plot_explanations(explanation)

plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "First 5 Cases Shown")

plot_explanations(explanation) +
    labs(title = "LIME Feature Importance Heatmap",
         subtitle = "First 5 Cases Shown")


# Performs correlation analysis
corrr_analysis <- train_data_ml %>%
  mutate(Churn = train_target_vec) %>%
  corrr::correlate() %>%
  focus(Churn) %>%
  rename(feature = rowname) %>%
  arrange(abs(Churn)) %>%
  mutate(feature = as_factor(feature)) 

corrr_analysis  %>% arrange(desc(Churn))
