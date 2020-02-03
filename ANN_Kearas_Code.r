# library(bannerCommenter)
# banner("Title:", "R code","Author: Iverson ZHOU","Date: 2020-02-01","Version 1","Topic: Churn Modeling With Artificial Neural Networks (Keras)"
#         +        ,"Dataset: Telco-Customer-Churn", emph = TRUE)

###########################################################################
###########################################################################
###                                                                     ###
###                                TITLE:                               ###
###                                R CODE                               ###
###                        AUTHOR: IVERSON ZHOU                         ###
###                          DATE: 2020-02-01                           ###
###                              VERSION 1                              ###
###    TOPIC: CHURN MODELING WITH ARTIFICIAL NEURAL NETWORKS (KERAS)    ###
###                    DATASET: TELCO-CUSTOMER-CHURN                    ###
###                                                                     ###
###########################################################################
###########################################################################

library(readxl)
WA_Fn_UseC_Telco_Customer_Churn <- read_excel("C:/Users/Zing/OneDrive/GitHub/R/Churn Modeling With Artificial Neural Networks (Keras)/WA_Fn-UseC_-Telco-Customer-Churn.xlsx")


getwd()
setwd("C:/Users/22323/OneDrive/GitHub/R/Churn Modeling With Artificial Neural Networks (Keras)")


Packages <- c(
  "keras", "lime", "tidyquant", "rsample", "recipes", "yardstick", "corrr","tidyverse","funModeling"
)
#install.packages(Packages)
lapply(Packages, library, character.only = TRUE)


head(WA_Fn_UseC_Telco_Customer_Churn,10)

glimpse(WA_Fn_UseC_Telco_Customer_Churn)



#Observations: 7,043
#Variables: 21
#$ customerID       <chr> "7590-VHVEG", "5575-GNVDE", "3668-QPYBK", "7795-CFOCW", "9237-HQITU", "9305-CDSKC", "1452-KIOVK", "6713-OKOMC", "7892-POOKP", "6388-TABGU", "9763-GRSKD", "7469-LKBCI", "8
#$ gender           <chr> "Female", "Male", "Male", "Male", "Female", "Female", "Male", "Female", "Female", "Male", "Male", "Male", "Male", "Male", "Male", 



Telco_Customer_Churn_pruned<-WA_Fn_UseC_Telco_Customer_Churn[,-c(1)]
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
  corrr::correlate() %>%
  corrr::focus(Churn) %>%
  corrr::fashion()



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
