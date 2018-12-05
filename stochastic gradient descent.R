#install.packages("dplyr")
#memory.limit(51200)
library("dplyr")
#################################################################################
# Read mnist_train.csv and mnist_test.csv separately.
train <- read.csv('mnist/mnist_train.csv', header=FALSE)
test <- read.csv('mnist/mnist_test.csv', header=FALSE)
#take transpose of dataframe to make it easier to partition (ie  row 785 becomes column 785)
train<- as.data.frame(t(train))
test<- as.data.frame(t(test))
#partition train dataframe according to classifion value (column 785 values)
train_3_5 <- filter(train, train[785]>=3)
train_0_1 <- filter(train, train[785]<3)
#Separate the true class label from all the partitions and save as vector
train_labels_3_5 <- (train_3_5[,785])
train_labels_0_1 <- (train_0_1[,785])
#remove class label 
train_3_5 <-  train_3_5[,-785]
train_0_1 <-  train_0_1[,-785]
#take transpose again to resume original dataframe orientation
train_3_5 <- as.data.frame(t(train_3_5))
train_0_1 <- as.data.frame(t(train_0_1))
#print dimensions of train partitions
dim(train_3_5)
dim(train_0_1)
#partition test dataframe according to classifion value (column 785 values)
test_3_5 <- filter(test, test[785]>=3)
test_0_1 <- filter(test, test[785]<3)
#Separate the true class label from all the partitions and save as vector 
test_labels_3_5 <- (test_3_5[,785])
test_labels_0_1 <- (test_0_1[,785])
#remove class label column
test_3_5 <-  test_3_5[,-785]
test_0_1 <-  test_0_1[,-785]
#take transpose again to resume original orientation
test_3_5 <- as.data.frame(t(test_3_5))
test_0_1 <- as.data.frame(t(test_0_1))
#print dimensions of test partitions
dim(test_3_5)
dim(test_0_1)
##########################################################################################


#HW4:
##########################################################################################
#part 1:
#add bias term to the existing matrix
addbias <- function(data) {
  c<-length(data[1,])
  r<-length(data[,1])
  biasterm<-rep(1, c)
  data[r+1,]<- biasterm
 return(data)
}
test_0_1<- addbias(test_0_1)
test_3_5<- addbias(test_3_5)
train_0_1<- addbias(train_0_1)
train_3_5<- addbias(train_3_5)
#print and check dimensions of datasets with added bias term
dim(test_3_5)
dim(test_0_1)
dim(train_3_5)
dim(train_0_1)
#############################################
#modify the label to become either 1 or -1
for (i in 1:length(test_labels_0_1)){
  if (test_labels_0_1[i]==0) {
    test_labels_0_1[i]<- -1
  }
}
for (i in 1:length(train_labels_0_1)){
  if (train_labels_0_1[i]==0) {
    train_labels_0_1[i]<- -1
  }
}
for (i in 1:length(test_labels_3_5)){
  if (test_labels_3_5[i]==3) {
    test_labels_3_5[i]<- -1
  }
  if (test_labels_3_5[i]==5) {
    test_labels_3_5[i]<- 1
  }
}
for (i in 1:length(train_labels_3_5)){
  if (train_labels_3_5[i]==3) {
    train_labels_3_5[i]<- (-1)
  }
  if (train_labels_3_5[i]==5) {
    train_labels_3_5[i]<- 1
  }
}

########################################
#create a "subset" function to ramdomly sample(or shuffle) dataset, input (data, sample%, datalabel),  return RANDOMLY sampled subset of data, store new label in global variable "newdatalabel" 
subset<- function(data,samplepercent,datalabel){
  n<-as.integer(length(data[1,])*samplepercent)
  totalcolumns<- seq.int(from = 1, to = length(data[1,])) 
  selectedcolumns<- sample(totalcolumns, size = n)
  newlabel<-NULL
  for (i in 1:n){
  newlabel[i]<-datalabel[selectedcolumns[i]] 
  }
  assign("newdatalabel", newlabel, envir = .GlobalEnv)  ##save new label to global variable newdatalabel##
  subsetdata=NULL
  for (i in (1:n)) {
    subsetdata<- cbind(subsetdata, data[,selectedcolumns[i]])
  }
  return(subsetdata)
}

##################################
#do SGD to the training set
samplepercent<- 0.1 #choose data sample percent, randomly sample say 10% of data

train<- function(data,labels,alpha){
  #randomly shuffle and sample a subset of the training set , shuffled new label is in globa variable newdatalabel
  data<-subset(data,samplepercent,labels)
  labels<-newdatalabel  # overide labels with shuffled new labels
   #initiate theta with random values
  d<- length(data[,1])
  theta<-NULL
  for (i in 1:d){
    theta[i]=runif(1)
  }
 #rename newlabel as y for easy reading
  y<-labels
  gradient<- (theta+2)
  looptime<- 0
  threshold<- 100
    while( threshold > 20 ) {  #repeat SGD until converge i.e. gradient dot gradient  become smaller than threshold
      threshold<- 0
      for (i in (1:length(data[1,]) )) {
      thetax<- theta%*%data[,i]
      numerator<- as.vector(exp(y[i]*thetax)) *y[i]*data[,i]
      denominator<- as.numeric(1+exp(y[i]*thetax)) 
      gradient<- as.vector(numerator/denominator)
      theta<- theta-(alpha*gradient) 
      #below is used for debugging and visualizing convergence status
      #    if ((i%%100)==0){
      #      print(c("operation on [i]= ", i))
      #      print(c("current accuracy", accuracy ( predict(theta, data),y)))
      #    }
      threshold<- threshold+ ((gradient%*%gradient)/length(data[1,])) # threshold is taken as the average gradient%*%gradient of the epoch
      #print(c("grad grad=", (gradient%*%gradient)))
      
      }
      #print(c("threshold=", threshold))
    alpha<-alpha*0.99 # reduce alpha after each while loop
    looptime<- looptime+1
    data<-subset(data,1,y) #shuffle dataset after each while loop
    y<-newdatalabel
      #print(c("looptime=",looptime))
    }
  assign("looptime", looptime, envir = .GlobalEnv) #return total number of while loops via looptime
  return (theta)
}
predict<- function(theta, data){
  labels<-NULL
  for  (i in (1:length(data[1,]) )){
    if ((theta %*% data[,i]) < 0){
      labels[i]<- 1
    }
    else  labels[i]<- -1
  }
  return(labels)
}
###################################
## for 0/1 set, visualize 2 correct and 2 wrong predictions 
truelabels<- train_labels_0_1
trainset<- train_0_1
theta_train <-train(trainset,truelabels,alpha=0.1)
predict_train<- predict(theta_train, trainset)

correct<-NULL
wrong<- NULL
  for (i in 1:length(truelabels)){
    if (truelabels[i]==predict_train[i])
      correct<-c(correct,i)
    else
      wrong<- c(wrong,i)
  }
#print 2 correct 0/1 images
image<- matrix(as.vector(trainset[1:784,correct[1]]),nrow =28,ncol =28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(image), col=gray(0:255/255) , main=c('train 0/1 image true classifier =',truelabels[correct[1]], 'predicted classifier =', predict_train[ correct[1]   ]))

image<- matrix(as.vector(trainset[1:784,correct[2]]),nrow =28,ncol =28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(image), col=gray(0:255/255) , main=c('train 0/1 image true classifier =',truelabels[correct[2]], 'predicted classifier =', predict_train[ correct[2]   ]))

#print 2 wrong 0/1 images
image<- matrix(as.vector(trainset[1:784,wrong[1]]),nrow =28,ncol =28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(image), col=gray(0:255/255) , main=c('image true classifier =',truelabels[wrong[1]], 'predicted classifier =', predict_train[ wrong[1]   ]))

image<- matrix(as.vector(trainset[1:784,wrong[2]]),nrow =28,ncol =28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(image), col=gray(0:255/255) , main=c('image true classifier =',truelabels[wrong[2]], 'predicted classifier =', predict_train[ wrong[2]   ]))

###################################
## for 3/5 set,  visualize 2 correct and 2 wrong predictions
truelabels<- train_labels_3_5
trainset<- train_3_5
theta_train <-train(trainset,truelabels,alpha=0.1)
predict_train<- predict(theta_train, trainset)


correct<-NULL
wrong<- NULL
for (i in 1:length(truelabels)){
  if (truelabels[i]==predict_train[i])
    correct<-c(correct,i)
  else
    wrong<- c(wrong,i)
}
#print 2 correct 0/1 images
image<- matrix(as.vector(trainset[1:784,correct[1]]),nrow =28,ncol =28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(image), col=gray(0:255/255) , main=c('train 3/5 image true classifier =',truelabels[correct[1]], 'predicted classifier =', predict_train[ correct[1]   ]))

image<- matrix(as.vector(trainset[1:784,correct[2]]),nrow =28,ncol =28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(image), col=gray(0:255/255) , main=c('train 3/5 image true classifier =',truelabels[correct[2]], 'predicted classifier =', predict_train[ correct[2]   ]))

#print 2 wrong 0/1 images
image<- matrix(as.vector(trainset[1:784,wrong[1]]),nrow =28,ncol =28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(image), col=gray(0:255/255) , main=c('train 3/5 image true classifier =',truelabels[wrong[1]], 'predicted classifier =', predict_train[ wrong[1]   ]))

image<- matrix(as.vector(trainset[1:784,wrong[2]]),nrow =28,ncol =28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(image), col=gray(0:255/255) , main=c('train 3/5 image true classifier =',truelabels[wrong[2]], 'predicted classifier =', predict_train[ wrong[2]   ]))
##################################
# debug and test
#thetatrain01 <-train(train_0_1,train_labels_0_1,alpha=0.1)
#predicttest01<- predict(theta=thetatrain01, data=test_0_1)
#acctest01<- accuracy(labels=test_labels_0_1, labels_pred=predicttest01)
#thetatrain35 <-train(train_3_5,train_labels_3_5,alpha=0.1)
#predicttest35<- predict(theta=thetatrain35, data=test_3_5)
#acctest35<- accuracy(labels=test_labels_3_5, labels_pred=predicttest35)
##########################################################################################
#part 2 Modelling:
accuracy<- function(labels, labels_pred){
   correct<-0
   for (i in 1:length(labels)){
     if (labels[i]==labels_pred[i])
     correct<-correct+1
   }
   acc<- correct/length(labels)
   return(acc)
 }
model<- function(train_data, train_labels, test_data, test_labels, alpha)  {
   theta<- train(train_data,train_labels,alpha)
   predicted_train_labels<- predict (theta, train_data)
   predicted_test_labels<- predict (theta, test_data)
   test_acc<- accuracy(test_labels, predicted_test_labels)
   train_acc<- accuracy(train_labels, predicted_train_labels) 
   outputlist <- list("theta" <- theta, "train_acc" <- train_acc, "test_acc"<- test_acc)
   return(outputlist)
}

####################################
#Train 2 models, one on the 0/1 set and another on the 3/5 set, and compute their training and test accuracies.
 #train and predict 0/1 set with alpha=0.01
samplepercent<- 0.1 #choose data sample percent, randomly sample say 10% of data

predict01<- model(train_data=train_0_1,train_labels=train_labels_0_1,test_data=test_0_1,test_labels=test_labels_0_1,alpha=0.1)
 train_acc01<-unlist(predict01[2])
 print (train_acc01) #print training set prediction accuracy
 test_acc01<-unlist(predict01[3])
 print (test_acc01) #print testing set prediction accuracy
 
 predict35<- model(train_data=train_3_5,train_labels=train_labels_3_5,test_data=test_3_5,test_labels=test_labels_3_5,alpha=0.1)
 train_acc35<-unlist(predict35[2])
 print (train_acc35) #print training set prediction accuracy
 test_acc35<-unlist(predict35[3])
 print (test_acc35) #print testing set prediction accuracy
 
##########################
#Repeat the above step 10 times, with varying learning rates. Plot the training and test accuracies against learning rate for 0/1 and 3/5.
#plot for 0/1 set
 alphavec<- NULL
 trainvec<- NULL
 testvec<- NULL
 for (i in 1:10){
   alpha<-(0.001*(2^i))    #take various alpha values 
   for (j in 1:5){  # each alpha repeat 5 times
     predict01<- model(train_data=train_0_1,train_labels=train_labels_0_1,test_data=test_0_1,test_labels=test_labels_0_1,alpha)
     train_acc01<-unlist(predict01[2])
     print (c("0/1 set  alpha = ", alpha, "train_accu= ", train_acc01) )#print training set prediction accuracy
     test_acc01<-unlist(predict01[3])
     print (c("0/1 set  alpha = ", alpha, "test_accu= ", test_acc01) ) #print testing set prediction accuracy
     alphavec<- c(alphavec,alpha)
     trainvec<- c(trainvec, train_acc01)
     testvec<- c(testvec, test_acc01)
     }
 }
  accu.data<- as.data.frame( cbind(alphavec,trainvec,testvec))
  ggplot(accu.data, aes(log2(alphavec),y=trainvec)) + geom_point() +  geom_smooth() +ggtitle("0/1 train accuracy VS learning rate alpha",subtitle ="repeated 5 times for each alpha") + ylab("accuracy")+xlab("log2(alpha)")
  ggplot(accu.data, aes(log2(alphavec),y=testvec)) + geom_point() +  geom_smooth()  + ggtitle("0/1 test accuracy VS learning rate alpha",subtitle ="repeated 5 times for each alpha") + ylab("accuracy")+xlab("log2(alpha)")
###################################
#plot for 3/5 set
  alphavec<- NULL
  trainvec<- NULL
  testvec<- NULL
  for (i in 1:10){
    alpha<-(0.001*(2^i))    #take various alpha values 
    for (j in 1:5){
      predict35<- model(train_data=train_3_5,train_labels=train_labels_3_5,test_data=test_3_5,test_labels=test_labels_3_5,alpha)
      train_acc35<-unlist(predict35[2])
      print (c("3/5 set  alpha = ", alpha, "train_accu= ", train_acc35) )#print training set prediction accuracy
      test_acc35<-unlist(predict35[3])
      print (c("3/5 set  alpha = ", alpha, "test_accu= ", test_acc35) ) #print testing set prediction accuracy
      alphavec<- c(alphavec,alpha)
      trainvec<- c(trainvec, train_acc35)
      testvec<- c(testvec, test_acc35)
    }
  }
  accu.data<- as.data.frame( cbind(alphavec,trainvec,testvec))
  ggplot(accu.data, aes(log2(alphavec),y=trainvec)) + geom_point() +  geom_smooth() +ggtitle("3/5 train accuracy VS learning rate alpha",subtitle ="repeated 5 times for each alpha") + ylab("accuracy")+xlab("log2(alpha)")
  ggplot(accu.data, aes(log2(alphavec),y=testvec)) + geom_point() +  geom_smooth()  + ggtitle("3/5 test accuracy VS learning rate alpha",subtitle ="repeated 5 times for each alpha") + ylab("accuracy")+xlab("log2(alpha)")
  
  
##########################################################################################
#part 3 Modelling:
  #Create two plots, one showing the learning curves (training and test) for 0/1, and another for 3/5. Comment on the trends of accuracy values you observe for each set.
  #based on above section, choose fasted alpha=0.2

########################################
  #for 3/5 set:
  alpha<- 0.2
  samplepercentvec<- NULL
  trainvec<- NULL
  testvec<- NULL
  for (i in (1:5)){ # used 10%, 20%, 30% ... 50% due to time constraint
    samplepercent<- i*0.1 #choose sample percentage
    for (j in 1:5){
      predict35<- model(train_data=train_3_5,train_labels=train_labels_3_5,test_data=test_3_5,test_labels=test_labels_3_5,alpha)
      train_acc35<-unlist(predict35[2])
      print (c("3/5 set  samplepercent = ", samplepercent, "train_accu= ", train_acc35) )#print training set prediction accuracy
      test_acc35<-unlist(predict35[3])
      print (c("3/5 set  samplepercent = ", samplepercent, "test_accu= ", test_acc35) ) #print testing set prediction accuracy
      samplepercentvec<- c(samplepercentvec,samplepercent)
      trainvec<- c(trainvec, train_acc35)
      testvec<- c(testvec, test_acc35)
    }
  }

  paccu.data<- as.data.frame( cbind(samplepercentvec,trainvec,testvec))
  ggplot(paccu.data, aes(samplepercentvec,y=trainvec)) + geom_point() +  geom_smooth() +ggtitle("3/5 train accuracy VS sample percentage",subtitle ="repeated 5 times for each sample %") + ylab("accuracy")+xlab("sample %")
  ggplot(paccu.data, aes(samplepercentvec,y=testvec)) + geom_point() +  geom_smooth()  + ggtitle("3/5 test accuracy VS sample percentage",subtitle ="repeated 5 times for each sample %") + ylab("accuracy")+xlab("sample %")
  
  

###################################
  #for 0/1 set:
  alpha<- 0.2
  samplepercentvec<- NULL
  trainvec<- NULL
  testvec<- NULL
  for (i in (1:5)){ 
    #NOTE# only used 10%, 20% ... 50% due to time constraint
    #change "for loop" to (i in (1:10)) wil create result from 10% to 100%
    samplepercent<- i*0.1 #choose sample percentage
    for (j in 1:5){
      predict01<- model(train_data=train_0_1,train_labels=train_labels_0_1,test_data=test_0_1,test_labels=test_labels_0_1,alpha)
      train_acc01<-unlist(predict01[2])
      print (c("0/1 set  samplepercent = ", samplepercent, "train_accu= ", train_acc01) )#print training set prediction accuracy
      test_acc01<-unlist(predict01[3])
      print (c("0/1 set  samplepercent = ", samplepercent, "test_accu= ", test_acc01) ) #print testing set prediction accuracy
      samplepercentvec<- c(samplepercentvec,samplepercent)
      trainvec<- c(trainvec, train_acc01)
      testvec<- c(testvec, test_acc01)
    }
  }
  
  paccu.data<- as.data.frame( cbind(samplepercentvec,trainvec,testvec))
  ggplot(paccu.data, aes(samplepercentvec,y=trainvec)) + geom_point() +  geom_smooth() +ggtitle("0/1 train accuracy VS sample percentage",subtitle ="repeated 5 times for each sample %") + ylab("accuracy")+xlab("sample %   (0.5 means 50%) ")
  ggplot(paccu.data, aes(samplepercentvec,y=testvec)) + geom_point() +  geom_smooth()  + ggtitle("0/1 test accuracy VS sample percentage",subtitle ="repeated 5 times for each sample %") + ylab("accuracy")+xlab("sample %   (0.5 means 50%) ")
  
  
 
 
 
 
 
 
 
 
 
 
 
 
