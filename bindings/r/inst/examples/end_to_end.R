library(innovate.R)
print(kernel_predict_model("logistic", list(inputs=list(time=c(0,1,2)), parameters=list(L=100,k=1,x0=1))))
