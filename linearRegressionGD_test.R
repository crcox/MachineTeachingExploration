X <- matrix(rnorm(200), ncol=2)
theta_0 <- matrix(rnorm(2),nrow=2)
y <- X%*%theta_0

m0 <- linearRegressionGD(y,X)
m1 <- linearRegressionSGD(y,X)
m2 <- linearRegressionSGD_Candide(y,X)
m3 <- linearRegressionSGD_Smith(y,X,theta_0)
m4 <- linearRegressionSGD_Smith(y,X,theta_0,omniscient=FALSE)
