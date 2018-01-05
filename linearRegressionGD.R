linearRegressionGD<- function(y, X, epsilon = 0.0001, eta = 10, iters = 1000){
    X = as.matrix(data.frame(rep(1,length(y)),X))
    N = dim(X)[1]
    print("Initialize parameters...")
    theta.init = as.matrix(rnorm(n=dim(X)[2], mean=0,sd = 1)) # Initialize theta
    theta.init = t(theta.init)
    e = t(y) - theta.init%*%t(X)
    grad.init = -(2/N)%*%(e)%*%X
    theta = theta.init - eta*(1/N)*grad.init
    l2loss = c()
    for(i in 1:iters){
        l2loss = c(l2loss,sqrt(sum((t(y) - theta%*%t(X))^2)))
        e = t(y) - theta%*%t(X)
        grad = -(2/N)%*%e%*%X
        theta = theta - eta*(2/N)*grad
        if(sqrt(sum(grad^2)) <= epsilon){
            break
        }
    }
    print("Algorithm converged")
    print(paste("Final gradient norm is",sqrt(sum(grad^2))))
    values<-list("coef" = t(theta), "l2loss" = l2loss)
    return(values)
}

linearRegressionSGD<- function(y, X, epsilon = 0.0001, eta = 10, iters = 1000){
    X = as.matrix(data.frame(rep(1,length(y)),X))
    N = dim(X)[1]
    print("Initialize parameters...")
    theta.init = as.matrix(rnorm(n=dim(X)[2], mean=0,sd = 1)) # Initialize theta
    theta.init = t(theta.init)
    e = t(y) - theta.init%*%t(X)
    grad.init = -(2/N)%*%(e)%*%X
    theta = theta.init - eta*(1/N)*grad.init
    l2loss = c()
    for(i in 1:iters){
        ix <- sample(N)
        for (j in ix) {
            e = y[j] - matrix(theta,nrow=1)%*%matrix(X[j,],ncol=1)
            grad = -(2/N)%*%e%*%X[j,]
            theta = theta - eta*(2/N)*grad
            if(sqrt(sum(grad^2)) <= epsilon){
                break
            }
        }
        l2loss = c(l2loss,sqrt(sum((t(y) - theta%*%t(X))^2)))
    }
    print("Algorithm converged")
    print(paste("Final gradient norm is",sqrt(sum(grad^2))))
    values<-list("coef" = t(theta), "l2loss" = l2loss)
    return(values)
}

linearRegressionSGD_Candide <- function(y, X, epsilon = 0.0001, eta = 10, iters = 1000){
    X = as.matrix(data.frame(rep(1,length(y)),X))
    N = dim(X)[1]
    print("Initialize parameters...")
    theta.init = as.matrix(rnorm(n=dim(X)[2], mean=0,sd = 1)) # Initialize theta
    theta.init = t(theta.init)
    e = t(y) - theta.init%*%t(X)
    grad.init = -(2/N)%*%(e)%*%X
    theta = theta.init - eta*(1/N)*grad.init
    l2loss = data.frame(example=numeric(iters*N),loss=numeric(iters*N))
    for(i in 1:(iters*N)) {
        l2loss_inner = rep(0,N)
        for (j in 1:N) {
            e = y[j] - matrix(theta,nrow=1)%*%matrix(X[j,],ncol=1)
            grad = -(2/N)%*%e%*%X[j,]
            theta_tmp = theta - eta*(2/N)*grad
            l2loss_inner[j] = sqrt(sum((t(y) - theta_tmp%*%t(X))^2))
        }
        k <- which.min(l2loss_inner)
        e = y[k] - matrix(theta,nrow=1)%*%matrix(X[k,],ncol=1)
        grad = -(2/N)%*%e%*%X[k,]
        theta = theta - eta*(2/N)*grad
        l2loss$example[i] = k
        l2loss$loss[i] = sqrt(sum((t(y) - theta%*%t(X))^2))
        if(sqrt(sum(grad^2)) <= epsilon){
            break
        }
    }
    l2loss <- l2loss[l2loss$example!=0,]
    print("Algorithm converged")
    print(paste("Final gradient norm is",sqrt(sum(grad^2))))
    values<-list("coef" = t(theta), "l2loss" = l2loss)
    return(values)
}

linearRegressionSGD_Smith <- function(y, X, epsilon = 0.0001, eta = 10, iters = 1000){
    X = as.matrix(data.frame(rep(1,length(y)),X))
    N = dim(X)[1]
    print("Initialize parameters...")
    theta.init = as.matrix(rnorm(n=dim(X)[2], mean=0,sd = 1)) # Initialize theta
    theta.init = t(theta.init)
    e = t(y) - theta.init%*%t(X)
    grad.init = -(2/N)%*%(e)%*%X
    theta = theta.init - eta*(1/N)*grad.init
    l2loss = c()
    for(i in 1:iters){
        ix <- sample(N)
        for (j in ix) {
            e = y[j] - matrix(theta,nrow=1)%*%matrix(X[j,],ncol=1)
            grad = -(2/N)%*%e%*%X[j,]
            theta = theta - eta*(2/N)*grad
            if(sqrt(sum(grad^2)) <= epsilon){
                break
            }
        }
        l2loss = c(l2loss,sqrt(sum((t(y) - theta%*%t(X))^2)))
    }
    print("Algorithm converged")
    print(paste("Final gradient norm is",sqrt(sum(grad^2))))
    values<-list("coef" = t(theta), "l2loss" = l2loss)
    return(values)
}
