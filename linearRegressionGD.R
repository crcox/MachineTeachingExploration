addBiasUnit <- function(X) {
    if ( ! (all(X[,1]==1) || all(X[,dim(X)[2]]==1)) ) {
        X <- as.matrix(data.frame(rep(1,dim(X)[1]),X))
        colnames(X)[1] <- '(Intercept)'
    }
    return(X)
}
gradientFun <- function(theta, y, X) {
    e <- matrix(y,ncol=1) - X %*% matrix(theta,ncol=1)
    grad <- -(2) %*% t(e) %*% X
    return(grad)
}
updateWeights <- function(theta,grad,eta) {
    theta_updated <- theta - 2*eta*matrix(grad,ncol=1)
    return(theta_updated)
}
sumOfSquaredError <- function(y,prediction) {
    sum((y - prediction)^2)
}
initOutputDataframe <- function(n,method_index)
    return(data.frame(
        example   = numeric(n+1),
        loss      = numeric(n+1),
        grad      = numeric(n+1),
        iter      = numeric(n+1),
        method    = factor(method_index,levels=1:4,c('GD','SGD','Candide','Smith')),
        model_err = numeric(n+1)
    ))
teacher.candide <- function(theta,y,X,eta) {
    N <- dim(X)[1]
    obj <- rep(0,N)
    for (j in 1:N) {
        grad <- gradientFun(theta, y[j], X[j,])# / N
        theta_tmp <- updateWeights(theta,grad,eta/N)
        obj[j] <- sqrt(sumOfSquaredError(y,X %*% theta_tmp))
    }
    return( which.min(obj) )
}
teacher.smith <- function(theta,true_theta,y,X,eta) {
    N <- dim(X)[1]
    obj <- rep(0, N)
    for (j in 1:N) {
        grad <- gradientFun(theta, y[j], X[j,])# / N
        obj[j] <- omniscient_loss(grad,theta,true_theta,eta)
    }
    return( which.min(obj) )
}
teacher.sgd <- function(N) {
    return(sample(N,1))
}
omniscient_loss <- function(grad,theta,teacher,eta) {
    difficulty <- sqrt(sum(grad^2))
    usefulness <- matrix((theta - teacher),nrow=1) %*% matrix(grad,ncol=1)
    obj <- (eta^2 * difficulty) - (2 * eta * usefulness)
    return(obj)
}
linearRegressionGD<- function(y, X, true_theta, epsilon = 0.0001, eta = 10, iters = 1000){
    METHOD_INDEX = 1
    X <- addBiasUnit(X)
    N <- dim(X)[1]
    print("Initialize parameters...")
    theta.init <- matrix(rnorm(n=dim(X)[2], mean=0, sd = 1), ncol=1) # Initialize theta
    grad.init <- gradientFun(theta.init, y, X) / N
    theta <- updateWeights(theta.init, grad.init, eta/N)
    l2loss <- initOutputDataframe(iters+1, METHOD_INDEX)
    l2loss$loss[1] <- sqrt(sumOfSquaredError(y,X %*% theta))
    for(i in 1:iters){
        grad <- gradientFun(theta, y, X) / N
        theta <- updateWeights(theta,grad,eta/N)
        l2loss$example[i+1] <- 0
        l2loss$loss[i+1] <- sqrt(sumOfSquaredError(y,X %*% theta))
        l2loss$grad[i+1] <- sqrt(sum(grad^2))
        l2loss$iter[i+1] <- i*N
        l2loss$model_err[i+1] <- sqrt(sumOfSquaredError(true_theta, theta))
        if(sqrt(sum(grad^2)) <= epsilon){
            break
        }
    }
    l2loss <- l2loss[1:(i+1),]
    print("Algorithm converged")
    print(paste("Final gradient norm is",sqrt(sum(grad^2))))
    values<-list("coef" = t(theta), "l2loss" = l2loss)
    return(values)
}

linearRegressionSGD<- function(y, X, true_theta, epsilon = 0.0001, eta = 10, iters = 1000, method=c('SGD','candide','smith')) {
    if ( tolower(method) == 'SGD' ) {
        METHOD_INDEX = 2
    } else if ( tolower(method) == 'candide' ) {
        METHOD_INDEX = 3
    } else if ( tolower(method) == 'smith' ) {
        METHOD_INDEX = 4
    } else {
        METHOD_INDEX = 2
    }
    X <- addBiasUnit(X)
    N <- dim(X)[1]
    print("Initialize parameters...")
    theta.init <- matrix(rnorm(n=dim(X)[2], mean=0, sd = 1), ncol=1) # Initialize theta
    grad.init <- gradientFun(theta.init, y, X) / N
    theta <- updateWeights(theta.init, grad.init, eta/N)
    l2loss <- initOutputDataframe((iters*N)+1, METHOD_INDEX)
    l2loss$loss[1] <- sqrt(sumOfSquaredError(y,X %*% theta))
    for(i in 1:(iters*N)) {
        if        ( METHOD_INDEX == 2 ) {
            k <- teacher.sgd(N)
        } else if ( METHOD_INDEX == 3 ) {
            k <- teacher.candide(theta,y,X,eta)
        } else if ( METHOD_INDEX == 4 ) {
            k <- teacher.smith(theta,true_theta,y,X,eta)
        }
        grad <- gradientFun(theta, y[k], X[k,])# / N
        theta <- updateWeights(theta,grad,eta/N)
        l2loss$example[i+1] <- k
        l2loss$loss[i+1] <- sqrt(sumOfSquaredError(y,X %*% theta))
        l2loss$grad[i+1] <- sqrt(sum(grad^2))
        l2loss$iter[i+1] <- i
        l2loss$model_err[i+1] <- sqrt(sumOfSquaredError(true_theta, theta))
        if(sqrt(sum(grad^2)) <= epsilon){
            break
        }
    }
    l2loss <- l2loss[1:(i+1),]
    print("Algorithm converged")
    print(paste("Final gradient norm is",sqrt(sum(grad^2))))
    values<-list("coef" = t(theta), "l2loss" = l2loss)
    return(values)
}
