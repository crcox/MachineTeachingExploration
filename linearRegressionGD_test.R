X <- matrix(rnorm(2000), ncol=10)
Xmean <- apply(X,2,mean)
Xsd <- apply(X,2,sd)
for ( i in 1:10 ) {
    X[,i] <- (X[,i]-Xmean[i]) / Xsd[i]
}
theta_0 <- matrix(rnorm(10),nrow=10)
y <- X%*%theta_0

true_theta = c(0,theta_0)

m0 <- linearRegressionGD(y,X,true_theta=true_theta,eta=0.3)
m1 <- linearRegressionSGD(y,X,true_theta=true_theta,eta=0.3,method='SGD')
m2 <- linearRegressionSGD(y,X,true_theta=true_theta,eta=0.3,method='candide')
m3 <- linearRegressionSGD(y,X,true_theta=true_theta,eta=0.3,method='smith')
#m4 <- linearRegressionSGD_Smith(y,X,theta_0,omniscient=FALSE)

gd <- m0$l2loss
gd$iter <- gd$iter/200
d <- rbind(gd,m1$l2loss,m2$l2loss,m3$l2loss)

library('ggplot2')

ggplot(d, aes(x=iter, y=loss, color=method)) + geom_line(size=1) + ggtitle('SSE on whole training set after each update')
ggplot(d, aes(x=iter, y=grad, color=method)) + geom_line() + ggplot2::coord_cartesian(xlim = c(0,1000), ylim = c(0,5))

ggplot(reshape2::melt(d,measure.vars=c('loss','grad','model_err'),variable.name='metric'), aes(x=iter, y=value, color=method)) +
    geom_line(size=1) +
    ggplot2::coord_cartesian(xlim = c(0,1000), ylim = c(0,5)) +
    facet_wrap('metric') +
    theme_bw(base_size = 16) +
    ggtitle('Machine Teaching Linear Regression Simulation', subtitle = '200 examples, 10 variables, eta=0.3')+
    xlab('Number of training examples')
ggsave('machine_teaching_curves.png', dpi=150, width=8, height=6)
