import torch
import numpy as np
import math
torch.device('cpu', 0)

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    #m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def sfa(X,k=3,loops=100,tolerance=0.0001):

    #find the hieght and width of the tensor
    (length,width) = X.size()
    #print(length)
    #print(width)

    #create a matrix (same size as X) that has the mean of each column as a repeat down each column
    mean_Matrix = torch.ones([length,width], dtype=torch.float32)*X.mean(0)
    X_new = X - mean_Matrix
    #print(X_new)

    #multipy data matrix by transpose and divide by the number of entries
    XXT = torch.mm(torch.transpose(X_new,0,1),X_new)/length
    #get the diagonal values of the new matrix
    #print(XXT)
    diagXX = torch.diag(XXT,0)
    #print(diagXX)

    #covariance matrix with column variances (because covaraince of something with itself is just the variance) along 
    #the diagnoal this matrix is square (should be 5x5 if there are only 5 components)
    #(uses the N-1 as the noramlization factor)
    covX = cov(X_new,rowvar=False)
    #print(covX)

    #get the determinant of the covaraint matrix
    cov_det = torch.det(covX)
    #print(cov_det)

    #create scaling factor for the factor loadings matrix
    scaling_Factor = cov_det**(1/width)

    #makes random matrix of size (# of components)x(number of factors)
    #then scales each of these by sqrt(scale/number of factors) to get the loading factors

    #L=torch.randn(width, k, dtype=torch.float32)*(scaling_Factor/k)**(1/2.0)
    FL = torch.tensor([[1.1290  ,  0.7853  ,  0.6562],
                        [0.0528  ,  0.5253  , -0.0142],
                        [0.4302  , -0.3982  ,  0.4852],
                        [0.5423  ,  0.1682  ,  0.0316],
                      [-0.4978  , -0.4876   , 0.1429]  ]);


    #print(FL)

    #gets the variances of each of the columns of cX (just gets the diagonals of cX)
    #put them into the starting value for the diagnola uniqueness matrix
    diag_unique = torch.diag(covX)
    #print(diag_unique)

    #create a KxK identity matrix
    I = torch.eye(k)
    #print(I)

    #create variables to store the log likelihood curve
    lik=0; LL=[];

    #create a constant to be used in later in the log likelihood curve calculation
    const=-width/2*math.log(2*math.pi)
    #print(const)

    for i in range(1,loops+1):
        ####E Step####
        #need to compute E(z|xi)    and  E(zz'|xi) for each data point xi,
        #given lamda and psi

        diag_unique_Inv = torch.inverse(torch.diag(diag_unique))
        #diagU_Inv = torch.diag(torch.reciprocal(diagU)) --> would also produce the same result
        #print(diagU_Inv)
        diag_unique_Inv_x_FL = torch.mm(diag_unique_Inv,FL)
        #print(diag_unique_Inv_x_FL)
        beta_Input = diag_unique_Inv - torch.mm(torch.mm(diag_unique_Inv_x_FL,
                             torch.inverse(torch.addmm(I,torch.transpose(FL,0,1),diag_unique_Inv_x_FL))),
                             torch.transpose(diag_unique_Inv_x_FL,0,1))
        #print(beta_Input)

        beta = torch.mm(torch.transpose(FL,0,1),beta_Input)
        #print(beta)

        XXT_x_betaT = torch.mm(XXT,torch.transpose(beta,0,1))
        #print(XXT_x_betaT)

        Ezz = I-torch.mm(beta,FL)+torch.mm(beta,XXT_x_betaT)
        #print(torch.addmm(torch.mm(beta,FL),beta,XXT_x_betaT))
        #print(Ezz)

        #calculations for the log likelihood function
        oldlik = lik

        last_arg = torch.sum(torch.diag(torch.mm(beta_Input,XXT)));
        #print(last_arg)


        lik = (length*const+length*math.log(math.sqrt(torch.det(beta_Input)))-0.5*length*torch.sum(torch.diag(torch.mm(beta_Input,XXT))))
        #print(lik)
        lik = round(lik.item(),4)


        print('cycle' + str(i) + ' lik='+str(lik))

        #store the likelihood
        LL.append(lik)
        #print(LL)

        #M step-where we re-estimate the parameters
        FL = torch.mm(XXT_x_betaT,torch.inverse(Ezz))
        diag_unique = diagXX-torch.diag(torch.mm(FL,torch.transpose(XXT_x_betaT,0,1)))

        #print(FL)
        #print(diag_unique)

        #set the cycle to stop when past the tolerance
        if i<=2:
            likbase=lik
        elif (lik<oldlik):
            print("Violation, likelihood getting worse!!")
        elif ((lik-likbase)<(1+tolerance)*(oldlik-likbase)or (not np.isfinite(lik))):
            break;
            
    return (FL,diag_unique,LL)

