import GPyOpt
import numpy as np


class FUR_W(GPyOpt.acquisitions.base.AcquisitionBase):

    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    analytical_gradient_prediction = False

    def __init__(
        self, model, space, optimizer, X_init, std, weight=None, cost_withGradients=None, **kwargs
    ):
        self.optimizer = optimizer
        self.X_init = X_init
        self.iter = 1
        self.std = std
        
        if weight == None:
            self.weight = [1.0, 1.0]
            
        elif isinstance(weight, float):
            self.weight = [weight, 1.0 - weight]
            
        else:
            self.weight = weight            
            
        super(FUR_W, self).__init__(model, space, optimizer)

    def _compute_acq(self, x):
        if x.shape[0] > 1:
            self.iter += 1
            self.delta = np.random.randn()
        m, s = self.model.predict(x)
        f_acqu = (
            -np.linalg.norm(x - self.X_init - self.std * self.delta / np.log(self.iter)) * self.weight[0]
            + s * self.weight[1]
        )
        return f_acqu
    
    
    def Set_Weight(weight):
        
        if isinstance(weight, float):
            self.weight = [weight, 1.0 - weight]
            
        else:
            self.weight = weight            



class FUR(GPyOpt.acquisitions.base.AcquisitionBase):

    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    analytical_gradient_prediction = False

    def __init__(
        self, model, space, optimizer, X_init, std, cost_withGradients=None, **kwargs
    ):
        self.optimizer = optimizer
        self.X_init = X_init
        self.iter = 1
        self.std = std
        super(FUR, self).__init__(model, space, optimizer)

    def _compute_acq(self, x,v=False):
        if x.shape[0] > 1:
            self.iter += 1
            self.delta = np.random.randn()
        m, s = self.model.predict(x)
        f_acqu = (
            -np.linalg.norm(x - self.X_init - self.std * self.delta / np.log(self.iter))
            + s
        )
        if v:
            print('x', x)
            print('m', m)
            print('s', s)
            print('f_acqu', f_acqu)
        return f_acqu


class UR(GPyOpt.acquisitions.base.AcquisitionBase):

    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, cost_withGradients=None, **kwargs):
        self.optimizer = optimizer
        super(UR, self).__init__(model, space, optimizer)

    def _compute_acq(self, x):
        m, s = self.model.predict(x)
        f_acqu = s
        return f_acqu


class UCB(GPyOpt.acquisitions.base.AcquisitionBase):

    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, cost_withGradients=None, **kwargs):
        self.optimizer = optimizer
        super(UCB, self).__init__(model, space, optimizer)

    def _compute_acq(self, x):
        m, s = self.model.predict(x)
        f_acqu = m + 2 * s
        return f_acqu
