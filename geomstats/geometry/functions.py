from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.base import VectorSpace
import math
import numpy as np
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.manifold import Manifold
import geomstats.backend as gs
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz


import pdb


class L2SpaceMetric(RiemannianMetric):
    """A Riemannian metric on the L2 space
    """
    def __init__(self,domain_samples):
        self.domain = domain_samples
        self.x = (self.domain-min(self.domain))/(max(self.domain)- min(self.domain))
        self.dim = len(self.domain)
        super().__init__(dim=self.dim)
    
    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner-product of two tangent vectors at a base point.

        """
        inner_prod = np.trapz(tangent_vec_a * tangent_vec_b, x = self.x)
        return inner_prod
    
    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential.
        """
       
        return base_point+tangent_vec
    
    def log(self, point, base_point, **kwargs):
        """Compute Riemannian logarithm of a point wrt a base point.
        """
        return point-base_point
        
        
class L2Space(VectorSpace):
    """Class for space of L2 functions.
    
    Real valued square interable functions defined on a unit interval are Hilbert spaces with a Riemannian inner product
    This class represents such manifolds.
    The L2Space (Lp in general) is a Banach Space that is a complete normed Vector space
    
    Ref : 
    Functional and Shape Data Analysis
    """
    def __init__(self, domain_samples):
        self.domain = domain_samples
        self.dim = len(self.domain)
        super().__init__(shape=(self.dim, ), metric=L2SpaceMetric(self.domain))
        
        
class SinfSpaceMetric(RiemannianMetric):
	"""A Riemannian metric on the S_{\inf} space

	Inputs:
	-------
	domain_samples : grid points on the domain (array of shape (n_samples, ))
	"""
	def __init__(self,domain_samples):
		self.domain = domain_samples
		self.x = (self.domain-min(self.domain))/(max(self.domain)- min(self.domain))
		self.dim = len(self.domain)
		super().__init__(dim=self.dim)

	def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
		"""Compute the inner-product of two tangent vectors at a base point.
		"""
		inner_prod = np.trapz(tangent_vec_a * tangent_vec_b, x = self.x)
		return inner_prod

	def exp(self, tangent_vec, base_point, **kwargs):
		"""Compute the Riemannian exponential.
		"""
		norm_v = self.norm(tangent_vec)
		t1 = np.cos(norm_v)*base_point
		t2 = (np.sin(norm_v)/norm_v)*base_point

		return t1+t2

	def log(self, point, base_point, **kwargs):
		"""Compute Riemannian logarithm of a point wrt a base point.
		"""
		theta = np.arccos(self.inner_product(point, base_point, base_point))

		return (point-base_point*np.cos(theta))*(theta/np.sin(theta))
        
        
class SinfSpace(VectorSpace):
    """Class for space of L2 functions with norm 1.
    
    Real valued square interable functions defined on a unit interval are Hilbert spaces with a Riemannian inner product
    This class represents such manifolds.
    The L2Space (Lp in general) is a Banach Space that is a complete normed Vector space
    
    Ref : 
    Functional and Shape Data Analysis
    """
    def __init__(self, domain_samples):
        self.domain = domain_samples
        self.dim = len(self.domain)
        super().__init__(shape=(self.dim, ), metric=L2SpaceMetric(self.domain))
        
    def projection(self, point):
        norm_p = self.metric.norm(point)
        return point/norm_p  
        

class ProbabilityDistributions(Manifold):
    """Class for the space of probability distributions
    
    This space of functions becomes a Riemannian manifold with the Fisher-Rao distance measure.
    For computational convienience, we make them into the following transformation q(x) = \sqrt(g(x)) when 
    g(x) \in P = {g | \int_{0}^{1} g(x)=1}
    This makes this a manifold of S^{\inf}
    """
    def __init__(self, domain_samples):
        self.x = domain_samples
        self.dim = len(self.x)
        super().__init__(dim = self.dim, metric=FisherRaoMetric(self.x))
        
    def belongs(self, point, atol=gs.atol):
        
        return gs.isclose(np.trapz(point, x=self.x), 1.0, atol=atol)
        
    def projection(self, point):
        
        return point/np.trapz(point, x=self.x) 
        
    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        
        return gs.isclose(np.trapz(vector, x=self.x), 0.0, atol)
        
    def to_tangent(self, vector, base_point=None):
        w = vector - np.trapz(vector, x=self.x)
        
        return w
        
    def random_point(self, n_samples=1, bound=1.0):
        point = gs.random.rand(n_samples, self.dim)
        
        return self.projection(point)

class FisherRaoMetric(RiemannianMetric):
    """The Fisher-Rao distace metric on unit Hilbert sphere
    """
    def __init__(self,domain_samples):
        self.domain = domain_samples
        self.x = self.domain #(self.domain-min(self.domain))/(max(self.domain)- min(self.domain))
        self.dim = len(self.domain)
        super().__init__(dim=self.dim)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute non-parametric Fisher-Rao inner product
        """
        integrand = tangent_vec_a*tangent_vec_b
        
        return np.trapz(integrand, x=self.x)
        
    def exp(self, tangent_vec, base_point, **kwargs):
        norm = self.squared_norm(tangent_vec)
        exp = np.cos(norm)[:,np.newaxis]*base_point + np.sin(norm)[:,np.newaxis]*(tangent_vec/norm[:,np.newaxis])
        
        return exp 
        
    def log(self, point, base_point, **kwargs):
        theta = np.cos(self.inner_product(point, base_point))
        log = (theta/np.sin(theta)[:,np.newaxis])*(base_point-point*np.cos(theta)[:,np.newaxis])
        
        return log 
        
    def dist(self, point_a, point_b):
        theta = np.trapz(np.sqrt(point_a*point_b), x=self.x)
        theta = gs.clip(theta, -1, 1)
        
        return gs.arccos(theta)
        
    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        
        if end_point is not None:
            theta = np.cos(self.inner_product(end_point, initial_point))
            def path(t):
                t = t[:,np.newaxis]
                t1 = (np.sin(theta*(1-t)))*initial_point 
                t2 = np.sin(t*theta)*end_point
                
                return (1/np.sin(theta))*(t1+t2)
            
        else:
            norm = self.squared_norm(initial_tangent_vec)
            def path(t):
                t = t[:,np.newaxis]
                t1 = np.cos(t*norm)*base_point
                t2 = np.sin(t*norm)*(tangent_vec/norm)
                
                return t1+t2
            
        return path 
        
class SRVF(Manifold):
    def __init__(self, domain_samples):
        self.x = domain_samples
        self.eval_dim = len(self.x)
        super().__init__(dim=np.inf , metric=SRVFMetric(self.x))
         
    def projection(self, point):
        return self.metric.to_srvf(point)
        
    def to_tangent(self, vector, base_point):
        return self.metric.to_tangent(vector, base_point)
        
    def belongs(self, point, atol=gs.atol):
        return len(point)==self.eval_dim
        
    def is_tangent(self, vector, base_point, atol=gs.atol):
        return self.belongs(vector)
    
    def random_point(self, n_samples=1, bound=1.0):
        return gs.random.rand(self.eval_dim, n_samples)
    
class SRVFMetric(RiemannianMetric):
    """A Riemannian metric on the SRVF space
    
    We transform a function to its SRVF representation and use L2 space computations
    to do the differential geometry
    """
    def __init__(self,domain_samples):
        self.domain = domain_samples
        self.x = (self.domain-min(self.domain))/(max(self.domain)- min(self.domain))
        self.eval_dim = len(self.domain)
        self.eps = np.finfo(np.double).eps
        super().__init__(dim=np.inf)
        
    def to_srvf(self, point):
        fdot = np.gradient(point, self.x)
        q = np.sign(fdot)*np.sqrt(np.fabs(fdot) + self.eps)
        return q
        
    def to_function(self, q, f0=0.0):
        integrand = q*np.fabs(q)
        f = cumtrapz(integrand,self.x, initial=f0)
        return f
        
    def to_tangent(self, vector, base_point):
        fdot = np.gradient(base_point, self.x)
        w_denom = 2*np.sqrt(np.fabs(fdot) + self.eps)
        if vector.ndim==1:
            vector = vector.reshape(1,len(vector))
        w = np.zeros(np.shape(vector))
        for i, vi in enumerate(vector):
            vdot = np.gradient(vi, self.x)
            w[i,...] = (vdot[:,None]/w_denom[:,None]).squeeze()
            
        return w.squeeze() if len(w)==1 else w

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner-product of two tangent vectors at a base point.

        """
        w_a = self.to_tangent(tangent_vec_a, base_point)
        w_b = self.to_tangent(tangent_vec_b, base_point)
        inner_prod = np.trapz(w_a * w_b, x = self.x)
        
        return inner_prod
    
    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential.
        """
        w = self.to_tangent(tangent_vec, base_point)
        q = self.to_srvf(base_point)
       
        return self.to_function(q+w)
    
    def log(self, point, base_point, **kwargs):
        """Compute Riemannian logarithm of a point wrt a base point.
        """
        q_base_point = self.to_srvf(base_point)
        q_point = self.to_srvf(point)
        
        return self.to_function(q_point-q_base_point)
        
    def dist(self, point_a, point_b):
        qa = self.to_srvf(point_a)
        qb = self.to_srvf(point_b)
        vec_ab = qa-qb
        
        return np.sqrt(np.trapz(gs.power(vec_ab,2), x=self.x))   
        
    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        q1 = self.to_srvf(initial_point)
        
        if end_point is not None:
            q2 = self.to_srvf(end_point)
            def path(t):
                t = t[:,np.newaxis]
                return self.to_function((1-t)*q1 + t*q2)
        else:
            v = self.to_tangent(initial_tangent_vec)
            def path(t):
                t = t[:,np.newaxis]
                return self.to_function(q1 + t*v)
            
        return path 
    




        
             