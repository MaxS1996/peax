from typing import Iterable, Tuple
from scipy.stats import norm
import numpy as np

def fit(data : Iterable) -> Tuple[float, float]:
  """
  Function to fit a normal distribution to the given data.
  Currently only wraps scipy.stats.norm.fit.

  Args:
      data (Iterable): the given data.

  Returns:
      Tuple[float, float]: mu and sigma of the fitted distribution.
  """

  return norm.fit(data)

def calculate_overlap(mu1:float, sigma1:float, mu2:float, sigma2:float) -> float:
  """Calculates the overlap between two normal distributions.
  Maximum value is 1, minimum is 0.

  Args:
      mu1 (float): mu of the first distribution
      sigma1 (float): sigma of first distribution
      mu2 (float): mu of the second distribution
      sigma2 (float): sigma of second distribution

  Returns:
      float: the overlap. Will be in range [0;1]
  """

  # Find the intersection points (might require numerical methods in general)
  # For simplicity, we'll assume the distributions intersect within a few standard deviations of the means
  lower_bound = min(mu1, mu2) - min(sigma1, sigma2) * 3
  upper_bound = max(mu1, mu2) + max(sigma1, sigma2) * 3
  resolution = 10000  # Increase for higher precision
  x = np.linspace(lower_bound, upper_bound, resolution)
  
  # Calculate the PDF values for both distributions
  pdf1 = norm.pdf(x, mu1, sigma1)
  pdf2 = norm.pdf(x, mu2, sigma2)
  
  # Calculate the minimum PDF at each x to find the overlap
  overlap_pdf = np.minimum(pdf1, pdf2)
  
  # Approximate the integral of the overlap
  area = np.trapz(overlap_pdf, x)
  
  return area