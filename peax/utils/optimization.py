import math
from typing import Tuple, List, Dict, Iterable
import numpy as np

def find_pareto_dominant(points : Iterable[Tuple[int, int]]) -> Tuple[int, int]:

  front = pareto_front(points)
  front = sorted(front, key=lambda x: x[-1]/x[0])

  prev_slope = None
  curr_slope = None
  max_slope = 0

  dominant = None
  max_slope_change = 0

  if len(front) == 1:
    return front[0]

  for i in range(len(front)-1):
    prev_solution = front[i]
    curr_solution = front[i+1]

    prev_slope = curr_slope
    curr_slope = (curr_solution[1] - prev_solution[1]) / (curr_solution[0] - prev_solution[0])

    if curr_slope > max_slope or i == 0:
      max_slope = curr_slope
      dominant = curr_solution
    
    '''# If the slope has changed, update the bending point and maximum change in slope
    if prev_slope is not None and curr_slope != prev_slope:
        slope_change = abs(curr_slope - prev_slope)
        if slope_change > max_slope_change:
            max_slope_change = slope_change
            dominant = (prev_solution, curr_solution)'''

  return dominant

def find_pareto_optimum(points : Iterable[Tuple[int, int]]) -> Tuple[int, int]:
  """Finds the best point on the Pareto Front.
  It is assumed to be the point with the best accuracy per cost.

  Args:
      points (Iterable[Tuple[int, int]]): The points of the search space

  Returns:
      Tuple[int, int]: The best point on the Pareto Front
  """
  front = pareto_front(points)
  front = sorted(front, key=lambda x: x[-1]/x[0])

  best_reward = -np.Infinity
  best_point = None
  for point in front:
    x, y = point

    reward = y/x
    if reward > best_reward:
      best_reward = reward
      best_point = point

  return best_point


def pareto_front(points : Iterable[Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
  """Constructs the Pareto front from a List or Set of points in the search space.

  Args:
      points (List[Tuple[int, int]]): the points in the search space.

  Returns:
      List[Tuple[int, int]]: the points of the search space that lie on the pareto front.
  """
  optimal = []
  for point in points:
    if is_optimal(point, points):
      optimal.append(point)

  optimal = sorted(optimal, key=lambda x: x[-1]/x[0])
  return optimal


def is_optimal(point : Tuple[int, int], points : List[Tuple[int, int]]) -> bool:
  """Checks if a point lies on the Pareto front of the given points.
  Assumes first dimension to be the (relative) accuracy and the second dimension the (relative) cost

  Args:
      point (Tuple[int, int]): point that will be evaluated
      points (List[Tuple[int, int]]): all points in the search space

  Returns:
      bool: True, if optimal / lies on Pareto front
  """

  is_dominated = False
  for other_point in points:
      if (
        other_point[0] < point[0]
        and other_point[-1] > point[-1]
      ):
        is_dominated = True
        break

  return not is_dominated