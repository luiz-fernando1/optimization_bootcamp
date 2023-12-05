# Classe de dados

from dataclasses import dataclass

@dataclass
class ValuableItem:
    opcao: str
    value: float
    retorno_esperado: float

    @property
    def value_razao(self) -> float:
        "Returns retorno esperado / value"
        return self.retorno_esperado / (self.value + 1e-9)


# Métodos auxiliares
import pandas as pd 
from typing import List

def items_to_table(opcao: List[ValuableItem]) -> pd.DataFrame:
    records = [{
        'Opção': i.opcao,
        'Valor ($)': i.value,
        'Retorno esperado ($)': i.retorno_esperado
    } for i in opcao]
    records.append({
        'Opcao': 'Total',
        'Valor ($)': sum(i.value for i in opcao),
        'Retorno esperado ($)': sum(i.retorno_esperado for i in opcao)
    })
    return pd.DataFrame.from_records(records)

# Exemplo


capacity = 1000000
values = [470006, 400000, 176000, 270000,
          340000, 230000, 50000, 440000]  # pesos

retorno_esperado = [410000, 330000, 140000, 250000,
                    326000, 326000, 90000, 190006]  # utilidade

available_items = [ValuableItem(
    f'opcao {i+1}', v, w) for i, (v, w) in enumerate(zip(values, retorno_esperado))]

items_to_table(available_items)

# Heuristica gulosa
from typing import List

def greedy_knapsack(
    capacity: float,
    available_items: List[ValuableItem]
) -> List[ValuableItem]:
    chosen_items = list()

    sorted_items = sorted(
        available_items,
        key=lambda i: i.value_razao,
        reverse=True)

    for opcao in sorted_items:
        if opcao.value <= capacity:
            chosen_items.append(opcao)
            capacity -= opcao.value
    return chosen_items


chosen_items = greedy_knapsack(capacity, available_items)
items_to_table(chosen_items)

# Modelagem

# ...

# Solução MILP
# Construção do problema usando a biblioteca science-optimization, usando o algoritmo GLOP (open-source):
# uncomment and run to install framework
# python -m pip install science-optimization


# Contrutor do problema
from science_optimization.builder import (
  BuilderOptimizationProblem, #Classe Abstrata
  Variable, 
  Constraint, 
  Objective, 
  OptimizationProblem
)

from science_optimization. function import (
  FunctionsComposite, 
  LinearFunction,
)

from science_optimization.solvers import Optimizer 
from science_optimization.algorithms.linear_programming import Glop 
import numpy as np

class Knapsack(Builder0ptimizationProblem):
  def __init__(
    self, 
    capacity: float, 
    available_items: List[ValuableItem]):

    self.__capacity = capacity
    self.__itens = available_items

  @property 
  def __num_vars (self) -> int:
    return len(self.__items)
  
  @property 
  def __weights(self) -> np.array:
    return np.array([item.value for item in self.__items]).reshape(-1, 1)
  
  @property
  def __values(self) -> np.array:
    return np.array([item.retorno_esperado for item in self.__items]).reshape(-1, 1)
  
  #Metodo abstrato de Builder0ptimizationProblem
  #Aqui contrai as varaveis  
  def build_variables(self):
    x_min = np.zeros((self.__num_vars, 1))
    x_max = np.ones((self.__num_vars, 1))
    x_type=['d']*self.__num_vars # Discrete variable
    variables =   Variable(x_min, x_max, x_type)
    
    return variables

  #Metodo abstrato de Builder0ptimizationProblem
  #Aqui contrai as restricoes
  def build_constraints(self) -> Constraint:
    """Weights cannot exceed capacity""" 
    # с * x - d <= 0
    constraint = LinearFunction(c=self._weights, d=-self._capacity)
  
    ineq_cons = FunctionsComposite ()
    ineq_cons.add(constraint)
    constraints = Constraint (ineq_cons=ineq_cons)
    return constraints
  
  #Metodo abstrato de Builder0ptimizationProblem
  #Aqui contrai a funcao objetivo
  def build_objectives(self) -> Objective:
    # minimize -v*x
    obj_fun = LinearFunction(c=-self.__values)
     
    obj_funs = FunctionsComposite ()
    obj_funs.add(obj_fun)
    objective = Objective(objective=obj_funs)
    
    return objective

def optimization_problem( 
  capacity: float, 
  available_items: List[ValuableItem],
  verbose: bool = False
) -> OptimizationProblem:
  knapsack = Knapsack(capacity, available_items)
  problem = OptimizationProblem(builder=knapsack)
  if verbose:
    print(problem.info())
  return problem

# Otimização

def run_optimization(
    problem: OptimizationProblem,
    verbose: bool = False
) -> np.array:
    optimizer = Optimizer(
        opt_problem=problem,
        algorithm=Glop()
    )
    results = optimizer.optimize()
    decision_variables = results.x.ravel()
    if verbose:
        print(f"Decision variable:\n{decision_variables}")
    return decision_variables


def knapsack_milp(
        capacity: float,
        items: List[ValuableItem],
        verbose: bool = False) -> List[ValuableItem]:

    problem = optimization_problem(capacity, available_items, verbose)
    decision_variables = run_optimization(problem, verbose)

    # Build list of chosen items
    chosen_items = list()
    for item, item_was_chosen in zip(available_items, decision_variables):
        if item_was_chosen:
            chosen_items.append(item)
    return chosen_items


chosen_items = knapsack_milp(capacity, available_items, verbose=True)
items_to_table(chosen_items)

# Testes


class TestGreedyKnapsack (unittest.TestCase):
    def test_if_no_available_items_knapsack_is_empty(self):
        available_items = list()
        capacity = 15
        chosen_items = greedy_knapsack(capacity, available_items)
        self.assertEqual(chosen_items, list())

    def test_single_item_is_chosen(self):
        available_items = [ValuableItem('ItemX', value=5, retorno_esperado=30)]
        capacity = 15
        chosen_items = greedy_knapsack(capacity, available_items)
        self.assertEqual(chosen_items, available_items)

    def test_item_that_does_not_fit_in_backpack_is_not_chosen(self):
        light_item = ValuableItem('ItemX', value=15, retorno_esperado=1)
        heavy_item = ValuableItem('ItemY', value=16, retorno_esperado=100)
        available_items = [light_item, heavy_item]
        capacity = 15
        chosen_items = greedy_knapsack(capacity, available items)
        self.assertEqual(chosen_items, [light_item])


unittest.main(argv=[''], verbosity=2, exit=False)
