"""Utility functions for reading the datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, NamedTuple, Literal
from copy import deepcopy

@dataclass
class Metadata:
  periodic: bool
  group_u: str
  group_c: str
  group_x: str
  type: Literal['poseidon', 'rigno', 'gaot']
  fix_x: bool
  domain_x: tuple[Sequence[int], Sequence[int]]
  domain_t: tuple[int, int]
  active_variables: Sequence[int]  # Index of variables in input/output
  chunked_variables: Sequence[int]  # Index of variable groups
  num_variable_chunks: int  # Number of variable chunks
  signed: dict[str, Union[bool, Sequence[bool]]]
  names: dict[str, Sequence[str]]
  global_mean: Sequence[float]
  global_std: Sequence[float]

"""
Reference: https://github.com/camlab-ethz/rigno/blob/main/rigno/dataset.py
"""

ACTIVE_VARS_NS = [0, 1]
ACTIVE_VARS_CE = [0, 1, 2, 3]
ACTIVE_VARS_GCE = [0, 1, 2, 3, 5]
ACTIVE_VARS_RD = [0]
ACTIVE_VARS_WE = [0]
ACTIVE_VARS_PE = [0]

CHUNKED_VARS_NS = [0, 0]
CHUNKED_VARS_CE = [0, 1, 1, 2, 3]
CHUNKED_VARS_GCE = [0, 1, 1, 2, 3, 4]
CHUNKED_VARS_RD = [0]
CHUNKED_VARS_WE = [0]
CHUNKED_VARS_PE = [0]

SIGNED_NS = {'u': [True, True], 'c': None}
SIGNED_CE = {'u': [False, True, True, False, False], 'c': None}
SIGNED_GCE = {'u': [False, True, True, False, False, False], 'c': None}
SIGNED_RD = {'u': [True], 'c': None}
SIGNED_WE = {'u': [True], 'c': [False]}
SIGNED_PE = {'u': [True], 'c': [True]}

NAMES_NS = {'u': ['$v_x$', '$v_y$'], 'c': None}
NAMES_CE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$'], 'c': None}
NAMES_GCE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', 'E', '$\\phi$'], 'c': None}
NAMES_RD = {'u': ['$u$'], 'c': None}
NAMES_WE = {'u': ['$u$'], 'c': ['$c$']}
NAMES_PE = {'u': ['$u$'], 'c': ['$f$']}

DATASET_METADATA = {
  # steady Euler
  'compressible_flow/naca2412': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([-1, -1.5], [2.5, 2]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False, False]},
    names={'u': ['$\\rho$'], 'c': ['Mach', 'AOA', 'SDF']},
    global_mean=[0.96086993],
    global_std=[0.18490477],
  ),
  'compressible_flow/naca0012': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([-1, -1.5], [2.5, 2]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False, False]},
    names={'u': ['$\\rho$'], 'c': ['Mach', 'AOA', 'SDF']},
    global_mean=[0.96999054],
    global_std=[0.17089098],
  ),
  'compressible_flow/rae2822': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([-1, -1.5], [2.5, 2]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False, False]},
    names={'u': ['$\\rho$'], 'c': ['Mach', 'AOA', 'SDF']},
    global_mean=[0.96746538],
    global_std=[0.17268029],
  ),
  'compressible_flow/bluff': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([-9.0, -9.0], [9.0, 9.0]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False, False]},
    names={'u': ['$\\rho$'], 'c': ['Mach', 'AOA', 'SDF']},
    global_mean=[0.95306754],
    global_std=[0.3144897],
  ),
 
  # compressible_flow: [density, velocity, velocity, pressure, energy]
  'compressible_flow/CE-Gauss': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 2.513],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/CE-RP': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.215],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/CE-CRP': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.553],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/CE-KH': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 1.0],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/CE-RPUI': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 1.33],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
 
  # incompressible_fluids: [velocity, velocity]
  'incompressible_fluids/NS-Gauss': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/NS-PwC': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/NS-SL': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/NS-SVS': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/NS-Sines': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
 
  # elliptic PDEs
  'elliptic_pdes/Elasticity': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False]},
    names={'u': ['$\\sigma$'], 'c': ['$d$']},
    global_mean=[187.477],
    global_std=[127.046],
  ),
  'elliptic_pdes/Poisson-C-Sines': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=None,
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': [True]},
    names={'u': ['$u$'], 'c': ['$f$']},
    global_mean=[0.],
    global_std=[0.00064911455],
  ),
  'elliptic_pdes/Poisson-Gauss': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),

  # Parabolic PDEs
  'parabolic_pdes/Heat-L-Sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0., 0.], [1., 1.]),
    domain_t=(0, 0.002),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[-0.009399102],
    global_std=[0.020079814],
  ),
  'parabolic_pdes/ACE': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 0.0002),
    fix_x=True,
    active_variables=ACTIVE_VARS_RD,
    chunked_variables=CHUNKED_VARS_RD,
    num_variable_chunks=len(set(CHUNKED_VARS_RD)),
    signed=SIGNED_RD,
    names=NAMES_RD,
    global_mean=[0.002484262],
    global_std=[0.65351176],
  ),
  
  # Hyperbolic PDEs
  'hyperbolic_pdes/Wave-C-Sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=(0, 0.1),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[0.],
    global_std=[0.011314605],
  ),
  'hyperbolic_pdes/Wave-Layer': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.03467443221585092],
    global_std=[0.10442421752963911],
  ),
  'hyperbolic_pdes/Wave-Gauss': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.0334376316],
    global_std=[0.1171879068],
  ),
  'hyperbolic_pdes/Wave-L-Sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0.5, 0.], [1.5, 1.]),
    domain_t=(0, 0.1),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[0.],
    global_std=[0.01080257],
  ),

}

