#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)

from autotst.reaction import AutoTST_Reaction

from arc.species import get_xyz_string

"""
run AutoTST
"""

##################################################################


def run_autotsts(reaction_string, family=''):
    """
    Run AutoTST
    `reaction_string` is in the form of r1+r2_p1+p2 (e.g., `CCC+[O]O_[CH2]CC+OO`)
    Return XYZ in a string format of the TS guess
    """
    if not family:
        family = str('H_Abstraction')
    reaction_string = str(reaction_string)
    reaction = AutoTST_Reaction(label=reaction_string, reaction_family=family)
    positions = reaction.ts.ase_ts.get_positions()
    numbers = reaction.ts.ase_ts.get_atomic_numbers()
    xyz_guess = get_xyz_string(xyz=positions, number=numbers)
    return xyz_guess


def generate_reaction_string(reactants, products):
    """
    Returns the AutoTST reaction string in the form of r1+r2_p1+p2 (e.g., `CCC+[O]O_[CH2]CC+OO`).
    `reactants` and `products` are lists of class:`Molecule`s.
    """
    if len(reactants) > 1:
        reactants_string = '+'.join([reactant.toSMILES() for reactant in reactants])
    else:
        reactants_string = reactants[0].toSMILES()
    if len(products) > 1:
        products_string = '+'.join([product.toSMILES() for product in products])
    else:
        products_string = products[0].toSMILES()
    reaction_string = '_'.join([reactants_string, products_string])
    return reaction_string
