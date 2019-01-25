#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the ARC class
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest

from autotst.reaction import AutoTST_Reaction
from arc.species import get_xyz_string
from rmgpy.reaction import Reaction
from rmgpy.species import Species

from rmgpy.molecule.molecule import Molecule

################################################################################


class TestAutoTST(unittest.TestCase):
    """
    Contains unit tests for AutoTST
    """

    def test_1(self):
        """Test 1"""
        # import pdb; pdb.set_trace()
        # string1 = auto_tst.generate_reaction_string(reactants=[mol1, mol2], products=[mol3, mol4])
        # self.assertEqual(string1, 'CCC+[O]O_[CH2]CC+OO')
        string1 = str('[CH]=CC=C+[O]O_OO+[CH]=C[C]=C')
        # reaction = AutoTST_Reaction(label=string1, reaction_family="H_Abstraction")
        # spc1 = Species().fromSMILES(str('[CH]=CC=C'))
        # spc2 = Species().fromSMILES(str('[O]O'))
        # spc3 = Species().fromSMILES(str('[CH]=C[C]=C'))
        # spc4 = Species().fromSMILES(str('OO'))
        # rxn = Reaction(reactants=[spc1, spc2], products=[spc3, spc4])
        # reaction = AutoTST_Reaction(rmg_reaction=rxn, reaction_family="H_Abstraction")

        # xyz_guess = auto_tst.run_autotsts(reaction_string=string1, family=str('H_Abstraction'))
        # print('\n\n******\n')
        # print(reaction.ts)
        # print(dir(reaction.ts))
        # print('\n\n******\n')
        # positions = reaction.ts.ase_ts.get_positions()
        # numbers = reaction.ts.ase_ts.get_atomic_numbers()
        # xyz_guess = get_xyz_string(xyz=positions, number=numbers)
        # expected_xyz = """1"""
        # self.assertEqual(xyz_guess, expected_xyz)


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
