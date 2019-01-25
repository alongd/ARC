#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)


import os
import logging
import numpy as np

import rdkit
import rdkit.Chem.rdDistGeom
import rdkit.DistanceGeometry
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm3D import EmbedLib

import ase

import rmgpy
from rmgpy.molecule import Molecule
from rmgpy.species import Species
from rmgpy.reaction import Reaction, ReactionError
from rmgpy.data.rmg import RMGDatabase

import autotst
from autotst.base import DistanceData, TransitionStateDepository, TSGroups, TransitionStates
from autotst.molecule import AutoTST_Molecule
from autotst.geometry import Torsion, Angle, Bond, CisTrans

import py3Dmol

from arc.species import get_xyz_string


"""
run AutoTST
"""

##################################################################


def run_autotst(reaction_string='', family=''):
    """
    Run AutoTST
    `reaction_string` is in the form of r1+r2_p1+p2 (e.g., `CCC+[O]O_[CH2]CC+OO`)
    Return XYZ in a string format of the TS guess
    """
    if not family:
        family = str('H_Abstraction')
    if not reaction_string:
        reaction_string = '[CH]=CC=C+[O]O_OO+[CH]=C[C]=C'
    reaction_string = str(reaction_string)
    reaction = AutoTSTReaction(label=reaction_string, reaction_family=family)
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


class AutoTSTReaction():
    """
    COPIED from class AutoTST_Reaction() in AutoTST/autotst/reaction.py
    ################################################################################
    #
    #   AutoTST - Automated Transition State Theory
    #
    #   Copyright (c) 2015-2018 Prof. Richard H. West (r.west@northeastern.edu)
    #
    #   Permission is hereby granted, free of charge, to any person obtaining a
    #   copy of this software and associated documentation files (the 'Software'),
    #   to deal in the Software without restriction, including without limitation
    #   the rights to use, copy, modify, merge, publish, distribute, sublicense,
    #   and/or sell copies of the Software, and to permit persons to whom the
    #   Software is furnished to do so, subject to the following conditions:
    #
    #   The above copyright notice and this permission notice shall be included in
    #   all copies or substantial portions of the Software.
    #
    #   THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    #   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    #   DEALINGS IN THE SOFTWARE.
    #
    ################################################################################
    A class to describe reactions in 3D in ase, rdkit and rmg.

    INPUTS:
    * label (str): a label to describe the reaction in the following format: r1+r2_p1+p2. Where r1, r2, p1, p2 are SMILES strings for the reactants and products.
    * reaction_family (str): a string to describe the rmg_reaction family
    * rmg_reaction (RMG Reaction): an RMG Reaction object that describes the reaction of interest
    """
    rmg_database = None   # will be an RMGDatabase instance, once loaded.
    # a dictionary will have reaction family names as keys and TransitionStates instances as values, once loaded.
    ts_databases = dict()
    possible_families = [  # These families (and only these) will be loaded from both RMG and AutoTST databases
        "Disproportionation",
        "H_Abstraction",
        "intra_H_migration"
    ]

    def __init__(self, label=None, reaction_family=None, rmg_reaction=None):

        assert reaction_family, "Please provide a reaction family."
        assert (label or rmg_reaction), "An rmg_reaction or label needs to be provided."
        assert reaction_family in self.possible_families, "Reaction family is not supported by AutoTST. ({} is not one of {})".format(
            reaction_family, sorted(self.possible_families))

        self.label = label
        self.reaction_family = reaction_family
        self.load_databases()
        # a bit clumsy, but saves refactoring code for now.
        self.ts_database = self.ts_databases[reaction_family]
        self._ts = None
        self._distance_data = None

        if rmg_reaction:
            reactant_mols = []
            product_mols = []

            reactants = rmg_reaction.reactants
            products = rmg_reaction.products

            label = ""

            for i, reactant in enumerate(reactants):
                if type(reactant) == rmgpy.species.Species:
                    reactant_mols.append(AutoTST_Molecule(
                        rmg_molecule=reactant.molecule[0]))
                    label += reactant.molecule[0].toSMILES()
                elif type(reactant) == rmgpy.molecule.molecule.Molecule:
                    reactant_mols.append(
                        AutoTST_Molecule(rmg_molecule=reactant))
                    label += reactant.toSMILES()
                if i+1 != len(reactants):
                    label += "+"

            label += "_"

            for i, product in enumerate(products):
                if type(product) == rmgpy.species.Species:
                    product_mols.append(AutoTST_Molecule(
                        rmg_molecule=product.molecule[0]))
                    label += product.molecule[0].toSMILES()
                elif type(product) == rmgpy.molecule.molecule.Molecule:
                    product_mols.append(AutoTST_Molecule(rmg_molecule=product))
                    label += product.toSMILES()

                if i+1 != len(products):
                    label += "+"

            self.reactant_mols = reactant_mols
            self.product_mols = product_mols
            self.label = label
        elif label and reaction_family:
            logging.info("Label provided: {}".format(label))
            logging.info("Family provided: {}".format(reaction_family))
            self.get_reactants_and_products()

        self.get_rmg_reactions()

    def __repr__(self):
        return '<AutoTST Reaction "{0}">'.format(self.label)

    @property
    def ts(self):
        """
        The AutoTS transition state for this reaction.

        Calls create_ts_geometries() if it has not previously been found.
        To update, call create_ts_geometries() manually.
        """
        print('***1')
        if self._ts is None:
            print('***2')
            self.create_ts_geometries()
        print('***3')
        return self._ts

    @property
    def distance_data(self):
        """
        The distance data.

        Calls generate_distance_data() if it has not previously been found.
        To update, call create_distance_data()
        :return:
        """
        if self._distance_data is None:
            self.generate_distance_data()
        return self._distance_data

    @classmethod
    def load_databases(cls, force_reload=False):
        """
        Load the RMG and AutoTST databases, if they have not already been loaded,
        into the class level variables where they are stored.

        :param force_reload: if set to True then forces a reload, even if already loaded.
        :return: None
        """
        if cls.rmg_database and cls.ts_databases and not force_reload:
            return

        rmg_database = RMGDatabase()
        database_path = rmgpy.settings['database.directory']

        logging.info("Loading RMG database from '{}'".format(database_path))

        try:
            rmg_database.load(database_path,
                              kineticsFamilies=cls.possible_families,
                              transportLibraries=[],
                              reactionLibraries=[],
                              seedMechanisms=[],
                              thermoLibraries=[
                                  'primaryThermoLibrary', 'thermo_DFT_CCSDTF12_BAC', 'CBS_QB3_1dHR'],
                              solvation=False,
                              )
        except IOError:
            logging.info("RMG database not found at '{}'. This can occur if a git repository of the database is being"
                         "used rather than the binary version".format(database_path))
            database_path = os.path.join(database_path, 'input')
            logging.info("Loading RMG database instead from '{}'".format(database_path))
            rmg_database.load(database_path,
                              kineticsFamilies=cls.possible_families,
                              transportLibraries=[],
                              reactionLibraries=[],
                              seedMechanisms=[],
                              thermoLibraries=[
                                  'primaryThermoLibrary', 'thermo_DFT_CCSDTF12_BAC', 'CBS_QB3_1dHR'],
                              solvation=False,
                              )

        cls.rmg_database = rmg_database

        cls.ts_databases = dict()
        for reaction_family in cls.possible_families:
            ts_database = TransitionStates()
            path = os.path.join(autotst.settings['tst_database_path'], reaction_family)
            global_context = {'__builtins__': None}
            local_context = {'DistanceData': DistanceData}
            family = rmg_database.kinetics.families[reaction_family]
            ts_database.family = family
            ts_database.load(path, local_context, global_context)

            cls.ts_databases[reaction_family] = ts_database

    def get_reactants_and_products(self):
        """
        This uses the reaction label to create multi_molecule objects.
        The reaction string should look as follows: r1+r2_p1+p2
        Where r1, r2, p1, p2 are SMILES strings for the reactants and products.

        Stores results in self.reactant_mols and self.product_mols

        :return: None
        """
        reactants, products = self.label.split("_")
        reactants = reactants.split("+")
        products = products.split("+")

        reactant_mols = []
        product_mols = []

        for reactant in reactants:
            reactant_mols.append(AutoTST_Molecule(reactant))

        for product in products:
            product_mols.append(AutoTST_Molecule(product))

        self.reactant_mols = reactant_mols
        self.product_mols = product_mols

    def get_rmg_reactions(self):
        """
        This method creates a labeled rmg_reaction from the reactant and products.

        The reactants and products should be stored in self.reactant_mols
        and self.product_mols, eg. as generated by .get_reactants_and_products()
        """

        rmg_reactants = []
        rmg_products = []

        for reactant_mol in self.reactant_mols:
            rmg_reactants.append(reactant_mol.rmg_molecule)

        for product_mol in self.product_mols:
            rmg_products.append(product_mol.rmg_molecule)

        labeled_r, labeled_p = self.ts_database.family.getLabeledReactantsAndProducts(
            rmg_reactants, rmg_products)

        test_reaction = Reaction(
            reactants=labeled_r, products=labeled_p, reversible=True)

        reaction_list = self.rmg_database.kinetics.generate_reactions_from_families(
            rmg_reactants,
            rmg_products,
            only_families=[self.reaction_family]
        )

        assert reaction_list

        for reaction in reaction_list:
            if reaction.isIsomorphic(test_reaction):
                reaction.reactants = test_reaction.reactants
                reaction.products = test_reaction.products
                break
        else:  # didn't break
            raise ReactionError("Didn't find a {} reaction matching {} in this list: {}".format(
                self.reaction_family, test_reaction, reaction_list))
        self.rmg_reaction = reaction

    def generate_distance_data(self):
        """
        Generates the distance estimates using group additivity.
        Requires self.rmg_reaction
        Stores it in self.distance_data.

        :return: None
        """
        assert self.rmg_reaction, "try calling get_rmg_reactions() first"
        self._distance_data = self.ts_database.groups.estimateDistancesUsingGroupAdditivity(
            self.rmg_reaction)
        logging.info("The distance data is as follows: \n{}".format(
            self.distance_data))

    def create_ts_geometries(self):
        """
        A method to use the tools in rmg / autotst to create a reasonable TS geometry
        This will create the geometry in both rdkit and ase

        :return:
        self.ts: an autotst object object that contains geometries of a ts in
                        rdkit, ase, and rmg molecules
        """
        self._ts = AutoTS(self)


class AutoTS():
    def __init__(self, autotst_reaction):
        print('####### A')
        self.autotst_reaction = autotst_reaction
        print('####### B')
        # make sure that both the reaction and TS have same label
        self.label = autotst_reaction.label
        print('####### C')

        self.create_rdkit_ts_geometry()
        print('####### D')
        self.create_ase_ts_geometry()
        print('####### E')
        self.create_rmg_ts_geometry()
        print('####### F')
        self.get_ts_bonds()
        print('####### H')
        self.get_ts_angles()
        print('####### I')
        self.get_ts_torsions()
        print('####### J')

    def __repr__(self):
        return '<AutoTST TS "{0}">'.format(self.label)

    def create_rdkit_ts_geometry(self):
        print('####### rdkit 1')

        self.rmg_ts, product = self.setup_molecules()
        print('####### rdkit 2')

        self.rmg_ts.updateMultiplicity()
        print('####### rdkit 3')
        combined = self.rmg_ts.toRDKitMol(removeHs=False)
        print('####### rdkit 4')

        labels, atom_match = self.get_labels(self.rmg_ts)
        print('####### rdkit 5')

        for i, atom in enumerate(self.rmg_ts.atoms):
            assert atom.number == combined.GetAtoms()[i].GetAtomicNum()

        print('####### rdkit 6')
        Chem.rdDistGeom.EmbedMolecule(combined)
        print('####### rdkit 7')
        logging.info("Initially embedded molecule")
        print('####### rdkit 8')
        print(combined.GetAtoms())
        print(dir(combined))
        bm = Chem.rdDistGeom.GetMoleculeBoundsMatrix(combined)
        print('####### rdkit 9')

        logging.info("Editing bounds matrix")
        print('####### rdkit 10')
        bm = self.edit_matrix(self.rmg_ts, bm, labels)
        print('####### rdkit 11')

        logging.info("Performing triangle smoothing on bounds matrix.")
        print('####### rdkit 12')
        rdkit.DistanceGeometry.DoTriangleSmoothing(bm)
        print('####### rdkit 13')

        self.bm = bm
        print('####### rdkit 14')

        logging.info("Now attempting to embed using edited bounds matrix.")
        print('####### rdkit 15')

        self.rdkit_ts = self.rd_embed(
            combined, 10000, bm=bm, match=atom_match)[0]
        print('####### rdkit 16')

    def setup_molecules(self):

        merged_reacts = None
        merged_prods = None

        if len(self.autotst_reaction.rmg_reaction.reactants) == 2:
            merged_reacts = Molecule.merge(self.autotst_reaction.rmg_reaction.reactants[0],
                                           self.autotst_reaction.rmg_reaction.reactants[1])

        if len(self.autotst_reaction.rmg_reaction.products) == 2:
            merged_prods = Molecule.merge(self.autotst_reaction.rmg_reaction.products[0],
                                          self.autotst_reaction.rmg_reaction.products[1])

        return merged_reacts, merged_prods

    def get_labels(self, reactants):
        """
        A method to get the labeled atoms from a reaction

        :param reactants: a combined rmg_molecule object
        :return labels: the atom labels corresponding to the reaction center
        :return atomMatch: a tuple of tuples the atoms labels corresponding to the reaction center
        """

        if self.autotst_reaction.rmg_reaction.family.lower() in ['h_abstraction', 'r_addition_multiplebond', 'intra_h_migration']:
            # for i, atom in enumerate(reactants.atoms):
            lbl1 = reactants.getLabeledAtoms()["*1"].sortingLabel
            lbl2 = reactants.getLabeledAtoms()["*2"].sortingLabel
            lbl3 = reactants.getLabeledAtoms()["*3"].sortingLabel
            labels = [lbl1, lbl2, lbl3]
            atomMatch = ((lbl1,), (lbl2,), (lbl3,))
        elif self.autotst_reaction.rmg_reaction.family.lower() in ['disproportionation']:
            lbl1 = reactants.getLabeledAtoms()["*2"].sortingLabel
            lbl2 = reactants.getLabeledAtoms()["*4"].sortingLabel
            lbl3 = reactants.getLabeledAtoms()["*1"].sortingLabel

            labels = [lbl1, lbl2, lbl3]
            atomMatch = ((lbl1,), (lbl2,), (lbl3,))

        logging.info("The labled atoms are {}.".format(labels))

        return labels, atomMatch

    def set_limits(self, bm, lbl1, lbl2, value, uncertainty):
        """
        A method to set the limits of a particular distance between two atoms

        :param bm: an array of arrays corresponding to the bounds matrix
        :param lbl1: the label of one atom
        :param lbl2: the label of another atom
        :param value: the distance from a distance data object (float)
        :param uncertainty: the uncertainty of the `value` distance (float)
        :return bm: an array of arrays corresponding to the edited bounds matrix
        """
        logging.info("For atoms {0} and {1} we have a distance of: \t {2}".format(
            lbl1, lbl2, value))
        if lbl1 > lbl2:
            bm[lbl2][lbl1] = value + uncertainty / 2
            bm[lbl1][lbl2] = max(0, value - uncertainty / 2)
        else:
            bm[lbl2][lbl1] = max(0, value - uncertainty / 2)
            bm[lbl1][lbl2] = value + uncertainty / 2

        return bm

    def edit_matrix(self, rmg_ts, bm, labels):
        """
        A method to set the limits of the reaction center using the `set_limits` method

        :param rmg_ts: an rmg_molecule corresponding the the combined reactants or products (i.e. r1 and r2 or p1 and p2)
        :param bm: an array of arrays that corresponds to the bounds matrix
        :param labels: the atom indices corresponding to the reaction center
        :return bm: the edited bounds matrix
        """
        lbl1, lbl2, lbl3 = labels

        sect = []
        for atom in rmg_ts.split()[1].atoms:
            sect.append(atom.sortingLabel)

        uncertainties = {'d12': 0.02, 'd13': 0.02, 'd23': 0.02}
        bm = self.set_limits(
            bm, lbl1, lbl2, self.autotst_reaction.distance_data.distances['d12'], uncertainties['d12'])
        bm = self.set_limits(
            bm, lbl2, lbl3, self.autotst_reaction.distance_data.distances['d23'], uncertainties['d23'])
        bm = self.set_limits(
            bm, lbl1, lbl3, self.autotst_reaction.distance_data.distances['d13'], uncertainties['d13'])

        bm = self.bm_pre_edit(bm, sect)
        return bm

    def bm_pre_edit(self, bm, sect):
        """
        Clean up some of the atom distance limits before attempting triangle smoothing.
        This ensures any edits made do not lead to unsolvable scenarios for the molecular
        embedding algorithm.

        sect is the list of atom indices belonging to one species.
        """
        others = range(len(bm))
        for idx in sect:
            others.remove(idx)

        for i in range(len(bm)):  # sect:
            for j in range(i):  # others:
                if i < j:
                    continue
                for k in range(len(bm)):
                    if k == i or k == j or i == j:
                        continue
                    Uik = bm[i, k] if k > i else bm[k, i]
                    Ukj = bm[j, k] if k > j else bm[k, j]

                    maxLij = Uik + Ukj - 0.1
                    if bm[i, j] > maxLij:
                        logging.info(
                            "Changing lower limit {0} to {1}".format(bm[i, j], maxLij))
                        bm[i, j] = maxLij

        return bm

    def optimize(self, rdmol, boundsMatrix=None, atomMatch=None):
        """

        Optimizes the rdmol object using UFF.
        Determines the energy level for each of the conformers identified in rdmol.GetConformer.


        :param rdmol:
        :param boundsMatrix:
        :param atomMatch:
        :return rdmol, minEid (index of the lowest energy conformer)
        """

        energy = 0.0
        minEid = 0
        lowestE = 9.999999e99  # start with a very high number, which would never be reached

        for conf in rdmol.GetConformers():
            if boundsMatrix is None:
                AllChem.UFFOptimizeMolecule(rdmol, confId=conf.GetId())
                energy = AllChem.UFFGetMoleculeForceField(
                    rdmol, confId=conf.GetId()).CalcEnergy()
            else:
                _, energy = EmbedLib.OptimizeMol(rdmol, boundsMatrix, atomMatches=atomMatch,
                                                 forceConstant=100000.0)

            if energy < lowestE:
                minEid = conf.GetId()
                lowestE = energy

        return rdmol, minEid

    def rd_embed(self, rdmol, numConfAttempts, bm=None, match=None):
        """
        This portion of the script is literally taken from rmgpy but hacked to work without defining a geometry object

        Embed the RDKit molecule and create the crude molecule file.
        """
        if bm is None:  # bm = bounds matrix?
            AllChem.EmbedMultipleConfs(rdmol, numConfAttempts, randomSeed=1)

            rdmol, minEid = self.optimize(rdmol)
        else:
            """
            Embed the molecule according to the bounds matrix. Built to handle possible failures
            of some of the embedding attempts.
            """
            rdmol.RemoveAllConformers()
            for i in range(0, numConfAttempts):
                try:
                    EmbedLib.EmbedMol(rdmol, bm, atomMatch=match)
                    break
                except ValueError:
                    logging.info("RDKit failed to embed on attempt {0} of {1}".format(
                        i + 1, numConfAttempts))
                except RuntimeError:
                    logging.info("RDKit failed to embed.")
            else:
                logging.error("RDKit failed all attempts to embed")
                return None, None

            """
            RDKit currently embeds the conformers and sets the id as 0, so even though multiple
            conformers have been generated, only 1 can be called. Below the id's are resolved.
            """
            for i in range(len(rdmol.GetConformers())):
                rdmol.GetConformers()[i].SetId(i)

            rdmol, minEid = self.optimize(
                rdmol, boundsMatrix=bm, atomMatch=match)

        return rdmol, minEid

    def create_ase_ts_geometry(self):

        mol_list = AllChem.MolToMolBlock(self.rdkit_ts).split('\n')
        ase_atoms = []
        for i, line in enumerate(mol_list):

            if i > 3:

                try:
                    atom0, atom1, bond, rest = line
                    atom0 = int(atom0)
                    atom0 = int(atom1)
                    bond = float(bond)

                except ValueError:
                    try:
                        x, y, z, symbol = line.split()[0:4]
                        x = float(x)
                        y = float(y)
                        z = float(z)

                        ase_atoms.append(
                            ase.Atom(symbol=symbol, position=(x, y, z)))

                    except:
                        continue

        self.ase_ts = ase.Atoms(ase_atoms)

    def create_rmg_ts_geometry(self):

        for i, position in enumerate(self.ase_ts.get_positions()):
            self.rmg_ts.atoms[i].coords = position

    def view_ts(self, mol=None):
        """
        A method designed to create a 3D figure of the Multi_Molecule with py3Dmol
        """
        if not mol:
            mol = self.rdkit_ts

        mb = Chem.MolToMolBlock(mol)
        p = py3Dmol.view(width=400, height=400)
        p.addModel(mb, "sdf")
        p.setStyle({'stick': {}})
        p.setBackgroundColor('0xeeeeee')
        p.zoomTo()
        return p.show()

    def create_pseudo_geometry(self):

        # TODO: This only works for some reaction families, need to fix this

        rdmol_copy = self.rdkit_ts.__copy__()
        rdmol_copy = Chem.RWMol(rdmol_copy)
        rdmol_copy.UpdatePropertyCache(strict=False)

        for idx, rmg_atom in enumerate(self.rmg_ts.atoms):
            if rmg_atom.label == "*1":
                star1_index = idx
            if rmg_atom.label == "*2":
                star2_index = idx
            if rmg_atom.label == "*3":
                star3_index = idx

        if not rdmol_copy.GetBondBetweenAtoms(star1_index, star2_index):
            rdmol_copy.AddBond(star1_index, star2_index,
                               order=rdkit.Chem.rdchem.BondType.SINGLE)
        else:
            rdmol_copy.AddBond(star2_index, star3_index,
                               order=rdkit.Chem.rdchem.BondType.SINGLE)

        return rdmol_copy

    def get_ts_bonds(self):

        rdmol_copy = self.create_pseudo_geometry()
        bond_list = []
        for bond in rdmol_copy.GetBonds():
            bond_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        bonds = []
        for indices in bond_list:
            i, j = indices

            length = self.ase_ts.get_distance(i, j)

            reaction_center = "No"

            if (self.rmg_ts.atoms[i].label != "" and
                    self.rmg_ts.atoms[j].label != ""):
                reaction_center = "Yes"

            elif ((self.rmg_ts.atoms[i].label != "" and self.rmg_ts.atoms[j].label == "") or
                  (self.rmg_ts.atoms[i].label == "" and self.rmg_ts.atoms[j].label != "")):
                reaction_center = "Close"

            bond = Bond(indices=indices, length=length,
                        reaction_center=reaction_center)

            bonds.append(bond)
        self.bonds = bonds
        return self.bonds

    def get_ts_angles(self):

        rdmol_copy = self.create_pseudo_geometry()

        angle_list = []
        for atom1 in rdmol_copy.GetAtoms():
            for atom2 in atom1.GetNeighbors():
                for atom3 in atom2.GetNeighbors():
                    if atom1.GetIdx() == atom3.GetIdx():
                        continue

                    to_add = (atom1.GetIdx(), atom2.GetIdx(), atom3.GetIdx())
                    if (to_add in angle_list) or (tuple(reversed(to_add)) in angle_list):
                        continue
                    angle_list.append(to_add)

        angles = []
        for indices in angle_list:
            i, j, k = indices

            degree = self.ase_ts.get_angle(i, j, k)
            ang = Angle(indices=indices, degree=degree,
                        left_mask=[], right_mask=[])
            left_mask = self.get_ts_left_mask(ang)
            right_mask = self.get_ts_right_mask(ang)

            reaction_center = "No"

            if (self.rmg_ts.atoms[i].label != "" and
                self.rmg_ts.atoms[j].label != "" and
                    self.rmg_ts.atoms[k].label != ""):

                reaction_center = "Yes"
            elif ((self.rmg_ts.atoms[i].label != "" and
                   self.rmg_ts.atoms[j].label != "") or
                  (self.rmg_ts.atoms[j].label != "" and
                   self.rmg_ts.atoms[k].label != "")):

                reaction_center = "Close"

            angles.append(Angle(indices, degree, left_mask,
                                right_mask, reaction_center))
        self.angles = angles
        return self.angles

    def get_ts_torsions(self):
        rdmol_copy = self.create_pseudo_geometry()
        torsion_list = []
        cistrans_list = []

        for bond1 in rdmol_copy.GetBonds():
            atom1 = bond1.GetBeginAtom()
            atom2 = bond1.GetEndAtom()
            if atom1.IsInRing() or atom2.IsInRing():
                # Making sure that bond1 we're looking at are in a ring
                continue

            bond_list1 = list(atom1.GetBonds())
            bond_list2 = list(atom2.GetBonds())

            if not len(bond_list1) > 1 and not len(bond_list2) > 1:
                # Making sure that there are more than one bond attached to
                # the atoms we're looking at
                continue

            # Getting the 0th and 3rd atom and insuring that atoms
            # attached to the 1st and 2nd atom are not terminal hydrogens
            # We also make sure that all of the atoms are properly bound together

            # If the above are satisfied, we append a tuple of the torsion our torsion_list
            got_atom0 = False
            got_atom3 = False

            for bond0 in bond_list1:
                atomX = bond0.GetOtherAtom(atom1)
                if atomX.GetIdx() != atom2.GetIdx():
                    got_atom0 = True
                    atom0 = atomX

            for bond2 in bond_list2:
                atomY = bond2.GetOtherAtom(atom2)
                if atomY.GetIdx() != atom1.GetIdx():
                    got_atom3 = True
                    atom3 = atomY

            if not (got_atom0 and got_atom3):
                # Making sure atom0 and atom3 were not found
                continue

            # Looking to make sure that all of the atoms are properly bonded to eached
            if ("SINGLE" in str(rdmol_copy.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx()).GetBondType()) and
                rdmol_copy.GetBondBetweenAtoms(atom0.GetIdx(), atom1.GetIdx()) and
                rdmol_copy.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx()) and
                    rdmol_copy.GetBondBetweenAtoms(atom2.GetIdx(), atom3.GetIdx())):
                torsion_tup = (atom0.GetIdx(), atom1.GetIdx(),
                               atom2.GetIdx(), atom3.GetIdx())

                already_in_list = False
                for torsion_entry in torsion_list:
                    a, b, c, d = torsion_entry
                    e, f, g, h = torsion_tup

                    if (b, c) == (f, g) or (b, c) == (g, f):
                        already_in_list = True

                if not already_in_list:
                    torsion_list.append(torsion_tup)

            if ("DOUBLE" in str(rdmol_copy.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx()).GetBondType()) and
                rdmol_copy.GetBondBetweenAtoms(atom0.GetIdx(), atom1.GetIdx()) and
                rdmol_copy.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx()) and
                    rdmol_copy.GetBondBetweenAtoms(atom2.GetIdx(), atom3.GetIdx())):

                torsion_tup = (atom0.GetIdx(), atom1.GetIdx(),
                               atom2.GetIdx(), atom3.GetIdx())

                already_in_list = False
                for torsion_entry in torsion_list:
                    a, b, c, d = torsion_entry
                    e, f, g, h = torsion_tup

                    if (b, c) == (f, g) or (b, c) == (g, f):
                        already_in_list = True

                if not already_in_list:
                    cistrans_list.append(torsion_tup)

        torsions = []
        cistrans = []
        for indices in torsion_list:
            i, j, k, l = indices

            dihedral = self.ase_ts.get_dihedral(i, j, k, l)
            tor = Torsion(indices=indices, dihedral=dihedral,
                          left_mask=[], right_mask=[])
            left_mask = self.get_ts_left_mask(tor)
            right_mask = self.get_ts_right_mask(tor)
            reaction_center = "No"

            if ((self.rmg_ts.atoms[i].label != "" and
                 self.rmg_ts.atoms[j].label != "" and
                 self.rmg_ts.atoms[k].label != "") or (
                self.rmg_ts.atoms[j].label != "" and
                self.rmg_ts.atoms[k].label != "" and
                    self.rmg_ts.atoms[l].label != "")):
                reaction_center = "Yes"

            torsions.append(Torsion(indices, dihedral,
                                    left_mask, right_mask, reaction_center))

        for indices in cistrans_list:
            i, j, k, l = indices

            dihedral = self.ase_ts.get_dihedral(i, j, k, l)
            tor = CisTrans(indices=indices, dihedral=dihedral,
                           left_mask=[], right_mask=[])
            left_mask = self.get_ts_left_mask(tor)
            right_mask = self.get_ts_right_mask(tor)
            reaction_center = "No"

            cistrans.append(CisTrans(indices, dihedral,
                                     left_mask, right_mask, reaction_center))

        self.torsions = torsions
        self.cistrans = cistrans
        return self.torsions

    def get_ts_right_mask(self, torsion_or_angle):

        rdmol_copy = self.create_pseudo_geometry()

        rdkit_atoms = rdmol_copy.GetAtoms()

        if (isinstance(torsion_or_angle, autotst.geometry.Torsion) or
                isinstance(torsion_or_angle, autotst.geometry.CisTrans)):

            L1, L0, R0, R1 = torsion_or_angle.indices

            # trying to get the left hand side of this torsion
            LHS_atoms_index = [L0, L1]
            RHS_atoms_index = [R0, R1]

        elif isinstance(torsion_or_angle, autotst.geometry.Angle):
            a1, a2, a3 = torsion_or_angle.indices
            LHS_atoms_index = [a2, a1]
            RHS_atoms_index = [a2, a3]

        complete_RHS = False
        i = 0
        atom_index = RHS_atoms_index[0]
        while complete_RHS is False:
            try:
                RHS_atom = rdkit_atoms[atom_index]
                for neighbor in RHS_atom.GetNeighbors():
                    if (neighbor.GetIdx() in RHS_atoms_index) or (neighbor.GetIdx() in LHS_atoms_index):
                        continue
                    else:
                        RHS_atoms_index.append(neighbor.GetIdx())
                i += 1
                atom_index = RHS_atoms_index[i]

            except IndexError:
                complete_RHS = True

        right_mask = [
            index in RHS_atoms_index for index in range(len(self.ase_ts))]

        return right_mask

    def get_ts_left_mask(self, torsion_or_angle):

        rdmol_copy = self.create_pseudo_geometry()

        rdkit_atoms = rdmol_copy.GetAtoms()

        if (isinstance(torsion_or_angle, autotst.geometry.Torsion) or
                isinstance(torsion_or_angle, autotst.geometry.CisTrans)):

            L1, L0, R0, R1 = torsion_or_angle.indices

            # trying to get the left hand side of this torsion
            LHS_atoms_index = [L0, L1]
            RHS_atoms_index = [R0, R1]

        elif isinstance(torsion_or_angle, autotst.geometry.Angle):
            a1, a2, a3 = torsion_or_angle.indices
            LHS_atoms_index = [a2, a1]
            RHS_atoms_index = [a2, a3]

        complete_LHS = False
        i = 0
        atom_index = LHS_atoms_index[0]
        while complete_LHS is False:
            try:
                LHS_atom = rdkit_atoms[atom_index]
                for neighbor in LHS_atom.GetNeighbors():
                    if (neighbor.GetIdx() in LHS_atoms_index) or (neighbor.GetIdx() in RHS_atoms_index):
                        continue
                    else:
                        LHS_atoms_index.append(neighbor.GetIdx())
                i += 1
                atom_index = LHS_atoms_index[i]

            except IndexError:
                complete_LHS = True

        left_mask = [
            index in LHS_atoms_index for index in range(len(self.ase_ts))]

        return left_mask

    def set_rmg_ts_coords(self, molecule_base):

        if molecule_base == "RDKit":
            mol_list = AllChem.MolToMolBlock(self.rdkit_ts).split('\n')
            for i, atom in enumerate(self.rmg_ts.atoms):

                j = i + 4
                coords = mol_list[j].split()[:3]

                for k, coord in enumerate(coords):

                    coords[k] = float(coord)
                atom.coords = np.array(coords)

        elif molecule_base == "ASE":
            for i, position in enumerate(self.ase_ts.get_positions()):
                self.rmg_ts.atoms[i].coords = position

    def update_from_rdkit_ts(self):
        # In order to update the ase molecule you simply need to rerun the get_ase_molecule method
        self.create_ase_ts_geometry()
        self.set_rmg_ts_coords("RDKit")

        # Getting the new torsion angles
        self.get_ts_torsions()

    def update_from_ase_ts(self):

        self.set_rmg_ts_coords("ASE")

        # setting the geometries of the rdkit molecule

        positions = self.ase_ts.get_positions()

        conf = self.rdkit_ts.GetConformers()[0]

        for i, atom in enumerate(self.rdkit_ts.GetAtoms()):
            conf.SetAtomPosition(i, positions[i])

        # Getting the new torsion angles
        self.get_ts_torsions()

    def update_from_rmg_ts(self):

        conf = self.rdkit_ts.GetConformers()[0]
        ase_atoms = []
        for i, atom in enumerate(self.rmg_ts.atoms):
            x, y, z = atom.coords
            symbol = atom.symbol

            conf.SetAtomPosition(i, [x, y, z])

            ase_atoms.append(ase.Atom(symbol=symbol, position=(x, y, z)))

        self.ase_ts = ase.Atoms(ase_atoms)

        # Getting the new torsion angles
        self.get_ts_torsions()


