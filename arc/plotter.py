#!/usr/bin/env python
# encoding: utf-8

import logging
import numpy as np
import math
import matplotlib.pyplot as plt

import py3Dmol
from rdkit import Chem

from rmgpy.molecule.molecule import Atom, Molecule


##################################################################


def plot_rotor_scan(angle, v_list):
    """
    plots a 1D rotor PES for v_list vs. angle
    """
    angle = angle * 180 / math.pi  # convert radians to degree
    v_list = np.array(v_list, np.float64)
    v_list = v_list * 0.001  # convert to kJ/mol
    plt.figure(num=None, figsize=(4, 3), dpi=120, facecolor='w', edgecolor='k')
    plt.plot(angle, v_list, 'g.')
    plt.xlabel('dihedral (deg)')
    plt.xlim = (0, 360)
    plt.xticks(np.arange(0, 361, step=60))
    plt.ylabel('V (kJ/mol)')
    plt.tight_layout()
    plt.show()


def show_sticks(xyz):
    """
    Draws the molecule in a "sticks" style according to supplied xyz coordinates
    """
    mol = Molecule()
    coordinates = list()
    for line in xyz.split('\n'):
        if line:
            atom = Atom(element=line.split()[0])
            coordinates.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
            atom.coords = np.array(coordinates[-1], np.float64)
            mol.addAtom(atom)
    mol.connectTheDots()  # only adds single bonds, but we don't care
    rd_mol, rd_inds = mol.toRDKitMol(removeHs=False, returnMapping=True)
    Chem.AllChem.EmbedMolecule(rd_mol)  # unfortunately, this mandatory embedding changes the coordinates
    indx_map = dict()
    for xyz_index, atom in enumerate(mol.atoms):  # generate an atom index mapping dictionary
        rd_index = rd_inds[atom]
        indx_map[xyz_index] = rd_index
    conf = rd_mol.GetConformer(id=0)
    for i in xrange(rd_mol.GetNumAtoms()):  # reset atom coordinates
        conf.SetAtomPosition(indx_map[i], coordinates[i])

    mb = Chem.MolToMolBlock(rd_mol)
    p = py3Dmol.view(width=400, height=400)
    p.addModel(mb, 'sdf')
    p.setStyle({'stick': {}})
    # p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    p.show()


def log_thermo(label, path):
    """
    Logging thermodata from an Arkane output file
    """
    logging.info('\n\n')
    logging.debug('Thermodata for species {0}'.format(label))
    log = False
    with open(path, 'r') as f:
        line = f.readline()
        while line != '':
            if 'Thermodynamics for' in line:
                thermo_block = ''
                log = True
            elif 'thermo(' in line:
                log = False
            if log:
                thermo_block += line[2:]
            line = f.readline()
    logging.info(thermo_block)
    logging.info('\n\n')


