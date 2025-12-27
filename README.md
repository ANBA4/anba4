[![Actions Status](https://github.com/ANBA4/anba4/workflows/CI_anba4/badge.svg?branch=master)](https://github.com/ANBA4/anba4/actions)
[![Coverage Status](https://coveralls.io/repos/github/ANBA4/anba4/badge.svg?branch=master)](https://coveralls.io/github/ANBA4/anba4?branch=master)

# ANBA4
ANBA4 computes the 6x6 stiffness and mass matrices of arbitrarily complex composite beam cross sections.

## Theory

The theory of ANBA4 is described in this work (and references therein):
Marco Morandini, Maria Chierichetti and Paolo Mantegazza, "Characteristic Behavior of Prismatic Anisotropic Beam Via Generalized Eigenvectors", International Journal of Solids and Structures, Volume 47, Issue 10, 15 May 2010, pp. 1327-1337, https://dx.doi.org/doi:10.1016/j.ijsolstr.2010.01.017, ISSN 0020-7683.

ANBA4 has recently been verified against the commercial solver VABS and validated against experimental measurements. The comparison is described in

Roland Feil, Tobias Pflumm, Pietro Bortolotti and Marco Morandini,  "A cross-sectional aeroelastic analysis and structural optimization tool for slender composite structures", Composite Structures,
Volume 253, Issue 1, December 2020, 112755, https://doi.org/10.1016/j.compstruct.2020.112755.

## License

GPL v3, see COPYING

## Installation

ANBA4 depends on Dolfin, from https://www.fenicsproject.org

Due to this dependency, ANBA4 currently only runs on Linux and Mac.

If you need to stay on Windows your best bets are either the 
Windows Subsystem for Linux (WSL2, https://docs.microsoft.com/en-us/windows/wsl/ )
or Docker (https://docs.microsoft.com/en-us/windows/dev-environment/docker/overview );
see also https://fenicsproject.org/download/ .

On laptop and personal computers, installation with [Anaconda](https://www.anaconda.com) is the suggested approach because of the ability to create self-contained environments suitable for testing and analysis.  If you choose to use Anaconda, keep in mind that ANBA4 needs the 64-bit version (https://www.anaconda.com/distribution/). 

The installation instructions below use the environment name, "anba4-env," but any name is acceptable.    

1.  Setup and activate the Anaconda environment from a Terminal window

        conda create -n anba4-env -y fenics=2019.1.0=py39hf3d152e_26 mshr=2019.1.0 python=3.9
        conda activate anba4-env # (or source activate anba4-env)

2.  Navigate to your preferred folder, clone the repository, and install anba4
        
        cd <toyourpreferredfolder>
        git clone git@github.com:ANBA4/anba4.git # (or git clone https://github.com/ANBA4/anba4.git)
        cd anba_v4
        pip install -e .

3.  Try running an example
    
        cd examples
        python anbax_isotropic.py
