Data Files for Standalone Simulator
====================================

This folder originally held documentation for required data files.

IMPORTANT: The actual .mat files are now located in the models/ directory:
  - ../models/ParaV.mat    (Polyhedron vertices)
  - ../models/ParaF.mat    (Polyhedron faces)
  - ../models/SPH.mat      (Spherical harmonics coefficients)

File Descriptions:
------------------

ParaV.mat:
  - Polyhedron vertex coordinates
  - MATLAB struct with 'V' field containing (N, 3) vertex positions [km]
  - Used by PolyhedronGravityFast model
  - Represents asteroid/comet shape geometry

ParaF.mat:
  - Polyhedron face definitions
  - MATLAB struct with 'F' field containing (M, 3) vertex indices
  - Each row defines a triangular face by vertex indices
  - Used in conjunction with ParaV.mat

SPH.mat:
  - Spherical harmonics coefficients
  - Contains Cnm and Snm coefficients up to degree 4
  - Used by SphericalHarmonicsGravityFast model
  - Represents gravity field in spherical harmonic expansion

Data Format:
------------
All files are MATLAB .mat format (loadable via scipy.io.loadmat)

Structure:
  ParaV.mat: {'V': array(N, 3)}
  ParaF.mat: {'F': array(M, 3)}
  SPH.mat:   {'Cnm': array(5, 5), 'Snm': array(5, 5)}

Usage in Code:
--------------
The gravity models automatically load these files from the models/ directory:

  # Polyhedron model
  model = PolyhedronGravityFast(
      vertex_file='path/to/ParaV.mat',
      face_file='path/to/ParaF.mat'
  )

  # Spherical harmonics model
  model = SphericalHarmonicsGravityFast(
      mu=mu,
      coeffs_file='path/to/SPH.mat',
      max_degree=4,
      dims=(a, b, c)
  )

Notes:
------
- Files must be present for polyhedron and SPH models to work
- Point mass and Nagy models do not require these files
- Files are shared across all scripts in the standalone bundle
- Paths are resolved relative to the models/ directory in verify scripts
