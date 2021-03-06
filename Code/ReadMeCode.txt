Read me for FEniCS codes containing information on usage and descriptions of
each code. TH stands for Taylor Hood (non stabilized) formulations and were used
for testing. 

Any of the codes can be run in command line through
  python3 CodeName.py
If using option parameters
  python3 CodeName.py --Parameter1 1 --Parameter2 2

Plane Stress Formulations:
  2D-planestress-TH.py
    RectangleMesh or 2DShearTestRef.xml
    (Displacement + Pressure + F33) + (Damage) Formulation
    Using Taylor-Hood Function Space

  2D-planestress-stabilized.py
    Shear Test rectangle mesh
    (Displacement + Pressure + F33) + (Damage)
    Three terms of stabilization formulation

  2D-planestress-stabilized-trap.py
    Trapezoidal mesh: 2DTrapezoidal.xml
    (Displacement + Pressure + F33) + (Damage)
    Three terms of stabilization formulation

3D Formulations:
  3D-hybrid-TH.py
    BoxMesh
    (Displacement + Pressure) + (Damage)
    Taylor-Hood Elements > No stabilization
    Considers the decomposition of energy into active and passive terms
  3D-hybrid-stabilized.py
    3DEdgeCrack.xml
    (Displacement + Pressure) + (Damage) + Stabilization Formulation
    Stabilization includes all three terms
    Considers the decomposition of energy into active and passive terms
