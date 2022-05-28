# StabilizedPhaseFieldFracture

This FEniCS code is associated with a paper titled "Stabilized formulation for
phase-field fracture in nearly incompressible hyperelasticity" published in the
International Journal for Numerical Methods in Engineering (IJNME). If using or
referencing this code, please cite the associated paper.

This work presents a stabilized formulation for phase-field fracture of hyper-
elastic materials near the limit of incompressibility. At this limit, traditional
mixed displacement and pressure formulations must satisfy the inf-sup condition for
solution stability. The mixed formulation coupled with the damage field can lead
to an inhibition of crack opening as volumetric changes are severely penalized
effectively creating a pressure-bubble. To overcome this bottleneck, we utilize
a mixed formulation with a perturbed Lagrangian formulation which enforces the
incompressibility constraint in the undamaged material and reduces the pressure
effect in the damaged material. A mesh-dependent stabilization technique based
on the residuals of the Euler-Lagrange equations multiplied with a differential
operator acting on the weight space is used, allowing for linear interpolation
of all field variables of the elastic sub-problem.

Dependencies and versioning can be checked by 'pip3 list'
  Operating System: Ubuntu 20.04.4 LTS (Focal Fossa)
  Python: 3.8.10
  FEniCS: 2019.2.0.dev0 (type dolfin-version to check)
    Mshr will be the same version since it is packaged in FEniCS
  If using the Gmsh files provided:
    Gmsh: 4.9.5
  matplotlib: 3.1.2
  numpy: 1.22.3
  scipy 1.8.0
  sympy 1.5.1
