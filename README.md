# StabilizedPhaseFieldFracture

This FEniCS code is associated with a paper titled "Stabilized formulation for phase-field fracture in nearly incompressible hyperelasticity" published in the International Journal for Numerical Methods in Engineering (IJNME). This work presents a stabilized formulation for phase-field fracture of hyperelastic materials near the limit of incompressibility. At this limit, traditional mixed displacement and pressure formulations must satisfy the inf-sup condition for solution stability. The mixed formulation coupled with the damage field can lead to an inhibition of crack opening as volumetric changes are severely penalized effectively creating a pressure-bubble. To overcome this bottleneck, we utilize a mixed formulation with a perturbed Lagrangian formulation which enforces the incompressibility constraint in the undamaged material and reduces the pressure effect in the damaged material. A mesh-dependent stabilization technique based on the residuals of the Euler-Lagrange equations multiplied with a differential operator acting on the weight space is used, allowing for linear interpolation of all field variables of the elastic sub-problem. 

If using or referencing this code, please cite the associated paper. 

Dependencies and Versioning: 
This code was run in Ubuntu 20.04.4 LTS (Focal Fossa) with python version 3.8.10