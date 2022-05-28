// 2D version of Shear Test
//
// Usage to convert for FEniCS
// geo name.geo -format msh2 -3
// dolfin-convert name.msh name.xml

Mesh.Algorithm = 8; // Delaunay for quads

// Mesh sizes
ms_r = 0.02; // More refined for the phase-field width and defines hsize
ms = 0.15;   // Less refined

// Dimensions in x y
x_d = 6;
y_d = 1;
// This section contains the phase field
pf = 0.15;

// Points for plane defined counter-clockwise
Point(1) = {0, 0, 0, ms_r};
Point(2) = {x_d/2, 0, 0, ms_r};
Point(3) = {x_d/2, pf, 0, ms_r};
Point(4) = {x_d/2, y_d/2, 0, ms};
Point(5) = {-x_d/2, y_d/2, 0, ms};
Point(6) = {-x_d/2, pf, 0, ms_r};
Point(7) = {-x_d/2, 0, 0, ms_r};
Point(8) = {-x_d/2, -pf, 0, ms_r};
Point(9) = {-x_d/2, -y_d/2, 0, ms};
Point(10) = {x_d/2, -y_d/2, 0, ms};
Point(11) = {x_d/2, -pf, 0, ms_r};

// Circumference lines
Line(1) = {1,2}; Line(2) = {2,3};
Line(3) = {3,4}; Line(4) = {4,5};
Line(5) = {5,6}; Line(6) = {6,7};
Line(7) = {7,8}; Line(8) = {8,9};
Line(9) = {9,10}; Line(10) = {10,11};
Line(11) = {11,2};
// Line making up crack edge
Line(12) = {7,1};
// Mid lines
Line(13) = {3,6};
Line(14) = {8,11};

// Join lines together to form a surface
Line Loop(1) = {1,2,13,6,12};
Line Loop(2) = {3,4,5,-13};
Line Loop(3) = {14,11,-1,-12,7};
Line Loop(4) = {8,9,10,-14};
// Convert line loop to surface
Plane Surface(50) = {1};
Plane Surface(51) = {2};
Plane Surface(52) = {3};
Plane Surface(53) = {4};
// Place crack in surface
// Line{12} In Surface {51};
// Combine planes in loop
Surface Loop(100) = {50,51,52};
// Convert surface to volume
//Volume(1000) = {50};
