// 2D version of Shear Test

Mesh.Algorithm = 8; // Delaunay for quads

// Define points on surface
ms = 0.05;
ms_l = 0.05;
ms_r = 0.05;

// Dimensions in x y
x_d = 15;
y_d = 1.5;
// Length of crack
c_d = 5;

// Points for plane defined counter-clockwise
Point(1) = {0, 0, 0, ms_l};
Point(2) = {c_d, 0, 0, ms};
Point(3) = {x_d, 0, 0, ms_r};
Point(4) = {x_d, y_d, 0, ms_r};
Point(5) = {c_d, y_d, 0, ms};
Point(6) = {0, y_d, 0, ms_l};
Point(7) = {0, y_d/2+0.000001, 0, ms_l};
Point(8) = {c_d, y_d/2, 0, ms};
Point(9) = {0, y_d/2-0.000001, 0, ms_l};

// Perimeter lines
Line(1) = {1,2}; Line(2) = {2,3};
Line(3) = {3,4}; Line(4) = {4,5};
Line(5) = {5,6}; Line(6) = {6,7};
// Lines making up crack edge
Line(7) = {7,8};
Line(8) = {8,9};
//
Line(9) = {9,1};

// Join lines together to form a surface
Line Loop(1) = {1,2,3,4,5,6,7,8,9};
// Convert line loop to surface
Plane Surface(50) = {1};
// Place crack in surface
// Line{8} In Surface {50};
// Convert surface to volume
// Volume(1000) = {50};
