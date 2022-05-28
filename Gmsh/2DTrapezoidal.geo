// 2D Version of "Trapezoidal" mesh
//
// Usage to convert for FEniCS
// geo name.geo -format msh2 -3
// dolfin-convert name.msh name.xml

Mesh.Algorithm = 8; // Delaunay for quads

// Define points on surface
ms_r = 0.02; // More refined and defines hsize
ms = 0.4;    // Less refined

// Dimensions of middle box
// x_d > x_m > c_d
x_d = 8; x_m = 3; c_d = 2;
// y_d > y_m > c_d
y_d = 9.5; y_m = 0.3;
// Radius r_d > x_d
r_d = 5;
y_r = (r_d^2 - (x_d/2)^2)^(1/2);
y_c = r_d - y_r;

// Points for inner rectangle
// Bottom line
Point(1) = {-x_d/2, -y_m/2, 0, ms_r};
Point(2) = {-x_m/2, -y_m/2, 0, ms_r};
Point(3) = {x_m/2, -y_m/2, 0, ms_r};
Point(4) = {x_d/2, -y_m/2, 0, ms_r};
// Top line
Point(5) = {x_d/2, y_m/2, 0, ms_r};
Point(6) = {x_m/2, y_m/2, 0, ms_r};
Point(7) = {-x_m/2, y_m/2, 0, ms_r};
Point(8) = {-x_d/2, y_m/2, 0, ms_r};
// Mid-line and crack
Point(9) = {-x_m/2, 0, 0, ms_r};
Point(10) = {-c_d/2, 0, 0, ms_r};
Point(11) = {c_d/2, 0, 0, ms_r};
Point(12) = {x_m/2, 0, 0, ms_r};
// Discrete crack points
Point(13) = {0, -0.00001, 0, ms_r};
Point(14) = {0, +0.00001, 0, ms_r};
// Outer Points
Point(15) = {-x_d/2, -y_d/2, 0, ms};
Point(16) = {x_d/2, -y_d/2, 0, ms};
Point(17) = {x_d/2, y_d/2, 0, ms};
Point(18) = {-x_d/2, y_d/2, 0, ms};
// Arc points
Point(19) = {0, -(y_d/2 - y_c), 0, ms};
Point(20) = {0, (y_d/2 - y_c), 0, ms};
// Center of arc
Point(21) = {0, -(y_d/2 + y_r), 0, ms};
Point(22) = {0, (y_d/2 + y_r), 0, ms};

// Lines for Rectangle
Line(1) = {1,2}; Line(2) = {2,3};
Line(3) = {3,4}; Line(4) = {4,5};
Line(5) = {5,6}; Line(6) = {6,7};
Line(7) = {7,8}; Line(8) = {8,1};
// Lines for inner section
Line(9) = {7,9}; Line(10) = {9,2};
Line(11) = {3,12}; Line(12) = {12,6};
// Lines around crack
Line(13) = {9,10};
Line(14) = {10,13}; Line(15) = {13,11};
Line(16) = {11,12};
Line(17) = {11,14}; Line(18) = {14,10};
// Outer lines
Line(19) = {1,15}; Line(20) = {16,4};
Line(21) = {5,17}; Line(22) = {18,8};
// Arcs
Circle(23) = {15,21,19}; Circle(24) = {19,21,16};
Circle(25) = {17,22,20}; Circle(26) = {20,22,18};

// Join lines together to form a surface
Line Loop(1) = {1,-10,-9,7,8};
Line Loop(2) = {2,11,-16,-15,-14,-13,10};
Line Loop(3) = {9,13,-18,-17,16,12,6};
Line Loop(4) = {-12,-11,3,4,5};
//
Line Loop(5) = {19,23,24,20,-3,-2,-1};
Line Loop(6) = {21,25,26,22,-7,-6,-5};

// Convert line loop to surface
Plane Surface(50) = {1};
Plane Surface(51) = {2};
Plane Surface(52) = {3};
Plane Surface(53) = {4};
Plane Surface(54) = {5};
Plane Surface(55) = {6};
