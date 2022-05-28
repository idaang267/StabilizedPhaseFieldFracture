// 3D Strip with Notch Problem
//
// Usage to convert for FEniCS
// geo name.geo -format msh2 -3
// dolfin-convert name.msh name.xml

// mesh size
hr = 0.02;	// More refined
h  = 0.04;

// Set width, height, and thickness
W  = 0.5; H = .75; T =-4*hr;
H0 = 15*hr; H1 = 2*hr; W0 = H1;

// Upper surface Points
// Define outer points anti-clockwise
Point(101) = {-W, -H,  T, h}; Point(102) = {W, -H,  T, h};
Point(103) = { W, -H0, T, hr}; Point(104) = {W,  H0, T, hr};
Point(105) = { W,  H,  T, h}; Point(106) = {-W,  H,  T, h};
// Define Inner Points LHS of box
Point(107) = {-W0,  H0, T, hr}; Point(108) = {-W0, -H0, T, hr};
// Two points on outer surface
Point(109) = {-W,  H1, T, h}; Point(110) = {-W, -H1, T, h};
// Define Points for semi-circle
Point(111) = {-W0,-H1, T, hr};
Point(112) = {-W0, H1, T, hr};
Point(113) = {-W0, 0., T, hr};

// Outer lines
Line(101) = {101, 102}; Line(102) = {102, 103};
Line(103) = {103, 104}; Line(104) = {104, 105};
Line(105) = {105, 106}; Line(106) = {106, 109};
// Inner lines
Line(107) = {107, 112}; Line(108) = {109, 112};
Circle(109)={111,113,112};
Line(110) = {111,110}; Line(111) = {111, 108};
// Outer line
Line(112) = {110, 101};
// Top and bottom of box
Line(113) = {108, 103}; Line(114) = {107, 104};

// Bottom, Middle, and top section
Line Loop(11) = {101, 102, -113, -111,  110, 112};
Line Loop(12) = {-113, -111, 109, -107, 114, -103};
Line Loop(13) = {104, 105, 106, 108, -107, 114};

// Convert to plane surface
Plane Surface(11) = {11};
Plane Surface(12) = {12};
Plane Surface(13) = {13};

//+
Extrude {0, 0, -2*T} {
	Surface{11};
	Surface{12};
	Surface{13};
}

// 1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal, 7=BAMG, 8=DelQuad
//Mesh.Algorithm = 6;
//Mesh 3;
