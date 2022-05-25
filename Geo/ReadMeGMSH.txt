GMSH is a simple meshing tool that allows some flexibility in controlling mesh sizing

Note that the following instructions are for my personal Linux workstation and
have the assumption that atom is your code editor of choice. Bash files can be
opened in the terminal but I find it easier to use a text editor.

Download gmsh from the following website: https://gmsh.info/
Rename to something simple "gmsh-4.6.0"
Copy from your downloads folder to /usr/local/bin/
    Start in the folder that gmsh resides in
    sudo cp -R gmsh-4.6.0 /usr/local/bin/
    Navigate to your base directory by:
      cd
      atom .bashrc
    Add the following line to the bottom of your .bashrc file
      export PATH=$PATH:/usr/local/bin/gmsh-4.6.0/bin/
    Save your .bashrc file, close, and source it by:
      source .bashrc
    Now you can run from anywhere by writing in command line:
      gmsh
    For opening a specific geo file
      gmsh name.geo
      If you find that this command leads to an immediate crash of the
      application, it is likely you have an error in your gmsh file.

Instructions for converting a geo file to msh to xml
  gmsh name.geo -format msh2 -3
  dolfin-convert name.msh name.xml

You can then create a python script to convert your xml file to xdmf to view in
Paraview (Lines Below)
  from dolfin import *
  mesh = Mesh("name.xml")
  XDMFFile("mesh.xdmf").write(mesh)
