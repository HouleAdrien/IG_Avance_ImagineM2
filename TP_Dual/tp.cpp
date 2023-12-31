// -------------------------------------------
// gMini : a minimal OpenGL/GLUT application
// for 3D graphics.
// Copyright (C) 2006-2008 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

// -------------------------------------------
// Disclaimer: this code is dirty in the
// meaning that there is no attention paid to
// proper class attribute access, memory
// management or optimisation of any kind. It
// is designed for quick-and-dirty testing
// purpose.
// -------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
#include "src/jmkdtree.h"




std::vector< Vec3 > positions;
std::vector< Vec3 > normals;

std::vector< Vec3 > positions2;
std::vector< Vec3 > normals2;


// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 640;
static unsigned int SCREENHEIGHT = 480;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static bool fullScreen = false;

// ------------------------------------------------------------------------------------------------------------
// i/o and some stuff
// ------------------------------------------------------------------------------------------------------------
void loadPN (const std::string & filename , std::vector< Vec3 > & o_positions , std::vector< Vec3 > & o_normals ) {
    unsigned int surfelSize = 6;
    FILE * in = fopen (filename.c_str (), "rb");
    if (in == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    size_t READ_BUFFER_SIZE = 1000; // for example...
    float * pn = new float[surfelSize*READ_BUFFER_SIZE];
    o_positions.clear ();
    o_normals.clear ();
    while (!feof (in)) {
        unsigned numOfPoints = fread (pn, 4, surfelSize*READ_BUFFER_SIZE, in);
        for (unsigned int i = 0; i < numOfPoints; i += surfelSize) {
            o_positions.push_back (Vec3 (pn[i], pn[i+1], pn[i+2]));
            o_normals.push_back (Vec3 (pn[i+3], pn[i+4], pn[i+5]));
        }

        if (numOfPoints < surfelSize*READ_BUFFER_SIZE) break;
    }
    fclose (in);
    delete [] pn;
}
void savePN (const std::string & filename , std::vector< Vec3 > const & o_positions , std::vector< Vec3 > const & o_normals ) {
    if ( o_positions.size() != o_normals.size() ) {
        std::cout << "The pointset you are trying to save does not contain the same number of points and normals." << std::endl;
        return;
    }
    FILE * outfile = fopen (filename.c_str (), "wb");
    if (outfile == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    for(unsigned int pIt = 0 ; pIt < o_positions.size() ; ++pIt) {
        fwrite (&(o_positions[pIt]) , sizeof(float), 3, outfile);
        fwrite (&(o_normals[pIt]) , sizeof(float), 3, outfile);
    }
    fclose (outfile);
}
void scaleAndCenter( std::vector< Vec3 > & io_positions ) {
    Vec3 bboxMin( FLT_MAX , FLT_MAX , FLT_MAX );
    Vec3 bboxMax( FLT_MIN , FLT_MIN , FLT_MIN );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        for( unsigned int coord = 0 ; coord < 3 ; ++coord ) {
            bboxMin[coord] = std::min<float>( bboxMin[coord] , io_positions[pIt][coord] );
            bboxMax[coord] = std::max<float>( bboxMax[coord] , io_positions[pIt][coord] );
        }
    }
    Vec3 bboxCenter = (bboxMin + bboxMax) / 2.f;
    float bboxLongestAxis = std::max<float>( bboxMax[0]-bboxMin[0] , std::max<float>( bboxMax[1]-bboxMin[1] , bboxMax[2]-bboxMin[2] ) );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = (io_positions[pIt] - bboxCenter) / bboxLongestAxis;
    }
}

void applyRandomRigidTransformation( std::vector< Vec3 > & io_positions , std::vector< Vec3 > & io_normals ) {
    srand(time(NULL));
    Mat3 R = Mat3::RandRotation();
    Vec3 t = Vec3::Rand(1.f);
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = R * io_positions[pIt] + t;
        io_normals[pIt] = R * io_normals[pIt];
    }
}

void subsample( std::vector< Vec3 > & i_positions , std::vector< Vec3 > & i_normals , float minimumAmount = 0.1f , float maximumAmount = 0.2f ) {
    std::vector< Vec3 > newPos , newNormals;
    std::vector< unsigned int > indices(i_positions.size());
    for( unsigned int i = 0 ; i < indices.size() ; ++i ) indices[i] = i;
    srand(time(NULL));
    std::random_shuffle(indices.begin() , indices.end());
    unsigned int newSize = indices.size() * (minimumAmount + (maximumAmount-minimumAmount)*(float)(rand()) / (float)(RAND_MAX));
    newPos.resize( newSize );
    newNormals.resize( newSize );
    for( unsigned int i = 0 ; i < newPos.size() ; ++i ) {
        newPos[i] = i_positions[ indices[i] ];
        newNormals[i] = i_normals[ indices[i] ];
    }
    i_positions = newPos;
    i_normals = newNormals;
}

bool save( const std::string & filename , std::vector< Vec3 > & vertices , std::vector< unsigned int > & triangles ) {
    std::ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        std::cout << filename << " cannot be opened" << std::endl;
        return false;
    }

    myfile << "OFF" << std::endl;

    unsigned int n_vertices = vertices.size() , n_triangles = triangles.size()/3;
    myfile << n_vertices << " " << n_triangles << " 0" << std::endl;

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        myfile << vertices[v][0] << " " << vertices[v][1] << " " << vertices[v][2] << std::endl;
    }
    for( unsigned int f = 0 ; f < n_triangles ; ++f ) {
        myfile << 3 << " " << triangles[3*f] << " " << triangles[3*f+1] << " " << triangles[3*f+2];
        myfile << std::endl;
    }
    myfile.close();
    return true;
}

// ------------------------------------------------------------------------------------------------------------
// rendering.
// ------------------------------------------------------------------------------------------------------------

void initLight () {
    GLfloat light_position1[4] = {22.0f, 16.0f, 50.0f, 0.0f};
    GLfloat direction1[3] = {-52.0f,-16.0f,-50.0f};
    GLfloat color1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat ambient[4] = {0.3f, 0.3f, 0.3f, 0.5f};

    glLightfv (GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv (GL_LIGHT1, GL_SPOT_DIRECTION, direction1);
    glLightfv (GL_LIGHT1, GL_DIFFUSE, color1);
    glLightfv (GL_LIGHT1, GL_SPECULAR, color1);
    glLightModelfv (GL_LIGHT_MODEL_AMBIENT, ambient);
    glEnable (GL_LIGHT1);
    glEnable (GL_LIGHTING);
}

void init () {
    camera.resize (SCREENWIDTH, SCREENHEIGHT);
    initLight ();
    glCullFace (GL_BACK);
    glEnable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glClearColor (0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
}


void drawTriangleMesh(const std::vector<Vec3> &i_positions, const std::vector<unsigned int> &i_triangles) {
    glBegin(GL_TRIANGLES);
    for (unsigned int tIt = 0; tIt < i_triangles.size(); tIt += 3) {
        Vec3 p0 = i_positions[i_triangles[tIt]];
        Vec3 p1 = i_positions[i_triangles[tIt + 1]];
        Vec3 p2 = i_positions[i_triangles[tIt + 2]];

        Vec3 u = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
        Vec3 v = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        Vec3 n = {
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]
        };
        float length = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        n = {n[0] / length, n[1] / length, n[2] / length};

        glNormal3f(n[0], n[1], n[2]);
        glVertex3f(p0[0], p0[1], p0[2]);
        glVertex3f(p1[0], p1[1], p1[2]);
        glVertex3f(p2[0], p2[1], p2[2]);
    }
    glEnd();
}


void drawPointSet( std::vector< Vec3 > const & i_positions , std::vector< Vec3 > const & i_normals ) {
    glBegin(GL_POINTS);
    for(unsigned int pIt = 0 ; pIt < i_positions.size() ; ++pIt) {
        glNormal3f( i_normals[pIt][0] , i_normals[pIt][1] , i_normals[pIt][2] );
        glVertex3f( i_positions[pIt][0] , i_positions[pIt][1] , i_positions[pIt][2] );
    }
    glEnd();
}

std::vector<Vec3> global_positions;
std::vector<unsigned int> global_triangles;

void draw () {
    glPointSize(2); // for example...

 

  glColor3f(1,0.5,0.5);
 //drawPointSet(positions2 , normals2);
  drawTriangleMesh(global_positions ,global_triangles);
}

void display () {
    glLoadIdentity ();
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply ();
    draw ();
    glFlush ();
    glutSwapBuffers ();
}

void idle () {
    glutPostRedisplay ();
}

void key (unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen == true) {
            glutReshapeWindow (SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        } else {
            glutFullScreen ();
            fullScreen = true;
        }
        break;

    case 'w':
        GLint polygonMode[2];
        glGetIntegerv(GL_POLYGON_MODE, polygonMode);
        if(polygonMode[0] != GL_FILL)
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        else
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        break;

    default:
        break;
    }
    idle ();
}

void mouse (int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    } else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate (x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        } else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        } else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle ();
}

void motion (int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate (x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move ((x-lastX)/static_cast<float>(SCREENWIDTH), (lastY-y)/static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom (float (y-lastZoom)/SCREENHEIGHT);
        lastZoom = y;
    }
}


void reshape(int w, int h) {
    camera.resize (w, h);
}

 


float singularWeight(float radius, float distance, int p){
  const float EPSILON =1e-6f; //une petite valeur pour éviter la division par 0 sans trop changer les résultats des calculs ultérieurs
  if(distance < EPSILON) distance = EPSILON;
  return pow(radius/distance,p);
}

float gaussWeight(float radius, float distance){
    return exp(-(distance*distance)/(radius*radius));
}

float wendLand(float radius,float distance){
    if(distance < radius){
    	float ratio = distance/radius;
    	return pow(1-ratio,4)*(4*ratio+1);
    }
    return 0;
}

Vec3 projectOnPlane(Vec3 point_to_project, Vec3 centroid, Vec3 normal){
    Vec3 dc = point_to_project - centroid;
    float dot = Vec3::dot(dc,normal);
    Vec3 result = point_to_project - (dot * normal);
    return result;
}

void HPSS(Vec3 inputPoint, 
    Vec3 & outputPoint, Vec3 & outputNormal, 
    std::vector<Vec3> const & positions, std::vector<Vec3> const & normals, BasicANNkdTree const & kdtree,
    int kernel_type, float radius, unsigned int nbIterations = 10, unsigned int knn = 20)
{
    for(unsigned int i=0; i<nbIterations;i++){
    
    	ANNidxArray nearest_ids = new ANNidx[knn];
        ANNdistArray nearest_distances = new ANNdist[knn];
        kdtree.knearest(inputPoint,knn,nearest_ids,nearest_distances);

        Vec3 centroid = Vec3(0,0,0);
        Vec3 normalCent = Vec3(0,0,0);
        float count = 0;
        

        for(unsigned int j=0 ; j< knn ; j++){
            int id = nearest_ids[j];
            float weight = kernel_type == 0 ? singularWeight(radius,nearest_distances[j],2):
            		    kernel_type == 1 ? gaussWeight(radius,nearest_distances[j]):
            		    			wendLand(radius,nearest_distances[j]);
            
            centroid += weight * positions[id];
            normalCent += weight * normals[id];
            
            count += weight;
        }

        centroid = centroid / count;
        outputNormal = normalCent /count;

        Vec3 proj = projectOnPlane(inputPoint,centroid,outputNormal);
        inputPoint = proj;
        outputPoint = proj;

        delete[] nearest_ids;
        delete[] nearest_distances;
    }
        
}


struct VoxelPoint{
	Vec3 position;
	float sp;


    bool operator==(const VoxelPoint& other) const {
        return position[0] == other.position[0] &&
               position[1] == other.position[1] &&
               position[2] == other.position[2] &&
               sp == other.sp;
    }
};

std::vector<VoxelPoint> voxelGrid;
int resolutionX = 64;
int resolutionY = 64;
int resolutionZ = 64;



std::vector<Vec3> GenerateVoxelGrid(std::vector<Vec3> positionsModel){

	float maxFloat = std::numeric_limits<float>::max();
	float minFloat = std::numeric_limits<float>::min();

	Vec3 BBmax = Vec3(minFloat,minFloat,minFloat);
	Vec3 BBmin = Vec3(maxFloat,maxFloat,maxFloat);

	// on parcours tout les points pour récupérer les min max et faire une grid proche du model
	for(Vec3 pos : positionsModel){
		BBmin[0] = std::min(BBmin[0] ,pos[0]);
		BBmin[1] = std::min(BBmin[1],pos[1]);
		BBmin[2] = std::min(BBmin[2] ,pos[2]);

		BBmax[0] = std::max(BBmax[0] ,pos[0]);
		BBmax[1] = std::max(BBmax[1] ,pos[1]);
		BBmax[2] = std::max(BBmax[2] ,pos[2]);
	}
    
    // on s'assure de couvrir tout le model
	float offset = 0.1;

	BBmax = Vec3(BBmax[0] +offset,BBmax[1]+offset, BBmax[2]+offset);
	BBmin = Vec3(BBmin[0] -offset,BBmin[1]-offset, BBmin[2]-offset);



	std::vector<Vec3> pos;

	for(int i=0; i< resolutionX; i++ ){
		for(int j=0 ;j< resolutionY; j++ ){
			for(int k=0 ;k< resolutionZ; k++ ){

				VoxelPoint point =  VoxelPoint();

				point.position[0] = BBmin[0]  + ((BBmax[0] - BBmin[0])/ (resolutionX-1)) *i;
				point.position[1] = BBmin[1]  +((BBmax[1] - BBmin[1]) / (resolutionY-1)) *j;
				point.position[2] = BBmin[2]  + ((BBmax[2] - BBmin[2]) / (resolutionZ-1)) *k;

				pos.push_back(Vec3(point.position[0],point.position[1],point.position[2]));

				voxelGrid.push_back(point);
			}
		}
	}
    return pos;
}

void  EvaluateGridValues(std::vector<Vec3> positions,std::vector<Vec3> normals,BasicANNkdTree const & kdtree){


	 for(VoxelPoint& point : voxelGrid){
          
            Vec3 outputPoint;
            Vec3 outputNormal;

            HPSS( point.position , outputPoint , outputNormal ,  positions , normals , kdtree ,0, 10, 10);

            // on fait la direction entre le point projeté et celui d'origine
            Vec3 direction = point.position  - outputPoint ;
            //on fait le scalaire a la normal
            //si > 0 on est en dehors de la forme, = 0 on est sûr la forme, < 0 on est dans la forme
            float scal = Vec3::dot(direction,outputNormal);
           
            point.sp = scal;
           
        }
	
}

struct VoxelCell{
	VoxelPoint points[8];
	Vec3 center;
    unsigned int centerIndex;
};

std::vector<VoxelCell> voxelGridCells;

void ProcessVoxelCells(){
    voxelGridCells.resize(resolutionX * resolutionY * resolutionZ);

    for(int i = 0; i < resolutionX - 1; i++)
    {
        for(int j = 0; j < resolutionY - 1; j++)
        {
            for(int k = 0; k < resolutionZ - 1; k++)
            {
                VoxelPoint points[8];
                
                int idx = i + (j * resolutionX) + (k * resolutionX * resolutionY);

                // Récupération des points existants et création de nouveaux points
                points[0] = voxelGrid[idx];
                points[1] = voxelGrid[idx + 1];
                points[2] = voxelGrid[idx + resolutionX];
                points[3] = voxelGrid[idx + 1 + resolutionX];
                points[4] = voxelGrid[idx + resolutionX * resolutionY];
                points[5] = voxelGrid[idx + 1 + resolutionX * resolutionY];
                points[6] = voxelGrid[idx + resolutionX + resolutionX * resolutionY];
                points[7] = voxelGrid[idx + 1 + resolutionX + resolutionX * resolutionY];
                
                // Calcul du centre
                Vec3 center(0, 0, 0);
                for(int l = 0; l < 8; l++)  
                {
                    center[0] += points[l].position[0];
                    center[1] += points[l].position[1];
                    center[2] += points[l].position[2];
                }
                center[0] /= 8; center[1] /= 8; center[2] /= 8;
                
                // Création et enregistrement de la cellule
                VoxelCell cell;
                for(int m = 0; m < 8; m++) 
                {
                    cell.points[m] = points[m];
                }
                cell.center = center;
                
                voxelGridCells[idx] = cell;
            }
        }
    }
}


//Cette fonction permet de récupérer les 2 points de l'arrete au centre de mes 4 cellules

void ProcessVoxelEdges(int direction)
{
    int i_limit = direction == 0 ? resolutionX - 3 : resolutionX;
    int j_limit = direction == 1 ? resolutionY - 3 : resolutionY;
    int k_limit = direction == 2 ? resolutionZ - 3 : resolutionZ;

    for (int i = 0; i < i_limit; i++)
    {
        for (int j = 0; j < j_limit; j++)
        {
            for (int k = 0; k < k_limit; k++)
            {
                std::vector<VoxelCell> voxel_4;

                for (int offset = 0; offset < 4; offset++)
                {
                    int x_index = direction == 0 ? i + offset : i;
                    int y_index = direction == 1 ? j + offset : j;
                    int z_index = direction == 2 ? k + offset : k;

                    VoxelCell cell = voxelGridCells[x_index + (y_index * resolutionY) + (z_index * resolutionY * resolutionZ)];
                    voxel_4.push_back(cell);
                }

                int commonIndices[2];
               if(direction == 0) { commonIndices[0] = 1; commonIndices[1] = 0; }
                else if(direction == 1) { commonIndices[0] = 0; commonIndices[1] = 2; }
                else if(direction == 2) { commonIndices[0] = 0; commonIndices[1] = 4; }



                VoxelPoint commonPoints[2];

                commonPoints[0] = voxel_4[0].points[commonIndices[0]];
                commonPoints[1] = voxel_4[0].points[commonIndices[1]];

                bool isEdge = false;
                bool horaire = false;
                if(commonPoints[0].sp > 0 && commonPoints[1].sp < 0)
                {
                    
                    horaire = false;
                    isEdge = true;
                }

                if(commonPoints[0].sp < 0 && commonPoints[1].sp > 0) 
                {
                    horaire = true;
                    isEdge = true;
                }

                // Si c'est un bord, ajouter les centres des cellules et créer des triangles
                if(isEdge) {
                    int indices[4]; 
                    for(int c = 0; c < 4; ++c) {
                        global_positions.push_back(voxel_4[c].center); 
                        voxel_4[c].centerIndex = global_positions.size();
                    }

                    if(!horaire){
                        indices[0] = voxel_4[0].centerIndex;
                        indices[1] = voxel_4[1].centerIndex;
                        indices[2] = voxel_4[2].centerIndex;
                        indices[3] = voxel_4[3].centerIndex;
                    }else{
                        indices[0] = voxel_4[1].centerIndex;
                        indices[1] = voxel_4[0].centerIndex;
                        indices[2] = voxel_4[3].centerIndex;
                        indices[3] = voxel_4[2].centerIndex;
                    }
                    
                    global_triangles.push_back(indices[0]);
                    global_triangles.push_back(indices[1]);
                    global_triangles.push_back(indices[2]);

                    global_triangles.push_back(indices[0]);
                    global_triangles.push_back(indices[2]);
                    global_triangles.push_back(indices[3]);
                }
            }
        }
    }
}


int main (int argc, char ** argv) {
    if (argc > 2) {
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("tp point processing");

    init ();
    glutIdleFunc (idle);
    glutDisplayFunc (display);
    glutKeyboardFunc (key);
    glutReshapeFunc (reshape);
    glutMotionFunc (motion);
    glutMouseFunc (mouse);
    key ('?', 0, 0);


    {
        // Load a first pointset, and build a kd-tree:
        loadPN("pointsets/igea.pn" , positions , normals);

        BasicANNkdTree kdtree;
        kdtree.build(positions);

        // Create a second pointset that is artificial, and project it on pointset1 using MLS techniques:
        const unsigned int pointsetsize = 20000;
        positions2.resize( pointsetsize );
        normals2.resize(positions2.size());


        // PROJECT USING MLS (HPSS and APSS):

        
        /*for( unsigned int pIt = 0 ; pIt < positions2.size() ; ++pIt ) {
            // Ajout de bruit dans la direction de la normale initiale (avant la projection):
            double alpha = 2; 
            double noise = alpha*(2.0 * ((double)rand()/RAND_MAX)-1.0);
            
            positions2[pIt] += noise * normals[pIt];

            Vec3 outputPoint;
            Vec3 outputNormal;

            HPSS( positions2[pIt] , outputPoint , outputNormal ,   positions , normals , kdtree ,1, 10, 10);

            positions2[pIt] =outputPoint;
            normals2[pIt] =outputNormal;
        }*/
        positions2.resize( resolutionX*resolutionY*resolutionZ );
        normals2.resize( resolutionX*resolutionY*resolutionZ );
  
  // step 1
       positions2 = GenerateVoxelGrid(positions);
       // GenerateVoxelGrid(positions);
   //step 2
       EvaluateGridValues(positions,normals,kdtree);
      
 //step 3

    ProcessVoxelCells();
   ProcessVoxelEdges(0);
   ProcessVoxelEdges(1);
    ProcessVoxelEdges(2);

    }



    glutMainLoop ();
    return EXIT_SUCCESS;
}

