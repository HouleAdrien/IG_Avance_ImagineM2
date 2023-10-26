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

#include <algorithm>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
#include <map>
using namespace std;

enum DisplayMode{ WIRE=0, SOLID=1, LIGHTED_WIRE=2, LIGHTED=3 };

struct Triangle {
    inline Triangle () {
        v[0] = v[1] = v[2] = 0;
    }
    inline Triangle (const Triangle & t) {
        v[0] = t.v[0];   v[1] = t.v[1];   v[2] = t.v[2];
    }
    inline Triangle (unsigned int v0, unsigned int v1, unsigned int v2) {
        v[0] = v0;   v[1] = v1;   v[2] = v2;
    }
    unsigned int & operator [] (unsigned int iv) { return v[iv]; }
    unsigned int operator [] (unsigned int iv) const { return v[iv]; }
    inline virtual ~Triangle () {}
    inline Triangle & operator = (const Triangle & t) {
        v[0] = t.v[0];   v[1] = t.v[1];   v[2] = t.v[2];
        return (*this);
    }
    // membres indices des sommets du triangle:
    unsigned int v[3];
};


bool contain(vector<unsigned int> const & i_vector, unsigned int element) {
    for (unsigned int i = 0; i < i_vector.size(); i++) {
        if (i_vector[i] == element) return true;
    }
    return false;
}


void collect_one_ring (vector<Vec3> const & i_vertices,
                       vector< Triangle > const & i_triangles,
                       vector<vector<unsigned int> > & o_one_ring) {
    o_one_ring.clear();
    o_one_ring.resize(i_vertices.size()); //one-ring of each vertex, i.e. a list of vertices with which it shares an edge
    //Parcourir les triangles et ajouter les voisins dans le 1-voisinage
    //Attention verifier que l'indice n'est pas deja present
    for (unsigned int i = 0; i < i_triangles.size(); i++) {
        //Tous les points opposés dans le triangle sont reliés
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                if (j != k) {
                    if (!contain(o_one_ring[i_triangles[i][j]], i_triangles[i][k])) {
                        o_one_ring[i_triangles[i][j]].push_back(i_triangles[i][k]);
                    }
                }
            }
        }
    }
}

struct Mesh {
    vector< Vec3 > vertices; //array of mesh vertices positions
    vector< Vec3 > normals; //array of vertices normals useful for the display
    vector< Triangle > triangles; //array of mesh triangles
    vector< Vec3 > triangle_normals; //triangle normals to display face normals

    vector< Vec3 > vunicurvature;
    float minunicurvature;
    float maxunicurvature;

    map<int,float> tshape;
    map<int, map<int,double>> cotangentWeights;
   
    vector< Vec3 > vmeancurvature;
    float minmeancurvature;
    float maxmeancurvature;

    //Compute face normals for the display
    void computeTrianglesNormals(){

        //A faire : implémenter le calcul des normales par face
        //Attention commencer la fonction par triangle_normals.clear();
        //Iterer sur les triangles

        //La normal du triangle i est le resultat du produit vectoriel de deux ses arêtes e_10 et e_20 normalisé (e_10^e_20)
        //L'arete e_10 est représentée par le vecteur partant du sommet 0 (triangles[i][0]) au sommet 1 (triangles[i][1])
        //L'arete e_20 est représentée par le vecteur partant du sommet 0 (triangles[i][0]) au sommet 2 (triangles[i][2])

        //Normaliser et ajouter dans triangle_normales

        triangle_normals.clear();
        for( unsigned int i = 0 ; i < triangles.size() ;i++ ){
            const Vec3 & e0 = vertices[triangles[i][1]] - vertices[triangles[i][0]];
            const Vec3 & e1 = vertices[triangles[i][2]] - vertices[triangles[i][0]];
            Vec3 n = Vec3::cross( e0, e1 );
            n.normalize();
            triangle_normals.push_back( n );
        }
    }

    //Compute vertices normals as the average of its incident faces normals
    void computeVerticesNormals(  ){
        //Utiliser weight_type : 0 uniforme, 1 aire des triangles, 2 angle du triangle

        //A faire : implémenter le calcul des normales par sommet comme la moyenne des normales des triangles incidents
        //Attention commencer la fonction par normals.clear();
        //Initializer le vecteur normals taille vertices.size() avec Vec3(0., 0., 0.)
        //Iterer sur les triangles

        //Pour chaque triangle i
        //Ajouter la normal au triangle à celle de chacun des sommets en utilisant des poids
        //0 uniforme, 1 aire du triangle, 2 angle du triangle

        //Iterer sur les normales et les normaliser
        normals.clear();
        normals.resize( vertices.size(), Vec3(0., 0., 0.) );
        for( unsigned int i = 0 ; i < triangles.size() ;i++ ){
            for( unsigned int t = 0 ; t < 3 ; t++ )
                normals[ triangles[i][t] ] += triangle_normals[i];
        }
        for( unsigned int i = 0 ; i < vertices.size() ;i++ )
            normals[ i ].normalize();


    }
 
    void calc_uniform_mean_curvature(){
        vunicurvature.resize(vertices.size());
        vector<vector<unsigned int>> o_one_ring;
        collect_one_ring(vertices,triangles,o_one_ring);
        minunicurvature = FLT_MAX;
        maxunicurvature = FLT_MIN;

        for(int i=0; i < vertices.size() ; i++ )
        {
            vector<unsigned int> neighbors = o_one_ring.at(i);
            int n = neighbors.size();
            Vec3 v = vertices.at(i);
            Vec3 sum = Vec3(0,0,0);

            for(unsigned int j = 0 ; j < neighbors.size() ; j++)
            {
                sum += vertices.at(neighbors.at(j));
            }

            Vec3 curvature = ((1/float(n)) * sum) - v;
            vunicurvature.at(i) = curvature;
            minunicurvature = min(curvature.length()/2, minunicurvature ) ;
            maxunicurvature = max(curvature.length()/2, maxunicurvature ) ;
        }
    }

 
    int findOtherTriangle(unsigned int i, unsigned int j, unsigned int triangleId)
    {
        pair<unsigned int,unsigned int> p = make_pair(i,j);
        for(unsigned int t = 0; t < triangles.size();t++)
        {
            if(t == triangleId){continue;}

            int v1 = triangles.at(t)[0];
            int v2 = triangles.at(t)[1];
            int v3 = triangles.at(t)[2];

            pair<unsigned int,unsigned int> p1 = make_pair(v1,v2);
            pair<unsigned int,unsigned int> p2 = make_pair(v2,v3);
            pair<unsigned int,unsigned int> p3 = make_pair(v3,v1);
            pair<unsigned int,unsigned int> p4 = make_pair(v2,v1);
            pair<unsigned int,unsigned int> p5 = make_pair(v3,v2);
            pair<unsigned int,unsigned int> p6 = make_pair(v1,v3);


            if(p1 == p || p2 == p || p3 == p || p4 == p || p5 == p || p6 == p)
            {
                return t;
            }
        }
        return -1;
    }

   double calc_weight(unsigned int i, unsigned int j, int v3, unsigned int t1)
    {
        int t2 = findOtherTriangle(i,j,t1);
        if(t2 < 0){cout << "Didnt find another triangle for " << i << " " << j << endl; return 0;}

        int v4 = i;
        for(int vi = 0; vi < 3; vi++)
        {
            v4 = triangles.at(t2)[vi];
            if(v4 != i && v4 != j && v4 != v3){break;}
        }

        Vec3 alphai = vertices.at(i) - vertices.at(v3); 
        alphai.normalize();
        Vec3 alphaj = vertices.at(j) - vertices.at(v3);
        alphaj.normalize();
        double alpha = Vec3::dot(alphai,alphaj);

        Vec3 betai = vertices.at(i) - vertices.at(v4); 
        betai.normalize();
        Vec3 betaj = vertices.at(j) - vertices.at(v4);
         betaj.normalize();
        double beta = Vec3::dot(betai,betaj);

        double cotaplha = sin(alpha) / cos(alpha);
        double cotbeta = sin(beta) / cos(beta);

        return 0.5 * (cotbeta + cotaplha);
    }

    void calc_weights()
    {
        for(unsigned int t = 0; t < triangles.size();t++)
        {
            int v1 = triangles.at(t)[0];
            int v2 = triangles.at(t)[1];
            int v3 = triangles.at(t)[2];

            if(cotangentWeights[v1].count(v2) == 0)
            {
                cotangentWeights[v1][v2] = calc_weight(v1,v2,v3,t);
            }
            if(cotangentWeights[v2].count(v3)   == 0)
            {
                cotangentWeights[v2][v3] = calc_weight(v2,v3,v1,t);
            }
            if(cotangentWeights[v3].count(v1) == 0)
            {
                cotangentWeights[v3][v1] = calc_weight(v3,v1,v2,t);
            }
        }
    }




    void calc_mean_curvature() {
        vmeancurvature.resize(vertices.size());
        vector<vector<unsigned int>> o_one_ring;
        collect_one_ring(vertices,triangles,o_one_ring);
        minmeancurvature = FLT_MAX;
        maxmeancurvature = FLT_MIN;

        for (unsigned int i = 0; i < vertices.size(); i++) {
            vector<unsigned int> neighbors = o_one_ring.at(i);
    		int n = neighbors.size();
    		Vec3 v = vertices.at(i);

            Vec3 s = Vec3(0,0,0);
    		for(unsigned int vi :neighbors)
    		{
    			s += cotangentWeights[i][vi] * (vertices.at(vi) - v);
    		}

            vmeancurvature[i] = s;
            minmeancurvature = min(vmeancurvature[i].length()/2,minmeancurvature);
            maxmeancurvature = max(vmeancurvature[i].length()/2,maxmeancurvature);
        }
    }


    void uniform_smooth(unsigned int num_iter)
    {
        for(unsigned int iter = 0; iter < num_iter; iter++)
        {
            vector<Vec3> newVertices(vertices.size(),Vec3(0,0,0));

            for(unsigned int i=0; i< vertices.size() ;i++)
            {
                newVertices.at(i) = vertices.at(i) + (0.5f* vunicurvature.at(i) );
            }
            
            vertices = newVertices;
            calc_uniform_mean_curvature();
        }
        
        calc_triangle_quality();
        normalizeTriangleQuality();
        computeNormals();
    }

    void taubinSmooth(unsigned int num_iter, float lambda , float mu){
        for(unsigned int iter = 0; iter < num_iter; iter++)
        {
            
            vector<Vec3> newVertices(vertices.size(),Vec3(0,0,0));

            for(unsigned int i=0; i< vertices.size() ;i++)
            {
                newVertices.at(i) = vertices.at(i) + (lambda* vunicurvature.at(i) );
            }
             vertices = newVertices;
            calc_uniform_mean_curvature();

            vector<Vec3> newVertices2(newVertices.size(),Vec3(0,0,0));

            for(unsigned int i=0; i< newVertices.size() ;i++)
            {
                newVertices2.at(i) = newVertices.at(i) + (mu* vunicurvature.at(i) );
            }
                
            vertices = newVertices2;
            calc_uniform_mean_curvature();
        }

        calc_triangle_quality();
        normalizeTriangleQuality();
        computeNormals();
    }

    void computeNormals(){
        computeTrianglesNormals();
        computeVerticesNormals();
    }

    void addNoise(){
        for( unsigned int i = 0 ; i < vertices.size() ; i ++ )
        {
            float factor = 0.03;
            const Vec3 & p = vertices[i];
            const Vec3 & n = normals[i];
            
            vertices[i] = Vec3( p[0] + factor*((double)(rand()) / (double)
            (RAND_MAX))*n[0], p[1] + factor*((double)(rand()) / (double)
            (RAND_MAX))*n[1], p[2] + factor*((double)(rand()) / (double)
            (RAND_MAX))*n[2]);
        }
    }


  void calc_triangle_quality() {
        tshape.clear(); 

        const float quality_threshold = 1e-6;

        for (unsigned int i = 0; i < triangles.size(); i++) {
            unsigned int v0 = triangles[i][0];
            unsigned int v1 = triangles[i][1];
            unsigned int v2 = triangles[i][2];

            const Vec3 &vertex0 = vertices[v0];
            const Vec3 &vertex1 = vertices[v1];
            const Vec3 &vertex2 = vertices[v2];

            float edge0 = (vertex0 - vertex1).length();
            float edge1 = (vertex1 - vertex2).length();
            float edge2 = (vertex2 - vertex0).length();

            float perimeter = edge0 + edge1 + edge2;

            if (perimeter < quality_threshold) {
                tshape[i] = numeric_limits<float>::max();
            } else {
                // Calculez l'aire du triangle en utilisant la formule de Héron.
                float s = perimeter * 0.5f;
                float area = sqrt(s * (s - edge0) * (s - edge1) * (s - edge2));

                // Calculez le rayon du cercle circonscrit en utilisant la formule du rayon de la circonférence.
                float circum_radius = (edge0 * edge1 * edge2) / (4 * area);

                float min_edge_length = min({edge0, edge1, edge2});

                // Calculez le rapport entre le rayon du cercle circonscrit et la plus petite longueur d'arête.
                float triangle_quality = circum_radius / min_edge_length;

                tshape[i] = triangle_quality;
            }
        }
    }

    // Pour avoir un affichage propre avec ma fonction scalartoRgb je normalise mes qualités entre 0 et 1
    void normalizeTriangleQuality() {
        float min_quality = numeric_limits<float>::max();
        float max_quality = -numeric_limits<float>::max();

        for (const auto &entry : tshape) {
            float quality = entry.second;
            if (quality < min_quality) {
                min_quality = quality;
            }
            if (quality > max_quality) {
                max_quality = quality;
            }
        }

        if (min_quality == max_quality) {
            return; 
        }

        for (auto &entry : tshape) {
            float quality = entry.second;
            float normalized_quality = (quality - min_quality) / (max_quality - min_quality);
            entry.second = normalized_quality;
        }
    }

 void laplaceBeltramiSmooth(unsigned int num_iter) {
    for (unsigned int iter = 0; iter < num_iter; iter++) {
        calc_weights();

        // Ajouter une étape pour normaliser les poids cotangents
        map<int, map<int, double>> normalizedWeights = normalizeCotangentWeights();

        vector<Vec3> newVertices(vertices.size(), Vec3(0, 0, 0));

        for (unsigned int i = 0; i < vertices.size(); i++) {
            Vec3 laplace = Vec3(0, 0, 0);

            for (unsigned int j = 0; j < vertices.size(); j++) {
                if (i != j) {
                    laplace += normalizedWeights[i][j] * (vertices[j] - vertices[i]);
                }
            }
            newVertices[i] = vertices[i] + laplace;
        }

        vertices = newVertices;
        calc_mean_curvature();
        cout << iter << endl;
    }

    calc_triangle_quality();
    normalizeTriangleQuality();
    computeNormals();
    cout << "finit" << endl;
}


   map<int, map<int, double>> normalizeCotangentWeights() {
    map<int, map<int, double>> normalizedWeights;

    for (unsigned int i = 0; i < vertices.size(); i++) {
        double sum = 0.0;
        for (const auto &entry : cotangentWeights[i]) {
            sum += entry.second;
        }
        for (auto &entry : cotangentWeights[i]) {
            double normalizedWeight = entry.second / sum;
            normalizedWeights[i][entry.first] = normalizedWeight;
        }
    }

    return normalizedWeights;
}



};

//Transformation made of a rotation and translation
struct Transformation {
    Mat3 rotation;
    Vec3 translation;
};




//Input mesh loaded at the launch of the application
Mesh mesh;
vector< float > current_field; //normalized filed of each vertex

bool display_normals;
bool display_smooth_normals;
bool display_mesh;

DisplayMode displayMode;
int weight_type;

// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 1600;
static unsigned int SCREENHEIGHT = 900;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static bool fullScreen = false;

// ------------------------------------
// File I/O
// ------------------------------------
bool saveOFF( const string & filename ,
              vector< Vec3 > const & i_vertices ,
              vector< Vec3 > const & i_normals ,
              vector< Triangle > const & i_triangles,
              vector< Vec3 > const & i_triangle_normals ,
              bool save_normals = false ) {
    ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        cout << filename << " cannot be opened" << endl;
        return false;
    }

    myfile << "OFF" << endl ;

    unsigned int n_vertices = i_vertices.size() , n_triangles = i_triangles.size();
    myfile << n_vertices << " " << n_triangles << " 0" << endl;

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        myfile << i_vertices[v][0] << " " << i_vertices[v][1] << " " << i_vertices[v][2] << " ";
        if (save_normals) myfile << i_normals[v][0] << " " << i_normals[v][1] << " " << i_normals[v][2] << endl;
        else myfile << endl;
    }
    for( unsigned int f = 0 ; f < n_triangles ; ++f ) {
        myfile << 3 << " " << i_triangles[f][0] << " " << i_triangles[f][1] << " " << i_triangles[f][2]<< " ";
        if (save_normals) myfile << i_triangle_normals[f][0] << " " << i_triangle_normals[f][1] << " " << i_triangle_normals[f][2];
        myfile << endl;
    }
    myfile.close();
    return true;
}

void openOFF( string const & filename,
              vector<Vec3> & o_vertices,
              vector<Vec3> & o_normals,
              vector< Triangle > & o_triangles,
              vector< Vec3 > & o_triangle_normals,
              bool load_normals = true )
{
    ifstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open())
    {
        cout << filename << " cannot be opened" << endl;
        return;
    }

    string magic_s;

    myfile >> magic_s;

    if( magic_s != "OFF" )
    {
        cout << magic_s << " != OFF :   We handle ONLY *.off files." << endl;
        myfile.close();
        exit(1);
    }

    int n_vertices , n_faces , dummy_int;
    myfile >> n_vertices >> n_faces >> dummy_int;

    o_vertices.clear();
    o_normals.clear();

    for( int v = 0 ; v < n_vertices ; ++v )
    {
        float x , y , z ;

        myfile >> x >> y >> z ;
        o_vertices.push_back( Vec3( x , y , z ) );

        if( load_normals ) {
            myfile >> x >> y >> z;
            o_normals.push_back( Vec3( x , y , z ) );
        }
    }

    o_triangles.clear();
    o_triangle_normals.clear();
    for( int f = 0 ; f < n_faces ; ++f )
    {
        int n_vertices_on_face;
        myfile >> n_vertices_on_face;

        if( n_vertices_on_face == 3 )
        {
            unsigned int _v1 , _v2 , _v3;
            myfile >> _v1 >> _v2 >> _v3;

            o_triangles.push_back(Triangle( _v1, _v2, _v3 ));

            if( load_normals ) {
                float x , y , z ;
                myfile >> x >> y >> z;
                o_triangle_normals.push_back( Vec3( x , y , z ) );
            }
        }
        else if( n_vertices_on_face == 4 )
        {
            unsigned int _v1 , _v2 , _v3 , _v4;
            myfile >> _v1 >> _v2 >> _v3 >> _v4;

            o_triangles.push_back(Triangle(_v1, _v2, _v3 ));
            o_triangles.push_back(Triangle(_v1, _v3, _v4));
            if( load_normals ) {
                float x , y , z ;
                myfile >> x >> y >> z;
                o_triangle_normals.push_back( Vec3( x , y , z ) );
            }

        }
        else
        {
            cout << "We handle ONLY *.off files with 3 or 4 vertices per face" << endl;
            myfile.close();
            exit(1);
        }
    }

}

// ------------------------------------
// Application initialization
// ------------------------------------
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
    glDisable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glClearColor (0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    // uncomment if you want color to apply by triangles
    //glShadeModel(GL_FLAT);
    display_normals = false;
    display_mesh = true;
    display_smooth_normals = true;
    displayMode = LIGHTED;

}


// ------------------------------------
// Rendering.
// ------------------------------------

void drawVector( Vec3 const & i_from, Vec3 const & i_to ) {

    glBegin(GL_LINES);
    glVertex3f( i_from[0] , i_from[1] , i_from[2] );
    glVertex3f( i_to[0] , i_to[1] , i_to[2] );
    glEnd();
}

void drawAxis( Vec3 const & i_origin, Vec3 const & i_direction ) {

    glLineWidth(4); // for example...
    drawVector(i_origin, i_origin + i_direction);
}

void drawReferenceFrame( Vec3 const & origin, Vec3 const & i, Vec3 const & j, Vec3 const & k ) {

    glDisable(GL_LIGHTING);
    glColor3f( 0.8, 0.2, 0.2 );
    drawAxis( origin, i );
    glColor3f( 0.2, 0.8, 0.2 );
    drawAxis( origin, j );
    glColor3f( 0.2, 0.2, 0.8 );
    drawAxis( origin, k );
    glEnable(GL_LIGHTING);

}


typedef struct {
    float r;       // ∈ [0, 1]
    float g;       // ∈ [0, 1]
    float b;       // ∈ [0, 1]
} RGB;



RGB scalarToRGB( float scalar_value ) //Scalar_value ∈ [0, 1]
{
    RGB rgb;
    float H = scalar_value*360., S = 1., V = 0.85,
            P, Q, T,
            fract;

    (H == 360.)?(H = 0.):(H /= 60.);
    fract = H - floor(H);

    P = V*(1. - S);
    Q = V*(1. - S*fract);
    T = V*(1. - S*(1. - fract));

    if      (0. <= H && H < 1.)
        rgb = (RGB){.r = V, .g = T, .b = P};
    else if (1. <= H && H < 2.)
        rgb = (RGB){.r = Q, .g = V, .b = P};
    else if (2. <= H && H < 3.)
        rgb = (RGB){.r = P, .g = V, .b = T};
    else if (3. <= H && H < 4.)
        rgb = (RGB){.r = P, .g = Q, .b = V};
    else if (4. <= H && H < 5.)
        rgb = (RGB){.r = T, .g = P, .b = V};
    else if (5. <= H && H < 6.)
        rgb = (RGB){.r = V, .g = P, .b = Q};
    else
        rgb = (RGB){.r = 0., .g = 0., .b = 0.};

    return rgb;
}

void drawSmoothTriangleMesh( Mesh const & i_mesh , bool draw_field = false ) {
    glBegin(GL_TRIANGLES);
    for(unsigned int tIt = 0 ; tIt < i_mesh.triangles.size(); ++tIt) {

        for(unsigned int i = 0 ; i < 3 ; i++) {
            const Vec3 & p = i_mesh.vertices[i_mesh.triangles[tIt][i]]; //Vertex position
            const Vec3 & n = i_mesh.normals[i_mesh.triangles[tIt][i]]; //Vertex normal

            if( draw_field && current_field.size() > 0 ){
                RGB color = scalarToRGB( current_field[i_mesh.triangles[tIt][i]] );
                glColor3f( color.r, color.g, color.b );
            }
            glNormal3f( n[0] , n[1] , n[2] );
            glVertex3f( p[0] , p[1] , p[2] );
        }
    }
    glEnd();

}


void drawTriangleMesh(Mesh const &i_mesh, int mode = 1) {
    glBegin(GL_TRIANGLES);
    for (unsigned int tIt = 0; tIt < i_mesh.triangles.size(); ++tIt) {
        const Vec3 &n = i_mesh.triangle_normals[tIt]; // Triangle normal
        for (unsigned int i = 0; i < 3; i++) {
            const Vec3 &p = i_mesh.vertices[i_mesh.triangles[tIt][i]]; // Vertex position

            switch (mode) {
                case 0: {
                    float quality = i_mesh.tshape.at(tIt);
                    RGB color = scalarToRGB(quality);
                    glColor3f(color.r, color.g, color.b);
                    break;
                }
                case 1: {
                    float c = i_mesh.vunicurvature.at(i_mesh.triangles[tIt][i]).length() / 2;
                    c = (c - i_mesh.minunicurvature) / (i_mesh.maxunicurvature - i_mesh.minunicurvature);
                    RGB color = scalarToRGB(c);
                    glColor3f(color.r, color.g, color.b);
                    break;
                }
                case 2: {
                    float c = i_mesh.vmeancurvature.at(i_mesh.triangles[tIt][i]).length() / 2;
                     c = min(c,i_mesh.maxmeancurvature);
                    c = (c - i_mesh.minmeancurvature) / (i_mesh.maxmeancurvature - i_mesh.minmeancurvature);
                    RGB color = scalarToRGB(c);
                    glColor3f(color.r, color.g, color.b);
                    break;
                }
                default:
                    cout << "Unknown mode" << endl;
            }
            glNormal3f(n[0], n[1], n[2]);
            glVertex3f(p[0], p[1], p[2]);
        }
    }
    glEnd();
}


void drawMesh( Mesh const & i_mesh , bool draw_field = false , bool custom = true){
    if(display_smooth_normals){
        drawSmoothTriangleMesh(i_mesh, draw_field) ; //Smooth display with vertices normals
    }
    else{
        int mode = 0;
        drawTriangleMesh(i_mesh,mode) ; //Display with face normals
    }
}

void drawVectorField( vector<Vec3> const & i_positions, vector<Vec3> const & i_directions ) {
    glLineWidth(1.);
    for(unsigned int pIt = 0 ; pIt < i_directions.size() ; ++pIt) {
        Vec3 to = i_positions[pIt] + 0.02*i_directions[pIt];
        drawVector(i_positions[pIt], to);
    }
}

void drawNormals(Mesh const& i_mesh){

    if(display_smooth_normals){
        drawVectorField( i_mesh.vertices, i_mesh.normals );
    } else {
        vector<Vec3> triangle_baricenters;
        for ( const Triangle& triangle : i_mesh.triangles ){
            Vec3 triangle_baricenter (0.,0.,0.);
            for( unsigned int i = 0 ; i < 3 ; i++ )
                triangle_baricenter += i_mesh.vertices[triangle[i]];
            triangle_baricenter /= 3.;
            triangle_baricenters.push_back(triangle_baricenter);
        }

        drawVectorField( triangle_baricenters, i_mesh.triangle_normals );
    }
}

//Draw fonction
void draw () {

    if(displayMode == LIGHTED || displayMode == LIGHTED_WIRE){

        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_LIGHTING);

    }  else if(displayMode == WIRE){

        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        glDisable (GL_LIGHTING);

    }  else if(displayMode == SOLID ){
        glDisable (GL_LIGHTING);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);

    }

    if(displayMode == SOLID || displayMode == LIGHTED_WIRE){
        glEnable (GL_POLYGON_OFFSET_LINE);
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth (1.0f);
        glPolygonOffset (-2.0, 1.0);

        glColor3f(0.,0.,0.);
        drawMesh(mesh, false);

        glDisable (GL_POLYGON_OFFSET_LINE);
        glEnable (GL_LIGHTING);
    }else
    {
         glColor3f(0.8,1,0.8);
         drawMesh(mesh, true);
    }

    glDisable(GL_LIGHTING);
    if(display_normals){
        glColor3f(1.,0.,0.);
        drawNormals(mesh);
    }

    glEnable(GL_LIGHTING);


}

void changeDisplayMode(){
    if(displayMode == LIGHTED)
        displayMode = LIGHTED_WIRE;
    else if(displayMode == LIGHTED_WIRE)
        displayMode = SOLID;
    else if(displayMode == SOLID)
        displayMode = WIRE;
    else
        displayMode = LIGHTED;
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

// ------------------------------------
// User inputs
// ------------------------------------
//Keyboard event
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
        changeDisplayMode();
        break;


    case 'n': //Press n key to display normals
        display_normals = !display_normals;
        break;

    case '1': //Toggle loaded mesh display
        display_mesh = !display_mesh;
        break;

    case 's': //Switches between face normals and vertices normals
        display_smooth_normals = !display_smooth_normals;
        break;

    case '+': //Changes weight type: 0 uniforme, 1 aire des triangles, 2 angle du triangle
        weight_type ++;
        if(weight_type == 3) weight_type = 0;
        mesh.computeVerticesNormals(); //recalcul des normales avec le type de poids choisi
        break;
    case 'p':
        mesh.addNoise();
        mesh.calc_uniform_mean_curvature();

        mesh.calc_weights();
        mesh.calc_mean_curvature();

        mesh.calc_triangle_quality();
        mesh.normalizeTriangleQuality();

        mesh.computeNormals();
        break;

    case 'g':
        mesh.uniform_smooth(5);
        break;
    case 'k':
        mesh.taubinSmooth(5,0.330f,-0.331f);
        break;
    case 'l':
        mesh.laplaceBeltramiSmooth(5);
        break;

    default:
        break;
    }
    idle ();
}

//Mouse events
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

//Mouse motion, update camera
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

// ------------------------------------
// Start of graphical application
// ------------------------------------
int main (int argc, char ** argv) {
    if (argc > 2) {
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("TP HAI917I");

    init ();
    glutIdleFunc (idle);
    glutDisplayFunc (display);
    glutKeyboardFunc (key);
    glutReshapeFunc (reshape);
    glutMotionFunc (motion);
    glutMouseFunc (mouse);
    key ('?', 0, 0);

    //Mesh loaded with precomputed normals
    openOFF("data/avion_n.off", mesh.vertices, mesh.normals, mesh.triangles, mesh.triangle_normals);
   // mesh.addNoise();
 
    mesh.calc_uniform_mean_curvature();

    mesh.calc_weights();
    mesh.calc_mean_curvature();

    mesh.calc_triangle_quality();
    mesh.normalizeTriangleQuality();

   mesh.computeNormals();
    // A faire : normaliser les champs pour avoir une valeur flotante entre 0. et 1. dans current_field
    //***********************************************//

    current_field.clear();

    glutMainLoop ();
    return EXIT_SUCCESS;
}

