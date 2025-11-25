#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <functional>
#include <chrono>   
#include "mc_tables.hpp"





// ===========================================================
// Tablas estándar (declare aquí; define en un .hpp aparte):
//   - edgeTable[256]           : máscara de 12 bits por caseIndex
//   - triTable[256][16]        : tripletas de aristas por caseIndex (termina en -1)
extern const int EDGE_TABLE[256];
extern const int TRI_TABLE[256][16];
// ===========================================================

// Pares (a,b) de esquinas para cada una de las 12 aristas
static const int EDGE_ENDS[12][2] = {
  {0,1},{1,2},{2,3},{3,0},  // z=0
  {4,5},{5,6},{6,7},{7,4},  // z=1
  {0,4},{1,5},{2,6},{3,7}   // verticales
};

// Offsets (en coordenadas locales de celda) de las 8 esquinas
static const float CORNER_OFF[8][3] = {
  {0,0,0},{1,0,0},{1,1,0},{0,1,0},
  {0,0,1},{1,0,1},{1,1,1},{0,1,1}
};






struct Vec3 { float x,y,z; }; //Guardar un punto o vector 3D (x, y, z). Vértices, interpolados.
inline Vec3 operator+(const Vec3&a,const Vec3&b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; } // suma componente a componente
inline Vec3 operator-(const Vec3&a,const Vec3&b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; } // resta (vector que va de B a A).
inline Vec3 operator*(const Vec3&a,float s){ return {a.x*s,a.y*s,a.z*s}; } // escala el vector por un escalar s

static inline Vec3 lerp(const Vec3& A, const Vec3& B, float t){ //Interpolación lineal (lerp)
    return A + (B - A) * t;
     // t : Mide qué tan cerca de P_1 está de la isosuperficie
     // A y B: puntos de la arista
     // retorna punto de intersección de la arista con la isosuperficie
}






// Escribe PLY ASCII (triángulos sueltos: 3 vértices por cara) en orden
static bool write_ply_ascii(const char* path,
                            const std::vector<Vec3>& verts)
{
    const size_t ntri = verts.size()/3; // verts: contiene los puntos de todos los triangulos, tri0 → v0,v1,v2 ; tri1 → v3,v4,v5 ;
    // ntri es el número de triángulos: cada 3 vértices forman 1 cara.

    std::ofstream out(path); // abrimos el archivo 
    if(!out) return false;

    out << "ply\nformat ascii 1.0\n"; // formato de texto legible
    out << "element vertex " << verts.size() << "\n"; // número de puntos de todos los triangulos.
    out << "property float x\nproperty float y\nproperty float z\n"; // cada vértice tendrá 3 floats
    out << "element face " << ntri << "\n"; // número de triangulos
    out << "property list uchar int vertex_indices\n"; // cada cara vendrá como una lista: primero un conteo (aquí, 3) y luego 3 índices de vértices.
    out << "end_header\n";

    out.setf(std::ios::fixed); out.precision(6);
    // Escribir la lista de vértices
    for(const auto& v: verts) out << v.x << " " << v.y << " " << v.z << "\n"; // imprime cada Vec3 en una línea: x y z con 6 decimales
    // # índice v0  -> x0 y0 z0
    // # índice v1  -> x1 y1 z1
    // # índice v2  -> x2 y2 z2

    // Escribir la lista de caras (triángulos)
    for(size_t i=0;i<ntri;i++){ // dibuja superficies: cada tripleta define una cara triangular; el rasterizador la rellena.
        int base = int(i*3);
        // 3 0 1 2   # Cara 0 = triángulo con vértices [v0, v1, v2]
        // 3 3 4 5   # Cara 1 = triángulo con vértices [v3, v4, v5]
        out << "3 " << base << " " << base+1 << " " << base+2 << "\n"; // a cara i usa los índices (3i, 3i+1, 3i+2)
    }

    return true; // :D
}




// ---------------------------------------------------------------------------------
// Marching Cubes
// ---------------------------------------------------------------------------------
double marching_cubes(float (*fn)(float,float,float), // funcion escalar, devuelve float por punto
                    //límites del(bounding box)  volumen del terreno en 3D
                    uint32_t xmin, uint32_t ymin, uint32_t zmin,    
                    uint32_t xmax, uint32_t ymax, uint32_t zmax,
                    uint32_t resolution, // número nodos por eje, o tamaño de la rejilla. Si N=64, entonces sx=sy=sz=1/63 ≈ 0.01587
                    float isovalue) // umbral de la isosuperficie 0.0
{


auto start = std::chrono::high_resolution_clock::now();  // inicio


  // Espaciado uniforme (convertimos enteros a float)
    const float nx = float(resolution);
    const float ny = float(resolution);
    const float nz = float(resolution);

  // Para mapear coordenadas locales [0..1] de la celda a mundo:
    // tamaño de paso
    const float sx = (xmax - xmin) / (nx - 1.0f);
    const float sy = (ymax - ymin) / (ny - 1.0f);
    const float sz = (zmax - zmin) / (nz - 1.0f);



    auto to_world = [&](float gx, float gy, float gz)->Vec3{ // retorna la coordenada real 
        return Vec3{ xmin + gx * sx, ymin + gy * sy, zmin + gz * sz };
    };

  std::vector<Vec3> vertices; // triángulos sueltos


  // Pre-evaluamos f en toda la grilla de nodos para ahorrar evaluaciones
    const uint32_t NX = resolution, NY = resolution, NZ = resolution;

    auto idx = [&](uint32_t x,uint32_t y,uint32_t z)->size_t{
        return (size_t)z*NX*NY + (size_t)y*NX + x;
    };

    // vector de nodos que guardan los valores escalares evaluados en cada nodo
    std::vector<float> grid(NX*NY*NZ); // vector grid con tamaño #nodos^3, guarda los valores escalares evaluados en cada nodo con un indice que respeta el tamaño de paso. 


    // para cada nodo
    for(uint32_t k=0;k<NZ;k++)
    for(uint32_t j=0;j<NY;j++)
    for(uint32_t i=0;i<NX;i++){ // recorremos todos los nodos (i,j,k)
        Vec3 p = to_world(float(i), float(j), float(k)); // convertimos el indice de rejilla a coordenadas reales (x,y,z) del nodo.
        // fn: evaluamos el punto en la funcion escalar, y nos devuelve un valor escalar de ese nodo (note el signo)
        grid[idx(i,j,k)] = fn(p.x,p.y,p.z);
    }







    // Recorremos celdas: (res-1)^3 celdas
    const uint32_t Rx = resolution - 1;
    const uint32_t Ry = resolution - 1;
    const uint32_t Rz = resolution - 1;

    // Por cada celda…
    for(uint32_t k=0;k<Rz;k++)
    for(uint32_t j=0;j<Ry;j++)
    for(uint32_t i=0;i<Rx;i++){
    // Posición base (esquina 0) de la celda en coords de nodo
    float gx = float(i), gy = float(j), gz = float(k);

    // Posiciones y valores de las 8 esquinas
    Vec3 P[8]; float F[8];
    for(int c=0;c<8;c++){
        float cx = gx + CORNER_OFF[c][0];
        float cy = gy + CORNER_OFF[c][1];
        float cz = gz + CORNER_OFF[c][2];
        P[c] = to_world(cx,cy,cz);
        F[c] = grid[idx(uint32_t(cx), uint32_t(cy), uint32_t(cz))];
    }

    // caseIndex
    int caseIndex = 0; // 00000000 (8 bits)
    for(int c=0;c<8;c++) if (F[c] > isovalue) caseIndex |= (1<<c); // ¿la esquina c está "dentro"?, enciende el bit c
    if(caseIndex==0 || caseIndex==255) continue; //siguiente celda en el eje X

    // Aristas activas
    int edgeMask = EDGE_TABLE[caseIndex];
    if(edgeMask==0) continue; //redundante, grupo precavido

    // Vértices interpolados por arista
    Vec3 V[12];

    for(int e=0;e<12;e++){
        if(!(edgeMask & (1<<e))) continue;
        int a = EDGE_ENDS[e][0];
        int b = EDGE_ENDS[e][1];
        float f1 = F[a], f2 = F[b];
        float den = (f2 - f1);
        float t;
        if (std::fabs(den) < 1e-12f){
            // evitar inestabilidad; toma el extremo más cercano a tau
            t = std::fabs(isovalue - f1) < std::fabs(isovalue - f2) ? 0.0f : 1.0f;
        }else{
            t = (isovalue - f1)/den;
        }
        t = std::clamp(t, 0.0f, 1.0f);
        V[e] = lerp(P[a], P[b], t);
    }

    // Conectar triángulos según triTable
    for(int t=0; TRI_TABLE[caseIndex][t] != -1; t += 3){
        int e0 = TRI_TABLE[caseIndex][t+0];
        int e1 = TRI_TABLE[caseIndex][t+1];
        int e2 = TRI_TABLE[caseIndex][t+2];
        vertices.push_back(V[e0]);
        vertices.push_back(V[e1]);
        vertices.push_back(V[e2]);
    }
}

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double t_sec = duration.count() / 1000.0;   // segundos
    //std::cout << "Tiempo de cálculo: " << t_sec << " s\n";


    if(!write_ply_ascii("output.ply", vertices)){
        std::cerr << "No pude escribir output.ply\n";
    }else{
        // stcálculod::cout << "Listo: output.ply (" << vertices.size()/3 << " triangulos)\n";
    }
    return t_sec; 
    

}









// ====================== DEMO ===============================
// Ejemplo de función: esfera SDF de radio R centrada en (cx,cy,cz)
static float sdf_sphere(float x,float y,float z){
    const float cx=0.5f, cy=0.5f, cz=0.5f; // centro en mitad del bbox
    const float R=0.35f;
    float dx=x-cx, dy=y-cy, dz=z-cz;
    return std::sqrt(dx*dx+dy*dy+dz*dz) - R;
}

// Ejemplo de funcion: Toro (anillo) centrado en (0.5,0.5,0.5):
static float sdf_torus(float x,float y,float z){
    float cx=0.5f, cy=0.5f, cz=0.5f;
    float R=0.35f, r=0.12f;
    float X=x-cx, Y=y-cy, Z=z-cz;
    float q = std::sqrt(X*X+Y*Y) - R;
    return std::sqrt(q*q + Z*Z) - r;
}

// Gyroid (implícita periódica):
static float gyroid(float x,float y,float z){
    // escala a [0, 2π]
    const float s = 2.0f * 3.14159265f;
    x*=s; y*=s; z*=s;
    return std::sin(x)*std::cos(y) + std::sin(y)*std::cos(z) + std::sin(z)*std::cos(x);
    // isovalue≈0 da la famosa superficie de gyroid
}

static float terrain(float x,float y,float z){
    // “montaña” suave centrada
    float h = 0.4f + 0.15f*std::cos(6.28f*(x-0.5f))*std::cos(6.28f*(y-0.5f));
    return h - z; // isosuperficie a z = h(x,y)
}


int main() {
    // Número de celdas son potencias de 2^6, 2^7, ..., 2^10.

    int Ncells[] = {4,8,16,32,64, 128, 256, 512,1024}; // tamaños de problema (#celdas por lado)
    const int reps = 5;  // número de repeticiones por caso

    for (int nc : Ncells) {
        int res = nc + 1;   // nodos = celdas + 1

        for (int r = 0; r < reps; ++r) {
            double t = marching_cubes(terrain,  // o sdf_sphere, etc.
                                      0,0,0,
                                      1,1,1,
                                      res,
                                      0.0f);

            // N_celdas   tiempo_en_segundos   procesos(=1)
            std::cout << nc << " " << t << " " << 1 << "\n";
        }
    }

    return 0;
}

// g++ -o timesec.exe mc_sec.cpp
// ./timesec.exe > secuencial/tiempos_sec.txt
