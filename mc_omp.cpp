#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <functional>
#include <omp.h>   
#include "mc_tables.hpp"
using namespace std;

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

// Escribe PLY ASCII 
static bool write_ply_ascii(const char* path,
                            const std::vector<Vec3>& verts)
{
    const size_t ntri = verts.size()/3; // numero de triángulos

    std::ofstream out(path);
    if(!out) return false;

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << verts.size() << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "element face " << ntri << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    out.setf(std::ios::fixed); out.precision(6);
    for(const auto& v: verts) // Escribe TODOS los vértices (coordenadas)
        out << v.x << " " << v.y << " " << v.z << "\n";

    for(size_t i=0;i<ntri;i++){ // (indices de vértices) 
        int base = int(i*3);
        out << "3 " << base << " " << base+1 << " " << base+2 << "\n";
    }

    /*
    ... header ...
    0.130859 0.001953 0.501951   ← vértice 0
    0.130859 0.001543 0.501953   ← vértice 1
    0.130856 0.001953 0.501953   ← vértice 2
    ...
    3 0 1 2
    */
    return true;
}


// ------------------------------------------------------------------
// Versión paralela con OpenMP
// ------------------------------------------------------------------
double marching_cubes_omp(float (*fn)(float,float,float),
                        uint32_t xmin, uint32_t ymin, uint32_t zmin,
                        uint32_t xmax, uint32_t ymax, uint32_t zmax,
                        uint32_t resolution,
                        float isovalue)
{
    

    double t0 = omp_get_wtime();

    const float nx = float(resolution);
    const float ny = float(resolution);
    const float nz = float(resolution);

    const float sx = (xmax - xmin) / (nx - 1.0f);
    const float sy = (ymax - ymin) / (ny - 1.0f);
    const float sz = (zmax - zmin) / (nz - 1.0f);

    auto to_world = [&](float gx, float gy, float gz)->Vec3{
        return Vec3{ xmin + gx * sx, ymin + gy * sy, zmin + gz * sz };
    };

    const uint32_t NX = resolution, NY = resolution, NZ = resolution;

    auto idx = [&](uint32_t x,uint32_t y,uint32_t z)->size_t{
        return (size_t)z*NX*NY + (size_t)y*NX + x;
    };

    // Fase 1: calculo el volumen escalar grid[i][j][k] (paralelo)
    vector<float> grid(NX*NY*NZ);

    // Aplicamos la cláusula collapse(3) al for paralelo para fusionar los 3 bucles anidados
    // en un solo espacio de iteraciones de tamaño NX×NY×NZ. Así, todas las combinaciones (k,j,i)
    // (es decir, todos los nodos de la rejilla) pueden repartirse en paralelo entre los hilos.
    // Además un ordenamiento estatico, cada hilo se queda con sus iteraciones fijas.

    #pragma omp parallel for collapse(3) schedule(static)
    for(uint32_t k=0;k<NZ;k++)
    for(uint32_t j=0;j<NY;j++)
    for(uint32_t i=0;i<NX;i++){
        float x = xmin + i * sx;
        float y = ymin + j * sy;
        float z = zmin + k * sz;
        grid[idx(i,j,k)] = fn(x, y, z);
    }

    // -------------------------
    // Fase 2: Marching Cubes sobre celdas (paralelo)
    // -------------------------
    const uint32_t Rx = resolution - 1;
    const uint32_t Ry = resolution - 1;
    const uint32_t Rz = resolution - 1;

    // vertices es compartido entre todos los hilos (por defecto en OpenMP, las variables definidas fuera del parallel son shared
    std::vector<Vec3> vertices;   // global, se llenará fusionando buffers locales

    #pragma omp parallel // crea un equipo de hilos
    {
        // Cada hilo crea su propio local_vertices, es privado a cada hilo
        std::vector<Vec3> local_vertices;  // buffer local por hilo



        // Aplicamos la cláusula nowait al for paralelo, para que los hilos no hagan sincronización al terminar el for, y pasen a la fase de combinación.
        // además de collapse(3) para repartir todas las celdas en paralelo entre los hilos. Y con un ordenamiento dinámico
        // Cada hilo no tiene un número fijo de celdas. Cada hilo va pidiendo celdas conforme va terminando lo que le toca. Ya que 
        // no todas las celdas generan el mismo número de triángulos solucionando el desbalance de carga.
        #pragma omp for nowait collapse(3) schedule(dynamic)
        for(uint32_t k=0;k<Rz;k++)
        for(uint32_t j=0;j<Ry;j++)
        for(uint32_t i=0;i<Rx;i++){

            float gx = float(i), gy = float(j), gz = float(k);

            // Esquinas y valores (locales al hilo/iteración)
            Vec3 P[8];
            float F[8];
            for(int c=0;c<8;c++){
                float cx = gx + CORNER_OFF[c][0];
                float cy = gy + CORNER_OFF[c][1];
                float cz = gz + CORNER_OFF[c][2];
                P[c] = to_world(cx,cy,cz);
                F[c] = grid[idx(uint32_t(cx), uint32_t(cy), uint32_t(cz))];
            }

            // caseIndex
            int caseIndex = 0;
            for(int c=0;c<8;c++)
                if (F[c] > isovalue) caseIndex |= (1<<c);

            if(caseIndex==0 || caseIndex==255) continue;

            int edgeMask = EDGE_TABLE[caseIndex];
            if(edgeMask==0) continue;

            // Vértices interpolados por arista (locales)
            Vec3 V[12];

            for(int e=0;e<12;e++){
                if(!(edgeMask & (1<<e))) continue;
                int a = EDGE_ENDS[e][0];
                int b = EDGE_ENDS[e][1];
                float f1 = F[a], f2 = F[b];
                float den = (f2 - f1);
                float t;
                if (std::fabs(den) < 1e-12f){
                    t = std::fabs(isovalue - f1) < std::fabs(isovalue - f2) ? 0.0f : 1.0f;
                }else{
                    t = (isovalue - f1)/den;
                }
                t = std::clamp(t, 0.0f, 1.0f);
                V[e] = lerp(P[a], P[b], t);
            }

            // Triángulos según TRI_TABLE → se agregan al buffer local
            for(int t=0; TRI_TABLE[caseIndex][t] != -1; t += 3){
                int e0 = TRI_TABLE[caseIndex][t+0];
                int e1 = TRI_TABLE[caseIndex][t+1];
                int e2 = TRI_TABLE[caseIndex][t+2];
                local_vertices.push_back(V[e0]);
                local_vertices.push_back(V[e1]);
                local_vertices.push_back(V[e2]);
            }
        }

        // Cada hilo, después de procesar sus celdas y generar los triángulos(coordenadas de vértices de la superficie)(valor interpolado de la celda) 
        // en su buffer local (local_vertices), 
        // entra una vez a la seccion crítica para fusionar su buffer local en el vector global vertices.
        #pragma omp critical
        {
            vertices.insert(vertices.end(), local_vertices.begin(), local_vertices.end());
        }
    } // fin región parallel

    double t1 = omp_get_wtime();
    double dt = t1 - t0;
    //std::cout << "Tiempo de Marching Cubes (OMP): " << dt << " segundos\n";

    if(!write_ply_ascii("output_omp_sin_critical.ply", vertices)){  // NO se escribió el archivo correctamente.
        std::cerr << "No pude escribir output_omp.ply\n"; 
    }else{ // se escribió el archivo correctamente.
        //std::cout << "Listo: output_omp.ply (" << vertices.size()/3 << " triangulos)\n";
    }
    return dt;
}

static float terrain(float x,float y,float z){
    // “montaña” suave centrada
    float h = 0.4f + 0.15f*std::cos(6.28f*(x-0.5f))*std::cos(6.28f*(y-0.5f));
    return h - z; // isosuperficie a z = h(x,y)
}



int main() {
    int Ncells[]  = {512,1024};   // tamaños de problema (#celdas por lado)
    //int threads[] = {1, 2, 3, 4, 5, 6, 7, 8};  // #hilos a probar
    int threads[] = {16, 17, 18};
    //19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32

    const int reps = 5;                    // repeticiones por caso

    // Formato: N_celdas   tiempo(s)   hilos
    for (int nc : Ncells) {
        int res = nc + 1;  // nodos

        for (int p : threads) {
            omp_set_num_threads(p);

            for (int r = 0; r < reps; ++r) {
                double t = marching_cubes_omp(terrain,
                                              0,0,0,
                                              1,1,1,
                                              res,
                                              0.0f);
                std::cout << nc << " " << t << " " << p << "\n";
            }
        }
    }
    return 0;
}



// g++ -fopenmp mc_omp_flops.cpp -o mc_omp.exe
// ./mc_omp.exe > paralelo/tiempos_omp.txt
