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

// ----------------- GLOBAL FLOP COUNTER -----------------
long long g_flop_counter = 0;   // se llenará en marching_cubes_omp

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

// ----------------- Vectores 3D -----------------
struct Vec3 { float x,y,z; };

inline Vec3 operator+(const Vec3&a,const Vec3&b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 operator-(const Vec3&a,const Vec3&b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 operator*(const Vec3&a,float s){ return {a.x*s,a.y*s,a.z*s}; }

static inline Vec3 lerp(const Vec3& A, const Vec3& B, float t){
    // P = A + (B-A)*t
    return A + (B - A) * t;
}

// ----------------- Escritura PLY -----------------
static bool write_ply_ascii(const char* path,
                            const std::vector<Vec3>& verts)
{
    const size_t ntri = verts.size()/3;

    std::ofstream out(path);
    if(!out) return false;

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << verts.size() << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "element face " << ntri << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    out.setf(std::ios::fixed);
    out.precision(6);

    for(const auto& v: verts)
        out << v.x << " " << v.y << " " << v.z << "\n";

    for(size_t i=0;i<ntri;i++){
        int base = int(i*3);
        out << "3 " << base << " " << base+1 << " " << base+2 << "\n";
    }
    return true;
}

// ------------------------------------------------------------------
// Versión paralela con OpenMP + conteo de FLOPs
// ------------------------------------------------------------------
double marching_cubes_omp(float (*fn)(float,float,float),
                          uint32_t xmin, uint32_t ymin, uint32_t zmin,
                          uint32_t xmax, uint32_t ymax, uint32_t zmax,
                          uint32_t resolution,
                          float isovalue,
                          long long &flops_out)
{
    g_flop_counter = 0;  // reset global
    double t0 = omp_get_wtime();

    const float nx = float(resolution);
    const float ny = float(resolution);
    const float nz = float(resolution);

    const float sx = (xmax - xmin) / (nx - 1.0f);
    const float sy = (ymax - ymin) / (ny - 1.0f);
    const float sz = (zmax - zmin) / (nz - 1.0f);

    auto to_world = [&](float gx, float gy, float gz)->Vec3{
        // 3 mults + 3 sumas ≈ 6 FLOPs
        return Vec3{ xmin + gx * sx, ymin + gy * sy, zmin + gz * sz };
    };

    const uint32_t NX = resolution, NY = resolution, NZ = resolution;

    auto idx = [&](uint32_t x,uint32_t y,uint32_t z)->size_t{
        return (size_t)z*NX*NY + (size_t)y*NX + x;
    };

    // -------- Fase 1: volumen escalar fn(x,y,z) --------
    vector<float> grid(NX*NY*NZ);

    #pragma omp parallel for collapse(3) schedule(static)
    for(uint32_t k=0;k<NZ;k++)
    for(uint32_t j=0;j<NY;j++)
    for(uint32_t i=0;i<NX;i++){
        float x = xmin + i * sx;
        float y = ymin + j * sy;
        float z = zmin + k * sz;
        grid[idx(i,j,k)] = fn(x, y, z);
        // Si quisieras, aquí podrías sumar FLOPs del fn(...),
        // pero lo dejamos sencillo y solo contamos en la fase 2.
    }

    // -------- Fase 2: Marching Cubes sobre celdas --------
    const uint32_t Rx = resolution - 1;
    const uint32_t Ry = resolution - 1;
    const uint32_t Rz = resolution - 1;

    std::vector<Vec3> vertices;   // compartido entre hilos

    #pragma omp parallel
    {
        long long local_flops = 0;        // contador local por hilo
        std::vector<Vec3> local_vertices; // buffer local por hilo

        #pragma omp for nowait collapse(3) schedule(dynamic)
        for(uint32_t k=0;k<Rz;k++)
        for(uint32_t j=0;j<Ry;j++)
        for(uint32_t i=0;i<Rx;i++){

            float gx = float(i), gy = float(j), gz = float(k);

            Vec3  P[8];
            float F[8];

            for(int c=0;c<8;c++){
                float cx = gx + CORNER_OFF[c][0];
                float cy = gy + CORNER_OFF[c][1];
                float cz = gz + CORNER_OFF[c][2];

                // to_world: ~6 FLOPs
                local_flops += 6;
                P[c] = to_world(cx,cy,cz);
                F[c] = grid[idx(uint32_t(cx), uint32_t(cy), uint32_t(cz))];
            }

            int caseIndex = 0;
            for (int c=0;c<8;c++) {
                // comparaciones F[c] > isovalue: NO las contamos como FLOPs
                if (F[c] > isovalue) caseIndex |= (1<<c);
            }

            if(caseIndex==0 || caseIndex==255) continue;

            int edgeMask = EDGE_TABLE[caseIndex];
            if(edgeMask==0) continue;

            Vec3 V[12];

            for(int e=0;e<12;e++){
                if(!(edgeMask & (1<<e))) continue;
                int a = EDGE_ENDS[e][0];
                int b = EDGE_ENDS[e][1];

                float f1 = F[a], f2 = F[b];
                float den = (f2 - f1);               // 1 FLOP (resta)
                float t;

                if (std::fabs(den) < 1e-12f){
                    t = std::fabs(isovalue - f1) < std::fabs(isovalue - f2) ? 0.0f : 1.0f;
                    // aquí hay algunas restas/comparaciones; para simplificar no las sumamos
                }else{
                    t = (isovalue - f1)/den;         // 1 resta + 1 división = 2 FLOPs
                    local_flops += 2;
                }

                t = std::clamp(t, 0.0f, 1.0f);
                V[e] = lerp(P[a], P[b], t);
                // lerp ≈ 6 FLOPs (restas, mults, sumas)
                local_flops += 6;
            }

            // Triángulos → se agregan al buffer local
            for(int t=0; TRI_TABLE[caseIndex][t] != -1; t += 3){
                int e0 = TRI_TABLE[caseIndex][t+0];
                int e1 = TRI_TABLE[caseIndex][t+1];
                int e2 = TRI_TABLE[caseIndex][t+2];
                local_vertices.push_back(V[e0]);
                local_vertices.push_back(V[e1]);
                local_vertices.push_back(V[e2]);
            }
        } // fin bucles de celdas

        // Cada hilo suma su contador local UNA sola vez al global
        #pragma omp atomic
        g_flop_counter += local_flops;

        // Fusionar vértices locales en el vector global
        #pragma omp critical
        {
            vertices.insert(vertices.end(),
                            local_vertices.begin(),
                            local_vertices.end());
        }
    } // fin región parallel

    double t1 = omp_get_wtime();
    double dt = t1 - t0;

    if(!write_ply_ascii("output_omp.ply", vertices)){
        std::cerr << "No pude escribir output_omp.ply\n";
    }

    flops_out = g_flop_counter;
    return dt;
}

// ----------------- Función terreno -----------------
static float terrain(float x,float y,float z){
    float h = 0.4f + 0.15f*std::cos(6.28f*(x-0.5f))*std::cos(6.28f*(y-0.5f));
    return h - z; // isosuperficie a z = h(x,y)
}

// ----------------- MAIN: barrido Ncells / hilos -----------------
int main() {
    int Ncells[]  = {4,8,16,32,64,128,256,512};   // tamaños de problema
    int threads[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};  // hilos

    const int reps = 5;  // repeticiones por caso

    // Salida: N_celdas  tiempo(s)  hilos  FLOPs  GFLOPS
    for (int nc : Ncells) {
        int res = nc + 1;  // nodos = celdas + 1

        for (int p : threads) {
            omp_set_num_threads(p);

            for (int r = 0; r < reps; ++r) {
                long long flops;
                double t = marching_cubes_omp(terrain,
                                              0,0,0,
                                              1,1,1,
                                              res,
                                              0.0f,
                                              flops);

                double gflops = flops / (t * 1e9);

                std::cout << nc    << " "
                          << t     << " "
                          << p     << " "
                          << flops << " "
                          << gflops << "\n";
            }
        }
    }
    return 0;
}

/*
Formato de cada línea:
    Ncells   Tiempo(s)   Hilos   FLOPs   GFLOPS
*/


// g++ -fopenmp mc_omp.cpp -o mc_omp.exe
// ./mc_omp.exe > paralelo/tiempos_ompflop.txt
