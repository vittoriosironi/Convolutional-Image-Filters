#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;

void sparseKernel(SparseMatrix<double> &A1, const MatrixXd& Hav1, int width, int height) {
    std::vector<Triplet<double>> triplets;
    int kernel_rows = Hav1.rows();
    int kernel_cols = Hav1.cols();
    int center_r = kernel_rows / 2;
    int center_c = kernel_cols / 2;
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int current_pixel_idx = i * width + j;
            for (int kr = 0; kr < kernel_rows; ++kr) {
                for (int kc = 0; kc < kernel_cols; ++kc) {
                    
                    int neighbor_i = i + (kr - center_r);
                    int neighbor_j = j + (kc - center_c);
                    
                    if (neighbor_i >= 0 && neighbor_i < height && neighbor_j >= 0 && neighbor_j < width) {
                        int neighbor_pixel_idx = neighbor_i * width + neighbor_j;
                        double kernel_val = Hav1(kr, kc);
                        if (kernel_val != 0) {
                            triplets.push_back(Triplet<double>(current_pixel_idx, neighbor_pixel_idx, kernel_val));
                        }
                    }
                }
            }
        }
    }
    A1.setFromTriplets(triplets.begin(), triplets.end());
}

Eigen::VectorXd loadMtxVector(const std::string& filename){
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        } else {
            break;
        }
    }

    int n;
    {
        std::istringstream iss(line);
        iss >> n;
    }

    Eigen::VectorXd vec(n);
    vec.setZero();

    int index;
    double value;
    while (infile >> index >> value) {
        if (index >= 0 && index < n) {
            vec(index) = value;
        }
    }

    infile.close();
    return vec;
}

int main() {
    // Punto 1    
    const char* input_path = "./img/uma.jpg";
    int width, height, channels;
    
    unsigned char* image_data_p_1 = stbi_load(input_path, &width, &height, &channels, 1);

    MatrixXd img(height, width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            img(i, j) = image_data_p_1[index];
        }
    }

    stbi_image_free(image_data_p_1);

    cout << "Matrix size: " << height << " x " << width << " (m x n)" << endl;
    
    // Punto 2
    srand(static_cast<unsigned int>(time(0)));
    MatrixXd mat = MatrixXd::Random(height, width); 
    mat *= 40;
    
    MatrixXd img_noise = img + mat;
    img_noise = img_noise.unaryExpr([](double val) -> double {
        return std::min(std::max(val, 0.0), 255.0);
    });

    std::vector<unsigned char> img_data_p_2(width * height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            img_data_p_2[i * width + j] = static_cast<unsigned char>(img_noise(i, j));
        }
    }

    const std::string output_image_path_p_2 = "./img/p_2.png";
    if (stbi_write_png(output_image_path_p_2.c_str(), width, height, 1,
    img_data_p_2.data(), width) == 0) {
        std::cerr << "Error: Could not save noised image" << std::endl;
        return 1;
    }
    
    // Punto 3
    Eigen::VectorXd v = img.reshaped<Eigen::RowMajor>();
    Eigen::VectorXd w = img_noise.reshaped<Eigen::RowMajor>();
    cout << "Vector v has " << v.rows() << " components (m*n = " << height*width << ")" << endl;
    cout << "Vector w has " << w.rows() << " components (m*n = " << height*width << ")" << endl;
    cout << "Euclidean norm of v: " << v.norm() << endl;

    // Punto 4
    SparseMatrix<double> A1(width*height, width*height);
    std::vector<Triplet<double>> triplets;

    // Kernel of the filter
    MatrixXd Hav1(3, 3);
    Hav1 << 1, 1, 0,
            1, 2, 1,
            0, 1, 1;
    Hav1 /= 8;

    sparseKernel(A1, Hav1, width, height);

    cout << "Elementi non nulli A1: " << A1.nonZeros() << endl;

    // Punto 5
    VectorXd w_filtered = A1 * w;

    vector<unsigned char> image_data_p_4(width * height);
    for (int i = 0; i < w_filtered.size(); ++i) {
        double val = min(max(w_filtered(i), 0.0), 255.0);
        image_data_p_4[i] = static_cast<unsigned char>(val);
    }

    const string output_image_path_p_4 = "./img/p_5.png";
    if (stbi_write_png(output_image_path_p_4.c_str(), width, height, 1,
                        image_data_p_4.data(), width) == 0) {
        cerr << "Error: Could not save filtered image" << endl;

        return 1;
    }

    
    // Punto 6
    MatrixXd Hsh1(3, 3);
    Hsh1 << 0, -2, 0,
            -2, 9, -2,
            0, -2, 0;

    SparseMatrix<double> A2(width*height, width*height);

    sparseKernel(A2, Hsh1, width, height);

    cout << "Elementi non nulli A2: " << A2.nonZeros() << endl;

    // Check of the simmetry of matrix A2
    SparseMatrix<double> B = SparseMatrix<double>(A2.transpose()) - A2;
    std::cout << "Norm of skew-symmetric part for A2: " << B.norm() << endl;


    // Punto 7
    VectorXd w_filtered2 = A2 * v;

    vector<unsigned char> img_data_p_7(width * height);
    for (int i = 0; i < w_filtered2.size(); ++i) {
        double val = min(max(w_filtered2(i), 0.0), 255.0);
        img_data_p_7[i] = static_cast<unsigned char>(val);
    }

    const string output_image_path_p_7 = "./img/p_7.png";
    if (stbi_write_png(output_image_path_p_7.c_str(), width, height, 1,
                        img_data_p_7.data(), width) == 0) {
        cerr << "Error: Could not save filtered image" << endl;

        return 1;
    }

    // Punto 8
    std::string matrixFileOut("./data/A2.mtx");
    Eigen::saveMarket(A2, matrixFileOut);

    int n = w.size();
    FILE* out = fopen("./data/w.mtx","w");
    fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out,"%d\n", n);
    for (int i=0; i<n; i++) {
        fprintf(out,"%d %f\n", i ,w(i));
    }
    fclose(out);

    // lis command in order to solve the linear problem
    // ./../../lis-2.1.10/test/test1 data/A2.mtx data/w.mtx lis_solution/sol.mtx lis_solution/hist.txt -i cg -tol 1e-14 -p jacobi

    // number of processes = 1
    // matrix size = 228000 x 228000 (1138088 nonzero entries)

    // initial vector x      : all components set to 0
    // precision             : double
    // linear solver         : CG
    // preconditioner        : Jacobi
    // convergence condition : ||b-Ax||_2 <= 1.0e-14 * ||b-Ax_0||_2
    // matrix storage format : CSR
    // linear solver status  : normal end

    // CG: number of iterations = 65
    // CG:   double             = 65
    // CG:   quad               = 0
    // CG: elapsed time         = 1.928297e+00 sec.
    // CG:   preconditioner     = 1.046648e-01 sec.
    // CG:     matrix creation  = 7.645800e-05 sec.
    // CG:   linear solver      = 1.823632e+00 sec.
    // CG: relative residual    = 6.663198e-15

    // Punto 9
    Eigen::VectorXd sol = loadMtxVector("lis_solution/sol.mtx");
    vector<unsigned char> img_data_p_9(width * height);
    
    for (int i = 0; i < sol.size(); ++i) {
        double val = min(max(sol(i), 0.0), 255.0);
        img_data_p_9[i] = static_cast<unsigned char>(val);
    }

    const std::string output_image_path_p_9 = "./img/p_9.png";
    if (stbi_write_png(output_image_path_p_9.c_str(), width, height, 1, img_data_p_9.data(), width) == 0) {
        std::cerr << "Error: Could not save solution image" << std::endl;
        return 1;
    }


    // Punto 10
    MatrixXd Hed2(3, 3);
    Hed2 << -1, -2, -1,
             0,  0,  0,
             1,  2,  1;

    SparseMatrix<double> A3(width*height, width*height);
    sparseKernel(A3, Hed2, width, height);

    SparseMatrix<double> B3 = SparseMatrix<double>(A3.transpose()) - A3;  // Check symmetry
    std::cout << "Norm of skew-symmetric part for A3: " << B3.norm() << endl;

    // Punto 11
    VectorXd w_filtered3 = A3 * v;

    vector<unsigned char> img_data_p_11(width * height);
    for (int i = 0; i < w_filtered3.size(); ++i) {
        double val = min(max(w_filtered3(i), 0.0), 255.0);
        img_data_p_11[i] = static_cast<unsigned char>(val);
    }

    const string output_image_path_p_11 = "./img/p_11.png";
    if (stbi_write_png(output_image_path_p_11.c_str(), width, height, 1,
                        img_data_p_11.data(), width) == 0) {
        cerr << "Error: Could not save filtered image" << endl;

        return 1;
    }

    // Punto 12
    SparseMatrix<double> A4;
    SparseMatrix<double> I(width*height, width*height);
    I.setIdentity();
    I *= 3;
    A4 = A3 + I;
    Eigen::BiCGSTAB<SparseMatrix<double>> bicgstb;
    bicgstb.setTolerance(1e-8);
    bicgstb.compute(A4);
    VectorXd x = bicgstb.solve(w);

    std::cout << "BiCGSTAB" << endl;
    std::cout << "#iterations: " << bicgstb.iterations() << endl;
    std::cout << "residual: " << bicgstb.error() << endl;

    // Punto 13
    vector<unsigned char> img_data_p_13(width * height);
    for (int i = 0; i < x.size(); ++i) {
        double val = min(max(x(i), 0.0), 255.0);
        img_data_p_13[i] = static_cast<unsigned char>(val);
    }

    const string output_image_path_p_13 = "./img/p_13.png";
    if (stbi_write_png(output_image_path_p_13.c_str(), width, height, 1,
                        img_data_p_13.data(), width) == 0) {
        cerr << "Error: Could not save filtered image" << endl;

        return 1;
    }

    return 0;
}
