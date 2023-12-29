#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <numeric>
#include <cmath>

// Uncomment for ISPC
#include "module_ispc.h"
using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS          //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * embedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

         for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // QK_t = Q @ K.T = (B, H, N, d) @ (B, H, d, N) = (B, H, N, N)
            for (int i0 = 0; i0 < N; i0++) {
                for (int i1 = 0; i1 < N; i1++) {
                    float QK_t_i0i1 = 0.0;
                    for (int j = 0; j < d; j++) 
                        QK_t_i0i1 += fourDimRead(Q, b, h, i0, j, H, N, d) * fourDimRead(K, b, h, i1, j, H, N, d);
                    twoDimWrite(QK_t, i0, i1, N, QK_t_i0i1);
                }
                float rowSum = 0.0;
                for (int i1 = 0; i1 < N; i1++)
                    rowSum += exp(twoDimRead(QK_t, i0, i1, N));

                for (int i1 = 0; i1 < N; i1++) {
                    float QK_t_i0i1 = exp(twoDimRead(QK_t, i0, i1, N));
                    QK_t_i0i1 /= rowSum;
                    twoDimWrite(QK_t, i0, i1, N, QK_t_i0i1);
                }
            }
            // O = QK_t @ V = (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
            for (int i0 = 0; i0 < N; i0++) {
                for (int j = 0; j < d; j++) {
                    float O_bhi0j = 0.0;
                    for (int i1 = 0; i1 < N; i1++)
                        O_bhi0j += twoDimRead(QK_t, i0, i1, N) * fourDimRead(V, b, h, i1, j, H, N, d);
                    
                    fourDimWrite(O, b, h, i0, j, H, N, d, O_bhi0j);
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
#define BLOCK_SIZE 128
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i0 = 0; i0 < N; i0+=BLOCK_SIZE) {
                for (int i1 = 0; i1 < N; i1+=BLOCK_SIZE) {
                    for (int j = 0; j < d; j++) {
                        for (int i0_0 = i0; i0_0 < std::min(i0+BLOCK_SIZE, N); i0_0++) {
                            for (int i1_0 = i1; i1_0 < std::min(i1+BLOCK_SIZE, N); i1_0++) {
                                float QK_t_i0_0i1_0 = twoDimRead(QK_t, i0_0, i1_0, N);
                                QK_t_i0_0i1_0 += fourDimRead(Q, b, h, i0_0, j, H, N, d) * fourDimRead(K, b, h, i1_0, j, H, N, d);
                                twoDimWrite(QK_t, i0_0, i1_0, N, QK_t_i0_0i1_0);
                            }
                        }
                    }
                }
            }
            for (int i0 = 0; i0 < N; i0++) {
                float rowSum = 0.0;
                for (int i1 = 0; i1 < N; i1++)
                    rowSum += exp(twoDimRead(QK_t, i0, i1, N));
                for (int i1 = 0; i1 < N; i1++) {
                    float QK_t_i0i1 = exp(twoDimRead(QK_t, i0, i1, N));
                    QK_t_i0i1 /= rowSum;
                    twoDimWrite(QK_t, i0, i1, N, QK_t_i0i1);
                }
            }
            for (int i0 = 0; i0 < N; i0+=BLOCK_SIZE) {
                for (int j = 0; j < d; j+=BLOCK_SIZE) {
                    for (int i1 = 0; i1 < N; i1++) {
                        for (int i0_0 = i0; i0_0 < std::min(i0+BLOCK_SIZE, N); i0_0++) {
                            for (int j_0 = j; j_0 < std::min(j+BLOCK_SIZE, d); j_0++) {
                                float O_bhi0_0j_0 = fourDimRead(O, b, h, i0_0, j_0, H, N, d);
                                O_bhi0_0j_0 += twoDimRead(QK_t, i0_0, i1, N) * fourDimRead(V, b, h, i1, j_0, H, N, d);
                                fourDimWrite(O, b, h, i0_0, j_0, H, N, d, O_bhi0_0j_0);
                            }
                        }
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    // std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i0 = 0; i0 < N; i0++) { // Q[i0] is a single row of Q
		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor); // ORow.shape = N
		// YOUR CODE HERE
                for (int i1 = 0; i1 < N; i1++)
                    for (int j = 0; j < d; j++)
                        ORow[i1] += fourDimRead(Q, b, h, i0, j, H, N, d) * fourDimRead(K, b, h, i1, j, H, N, d);

                for (int i1 = 0; i1 < N; i1++)
                    ORow[i1] = exp(ORow[i1]);

                float rowSum = std::accumulate(ORow.begin(), ORow.end(), 0.0f);
                std::transform(ORow.begin(), ORow.end(), ORow.begin(), [rowSum](float x) { return x / rowSum; });

                for (int j = 0; j < d; j++) {
                    float O_bhi0j = 0.0f;
                    for (int i1 = 0; i1 < N; i1++) 
                        O_bhi0j += ORow[i1] * fourDimRead(V, b, h, i1, j, H, N, d);
                    fourDimWrite(O, b, h, i0, j, H, N, d, O_bhi0j);
                }
            }
	    }
    }

	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

void MM2d(std::vector<float> &A, bool transA, std::vector<float> &B, bool transB, 
          std::vector<float> &C, int M, int N, int P) {
    // credit to https://github.com/google/gemmlowp/blob/master/test/test.cc#L35
    int a_i_stride, a_k_stride;
    if (transA) {
        a_i_stride = 1;
        a_k_stride = M;
    } else {
        a_i_stride = P;
        a_k_stride = 1;
    }
    int b_j_stride, b_k_stride;
    if (transB) {
        b_j_stride = P;
        b_k_stride = 1;
    } else {
        b_j_stride = 1;
        b_k_stride = N;
    }
    
    for (int k = 0; k < P; k++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                const int a_index = i * a_i_stride + k * a_k_stride;
                const int b_index = j * b_j_stride + k * b_k_stride;
                float val = twoDimRead(C, i, j, N) + A[a_index] * B[b_index];
                twoDimWrite(C, i, j, N, val);
            }
}

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // -------- YOUR CODE HERE  -------- //
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i0 = 0; i0 < N; i0+=Br) {
                int _Br = std::min(Br, N - i0);
                // load a tile from Q to Qi
                auto Qtile = QTensor.index({b, h, torch::indexing::Slice(i0, i0 + _Br)});
                std::vector<float> Qi = formatTensor(Qtile);        // (Br, d)

                // allocate buffers for Oi, li
                std::vector<float> Oi = formatTensor(OiTensor);     // (Br, d)
                std::vector<float> li = formatTensor(LiTensor);     // (Br)

                std::vector<float> Kj;    // (Bc, d)
                std::vector<float> Vj;    // (Bc, d)

                for (int i1 = 0; i1 < N; i1+=Bc) {
                    int _Bc = std::min(Bc, N - i1);
                    // load a tile from K and V
                    auto Ktile = KTensor.index({b, h, torch::indexing::Slice(i1, i1 + _Bc)});
                    auto Vtile = VTensor.index({b, h, torch::indexing::Slice(i1, i1 + _Bc)});
                    Kj = formatTensor(Ktile);    // (Bc, d)
                    Vj = formatTensor(Vtile);    // (Bc, d)

                    // allocate buffers for Sij, Pij, lij, lnew
                    std::vector<float> Sij = formatTensor(SijTensor);   // (Br, Bc)
                    std::vector<float> Pij = formatTensor(PijTensor);   // (Br, Bc)
                    std::vector<float> lij = formatTensor(LijTensor);   // (Br)
                    std::vector<float> lnew = formatTensor(LnewTensor); // (Br)
                    
                    // Sij = QiKj_T
                    MM2d_v(Qi.data(), /*transA=*/false, 
                           Kj.data(), /*transB=*/true, 
                           Sij.data(), _Br, _Bc, d);

                    // Pij = exp(Sij)
                    for (int ir = 0; ir < _Br; ir++)
                        for (int ic = 0; ic < _Bc; ic++) {
                            float s = exp(twoDimRead(Sij, ir, ic, _Bc));
                            twoDimWrite(Pij, ir, ic, _Bc, s);
                        }      
                
                    // lij = rowSum(Pij)
                    for (int ir = 0; ir < _Br; ir++) {
                        std::vector<float> row(Pij.begin() + ir * _Bc, Pij.begin() + ir * _Bc + _Bc);
                        lij[ir] = rowSum_v(row.data(), _Bc);
                    }
                        

                    // lnew = li + lij
                    for (int ir = 0; ir < _Br; ir++)
                        lnew[ir] = li[ir] + lij[ir];
                    
                    // Oi = (liOi + PijVj) / lnew
                    for (int ir = 0; ir < _Br; ir++)
                        for (int j = 0; j < d; j++) {
                            float oi = li[ir] * twoDimRead(Oi, ir, j, d);
                            twoDimWrite(Oi, ir, j, d, oi);
                        }
                    
                    MM2d_v(Pij.data(), /*transA=*/false, 
                           Vj.data(), /*transB=*/false, 
                           Oi.data(), _Br, d, _Bc);
                    
                    for (int ir = 0; ir < _Br; ir++)
                        for (int j = 0; j < d; j++) {
                            float oi = twoDimRead(Oi, ir, j, d) / lnew[ir];
                            twoDimWrite(Oi, ir, j, d, oi);
                        }

                    li = lnew;  // update li
                }

                // write Oi to memory
                int offset = b * H * N * d + h * N * d + i0 * d;
                std::copy(Oi.begin(), Oi.begin() + _Br*d, O.begin() + offset);
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
