#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>

#define DEBUG 0
#define BUFF_SIZE 8192

int world_rank, world_size;
int n_split[20], n_offset[20];

void read_metrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    int num = 0, i = 0, j = 0, bytes = 0, hasnum = 0, eof = 0;
    char buffer[BUFF_SIZE], *p;

    while (!eof) {
        in.read(buffer, sizeof(buffer));
        p = buffer;
        bytes = BUFF_SIZE;

        while (bytes > 0) {
            if (*p == 26) {
                eof = 1;
                break;
            }

            if (*p == '\n' || *p == ' ') {
                if (hasnum) {
                    if (i < *n_ptr * *m_ptr)
                        (*a_mat_ptr)[i++] = num;
                    else
                        (*b_mat_ptr)[j++] = num;

                    if (j == *m_ptr * *l_ptr)
                        break;
                }
                num = 0;
                hasnum = 0;
                p++;
                bytes--;
            }
            else if (*p >= '0' && *p <= '9') {
                num = num * 10 + *p - '0';
                hasnum = 1;
                p++;
                bytes--;
            }
            else
                exit(1);
        }

        memset(buffer, 26, sizeof(buffer));
    }
}

void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    in >> *n_ptr >> *m_ptr >> *l_ptr;
#if DEBUG
    std::cout << "n: " << *n_ptr << " m: " << *m_ptr << " l: " << *l_ptr << std::endl;
#endif

    // calculate the split and offset for each process
    n_split[0] = *n_ptr / world_size + (0 < *n_ptr % world_size);
    for (int i = 1; i < world_size; i++) {
        n_split[i] = *n_ptr / world_size + (i < *n_ptr % world_size);
        n_offset[i] = n_offset[i - 1] + n_split[i - 1];
    }
#if DEBUG
    for (int i = 0; i < world_size; i++)
        std::cout << "n_split[" << i << "]: " << n_split[i] << ", n_offset[" << i << "]: " << n_offset[i] << std::endl;
#endif

    *a_mat_ptr = new int[*n_ptr * *m_ptr];
    *b_mat_ptr = new int[*m_ptr * *l_ptr];

    read_metrices(in, n_ptr, m_ptr, l_ptr, a_mat_ptr, b_mat_ptr);
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{
    // matrix multiplication
    alignas(64) int *c_mat = new int[n_split[world_rank] * l]();

    const int BLOCK_SIZE = 64;
    for (int ii = 0; ii < n_split[world_rank]; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < m; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < l; jj += BLOCK_SIZE) {
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, n_split[world_rank]); i++) {
                    for (int k = kk; k < std::min(kk + BLOCK_SIZE, m); k++) {
                        for (int j = jj; j < std::min(jj + BLOCK_SIZE, l); j++)
                            c_mat[i * l + j] += a_mat[(i + n_offset[world_rank]) * m + k] * b_mat[k * l + j];
                    }
                }
            }
        }
    }

    if (world_rank > 0) {
        MPI_Send(c_mat, n_split[world_rank] * l, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0) {
        MPI_Request requests[world_size - 1];
        MPI_Status stats[world_size - 1];
        int *c_mat_final = new int[n * l];
        memcpy(c_mat_final, c_mat, n_split[world_rank] * l * sizeof(int));

        for (int i = 1; i < world_size; i++)
            MPI_Irecv(c_mat_final + n_offset[i] * l, n_split[i] * l, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);

        MPI_Waitall(world_size - 1, requests, stats);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                printf("%d ", c_mat_final[i * l + j]);
            }
            putchar('\n');
        }

        delete[] c_mat_final;
    }

    delete[] c_mat;
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    delete[] a_mat;
    delete[] b_mat;
}