#include <iostream>
#include <mpi.h>
#include <odd-even-sort.hpp>
#include <vector>

namespace sort {
using namespace std::chrono;

Context::Context(int &argc, char **&argv) : argc(argc), argv(argv) {
  MPI_Init(&argc, &argv);
}

Context::~Context() { MPI_Finalize(); }

std::unique_ptr<Information> Context::mpi_sort(Element *begin,
                                               Element *end) const {
  int res;
  int rank;
  std::unique_ptr<Information> information{};

  res = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (MPI_SUCCESS != res) {
    throw std::runtime_error("failed to get MPI world rank");
  }

  if (0 == rank) {
    information = std::make_unique<Information>();
    information->length = end - begin;
    res = MPI_Comm_size(MPI_COMM_WORLD, &information->num_of_proc);
    if (MPI_SUCCESS != res) {
      throw std::runtime_error("failed to get MPI world size");
    };
    information->argc = argc;
    for (auto i = 0; i < argc; ++i) {
      information->argv.push_back(argv[i]);
    }
    information->start = high_resolution_clock::now();
  }

  {
    /// now starts the main sorting procedure
    /// @todo: please modify the following code

    // Initialize Local array for each process
    int num_of_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc);
    int global_length;
    if (0 == rank) {
      global_length = end - begin;
    }
    MPI_Bcast(&global_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // deal with the case that datasize < process amount:
    if (global_length < num_of_proc) {
      if (0 == rank) {
        std::sort(begin, end);
        information->end = high_resolution_clock::now();
      }
      return information;
    }

    // ***********************************
    // Parallel Odd-Even Sort:
    // Initialize varibles for the bubbling loop
    int local_length = global_length / num_of_proc; // local array length
    Element *local_array = (Element *)malloc(
        sizeof(Element) * local_length); // initialize local array
    // Scatter input data into each process
    MPI_Scatter(begin, local_length, MPI_LONG, local_array, local_length,
                MPI_LONG, 0, MPI_COMM_WORLD);

    // Do Odd-and-Even Sort
    Element local_buff;

    for (int idx = 0; idx < global_length; ++idx) {

      if (idx % 2 == 0) {
        // Even-Rank Process Inner Bubbling
        for (int loc_idx = local_length - 1; loc_idx > 0; loc_idx -= 2) {
          if (local_array[loc_idx - 1] > local_array[loc_idx]) {
            std::swap(local_array[loc_idx - 1], local_array[loc_idx]);
          }
        }
      }

      else {
        // Odd_rank Process Inner Bubbling
        for (int loc_idx = local_length - 2; loc_idx > 0; loc_idx -= 2) {
          if (local_array[loc_idx - 1] > local_array[loc_idx]) {
            std::swap(local_array[loc_idx - 1], local_array[loc_idx]);
          }
        }

        // Odd-Phase Communication
        if (rank % 2 == 1) {
          local_buff = local_array[0];
          MPI_Send(&local_buff, 1, MPI_LONG, rank - 1, 0, MPI_COMM_WORLD);
          MPI_Recv(&local_buff, 1, MPI_LONG, rank - 1, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          if (local_buff > local_array[0]) {
            std::swap(local_buff, local_array[0]);
          }
        } else {
          if (rank != num_of_proc - 1) {
            MPI_Recv(&local_buff, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            if (local_buff < local_array[local_length - 1]) {
              std::swap(local_buff, local_array[local_length - 1]);
            }
            MPI_Send(&local_buff, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD);
          }
        }

        // Even-Phase Communication
        if (rank % 2 == 0) {
          local_buff = local_array[0];
          int target_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
          MPI_Send(&local_buff, 1, MPI_LONG, target_rank, 0, MPI_COMM_WORLD);
          MPI_Recv(&local_buff, 1, MPI_LONG, target_rank, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          if (local_buff > local_array[0]) {
            std::swap(local_buff, local_array[0]);
          }
        } else {
          // Rationale: MUST make sure MPI_Recv will recv a value or local_buff
          // is the original local array value
          if (rank != num_of_proc - 1) {
            MPI_Recv(&local_buff, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            if (local_buff < local_array[local_length - 1]) {
              std::swap(local_buff, local_array[local_length - 1]);
            }

            MPI_Send(&local_buff, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD);
          }
        }
      }
    }

    // Gather local_array back to global array
    MPI_Gather(local_array, local_length, MPI_LONG, begin, local_length,
               MPI_LONG, 0, MPI_COMM_WORLD);
    // ***********************************

    // // ***********************************
    // To run the sequential version, just de-comment this part and then comment
    // the parallel part
    // // ***********************************
    // // Sequantial Odd-Even Sort
    // if (0 == rank) {
    //     bool is_sorted = false;
    //     while (!is_sorted){
    //         is_sorted = true;

    //         for (int i = 1; i <= global_length - 2; i = i + 2){
    //             if (begin[i] > begin[i+1]){
    //                 std::swap(begin[i], begin[i+1]);
    //                 is_sorted = false;
    //             }
    //         }

    //         for (int i = 0; i <= global_length - 2; i = i + 2){
    //             if (begin[i] > begin[i+1]){
    //                 std::swap(begin[i], begin[i+1]);
    //                 is_sorted = false;
    //             }
    //         }

    //     }
    // }
    // // ***********************************

    // Residualization Strategy: sort residuals into the original array
    if (0 == rank) {
      int residual = global_length % num_of_proc;

      while (residual != 0) {
        auto iter = residual;
        while (begin[global_length - iter] < begin[global_length - iter - 1]) {
          std::swap(begin[global_length - iter],
                    begin[global_length - iter - 1]);
          ++iter;
          if (iter == global_length)
            break;
        }
        --residual;
      }
    }
  }

  if (0 == rank) {
    information->end = high_resolution_clock::now();
  }
  return information;
}

std::ostream &Context::print_information(const Information &info,
                                         std::ostream &output) {
  auto duration = info.end - info.start;
  auto duration_count = duration_cast<nanoseconds>(duration).count();
  auto mem_size = static_cast<double>(info.length) * sizeof(Element) / 1024.0 /
                  1024.0 / 1024.0;
  output << "input size: " << info.length << std::endl;
  output << "proc number: " << info.num_of_proc << std::endl;
  output << "duration (ns): " << duration_count << std::endl;
  output << "throughput (gb/s): "
         << mem_size / static_cast<double>(duration_count) * 1'000'000'000.0
         << std::endl;
  return output;
}
} // namespace sort
