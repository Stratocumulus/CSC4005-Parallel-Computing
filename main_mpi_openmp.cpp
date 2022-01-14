#include <chrono>
#include <cstring>
#include <graphic/graphic.hpp>
#include <hdist/hdist.hpp>
#include <imgui_impl_sdl.h>
#include <mpi.h>

template <typename... Args> void UNUSED(Args &&...args [[maybe_unused]]) {}

ImColor temp_to_color(double temp) {
  auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
  return {value, 0, 255 - value};
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_tag = 0;
  int stop_tag = 1;
  int stop_flag = 0;
  int sub_size = 300 / mpi_size;
  int start_index = mpi_rank * sub_size;

  bool first = true;
  bool finished = false;
  static hdist::State current_state, last_state;

  auto grid = hdist::Grid{static_cast<size_t>(current_state.room_size),
                          current_state.border_temp, current_state.source_temp,
                          static_cast<size_t>(current_state.source_x),
                          static_cast<size_t>(current_state.source_y)};

  if (mpi_rank == 0) {

    static std::chrono::high_resolution_clock::time_point begin, end;
    static const char *algo_list[2] = {"jacobi", "sor"};
    graphic::GraphicContext context{"Assignment 4 - MPI Implementation"};

    context.run([&](graphic::GraphicContext *context [[maybe_unused]],
                    SDL_Window *) {
      auto io = ImGui::GetIO();
      ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
      ImGui::SetNextWindowSize(io.DisplaySize);
      ImGui::Begin("Assignment 4 - MPI Implementation", nullptr,
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                       ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600,
                     "%d");
      ImGui::DragFloat("Block Size", &current_state.block_size, 0.01, 0.1, 10,
                       "%f");
      ImGui::DragFloat("Source Temp", &current_state.source_temp, 0.1, 0, 100,
                       "%f");
      ImGui::DragFloat("Border Temp", &current_state.border_temp, 0.1, 0, 100,
                       "%f");
      ImGui::DragInt("Source X", &current_state.source_x, 1, 1,
                     current_state.room_size - 2, "%d");
      ImGui::DragInt("Source Y", &current_state.source_y, 1, 1,
                     current_state.room_size - 2, "%d");
      ImGui::DragFloat("Tolerance", &current_state.tolerance, 0.01, 0.01, 1,
                       "%f");
      ImGui::ListBox("Algorithm", reinterpret_cast<int *>(&current_state.algo),
                     algo_list, 2);

      if (current_state.algo == hdist::Algorithm::Sor) {
        ImGui::DragFloat("Sor Constant", &current_state.sor_constant, 0.01, 0.0,
                         20.0, "%f");
      }

      if (current_state.room_size != last_state.room_size) {
        grid = hdist::Grid{static_cast<size_t>(current_state.room_size),
                           current_state.border_temp, current_state.source_temp,
                           static_cast<size_t>(current_state.source_x),
                           static_cast<size_t>(current_state.source_y)};
        first = true;
      }

      if (current_state != last_state) {
        last_state = current_state;
        finished = false;
      }

      if (first) {
        first = false;
        finished = false;
        begin = std::chrono::high_resolution_clock::now();
      }

      // control child processes
      for (int i = 1; i < mpi_size; i++) {
        MPI_Send(&stop_flag, 1, MPI_INT, i, stop_tag, MPI_COMM_WORLD);
      }

      // calculate temp
      if (!finished) {

        // finished = hdist::calculate(current_state, grid);

        bool stabilized = true;
        int buffer_size = current_state.room_size * current_state.room_size;
        std::vector<double> buffer = grid.get_current_buffer();

        switch (current_state.algo) {
        case hdist::Algorithm::Jacobi:

          // send grid to child process
          for (int i = 1; i < mpi_size; i++) {
            MPI_Send(&buffer[0], buffer_size, MPI_DOUBLE, i, mpi_tag,
                     MPI_COMM_WORLD);
          }

// update temp
#pragma omp parallel for collapse(2) num_threads(4)
          for (size_t i = start_index; i < start_index + sub_size; ++i) {
            for (size_t j = 0; j < current_state.room_size; ++j) {

              auto result = update_single(i, j, grid, current_state);
#pragma omp single
              stabilized &= result.stable;
              grid[{hdist::alt, i, j}] = result.temp;
            }
          }

          //  recv from child process
          for (int i = 1; i < mpi_size; i++) {
            MPI_Recv(&buffer[0], buffer_size, MPI_DOUBLE, i, mpi_tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for collapse(2) num_threads(4)
            for (int x = i * sub_size; x < i * sub_size + sub_size; ++x) {
              for (int y = 0; y < current_state.room_size; ++y) {
                grid[{hdist::alt, x, y}] =
                    buffer[x * current_state.room_size + y];
              }
            }
          }
          grid.switch_buffer();
          finished = stabilized;
          break;

        case hdist::Algorithm::Sor:
          for (auto k : {0, 1}) {
            for (size_t i = 0; i < current_state.room_size; i++) {
              for (size_t j = 0; j < current_state.room_size; j++) {
                if (k == ((i + j) & 1)) {
                  auto result = update_single(i, j, grid, current_state);
                  stabilized &= result.stable;
                  grid[{hdist::alt, i, j}] = result.temp;
                } else {
                  grid[{hdist::alt, i, j}] = grid[{i, j}];
                }
              }
            }
            grid.switch_buffer();
          }
          finished = stabilized;
        }

        if (finished)
          end = std::chrono::high_resolution_clock::now();
      } else {
        ImGui::Text(
            "stabilized in %ld ns",
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count());
      }

      const ImVec2 p = ImGui::GetCursorScreenPos();
      float x = p.x + current_state.block_size,
            y = p.y + current_state.block_size;
      for (size_t i = 0; i < current_state.room_size; ++i) {
        for (size_t j = 0; j < current_state.room_size; ++j) {
          auto temp = grid[{i, j}];
          auto color = temp_to_color(temp);
          draw_list->AddRectFilled(ImVec2(x, y),
                                   ImVec2(x + current_state.block_size,
                                          y + current_state.block_size),
                                   color);
          y += current_state.block_size;
        }
        x += current_state.block_size;
        y = p.y + current_state.block_size;
      }
      ImGui::End();

      // close child processes
      if (context->finished) {
        stop_flag = 1;
        for (int i = 1; i < mpi_size; i++) {
          MPI_Send(&stop_flag, 1, MPI_INT, i, stop_tag, MPI_COMM_WORLD);
        }
      }
    });
  }

  else {
    while (true) {
      MPI_Recv(&stop_flag, 1, MPI_INT, 0, stop_tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      if (stop_flag == 1) {
        break;
      }

      bool stabilized = true;
      int buffer_size = current_state.room_size * current_state.room_size;
      std::vector<double> buffer = grid.get_current_buffer();

      // receive grid from the root process
      MPI_Recv(&buffer[0], buffer_size, MPI_DOUBLE, 0, mpi_tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      grid.get_current_buffer() = buffer;

#pragma omp parallel for collapse(2) num_threads(4)
      for (size_t i = start_index; i < start_index + sub_size; ++i) {
        for (size_t j = 0; j < current_state.room_size; ++j) {

          auto result = update_single(i, j, grid, current_state);
#pragma omp single
          stabilized &= result.stable;
          grid[{hdist::alt, i, j}] = result.temp;
        }
      }
      grid.switch_buffer();

      // send grid back to mother process
      buffer = grid.get_current_buffer();
      MPI_Send(&buffer[0], buffer_size, MPI_DOUBLE, 0, mpi_tag, MPI_COMM_WORLD);
    }
  }
  MPI_Finalize();
}
