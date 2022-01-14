#include <chrono>
#include <cstring>
#include <graphic/graphic.hpp>
#include <hdist/hdist.hpp>
#include <imgui_impl_sdl.h>
#include <pthread.h>

template <typename... Args> void UNUSED(Args &&...args [[maybe_unused]]) {}

ImColor temp_to_color(double temp) {
  auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
  return {value, 0, 255 - value};
}

struct Pthread_Arg {
  hdist::Grid *grid;
  hdist::State *state;
  bool *stabilized;
  pthread_mutex_t *lock_on_stablized;
  int *line_remain;
  pthread_mutex_t *lock_on_line_remain;
  int start_index;
};

void *pthreadJacobi(void *argp) {
  struct Pthread_Arg *args = (struct Pthread_Arg *)argp;
  int start_index = args->start_index;
  bool sub_stabilized = true;
  int room_size = (*(args->state)).room_size;

  while (true) {
    for (size_t j = 0; j < room_size; ++j) {
      auto result =
          update_single(start_index, j, *(args->grid), *(args->state));

      sub_stabilized &= result.stable;
      (*(args->grid))[{hdist::alt, start_index, j}] = result.temp;
    }

    // dynamic scheduling
    pthread_mutex_lock(args->lock_on_line_remain);
    if (*(args->line_remain) <= 0) {
      pthread_mutex_unlock(args->lock_on_line_remain);
      pthread_exit(0);
    } else {
      start_index = --*(args->line_remain);
      pthread_mutex_unlock(args->lock_on_line_remain);
    }

    // need to work on the meaning of stablized
    pthread_mutex_lock(args->lock_on_stablized);
    (*(args->stabilized)) &= sub_stabilized;
    pthread_mutex_unlock(args->lock_on_stablized);
  }
}

int main(int argc, char **argv) {
  UNUSED(argc, argv);
  bool first = true;
  bool finished = false;
  static int pthread_nums = 8;

  static hdist::State current_state, last_state;
  static std::chrono::high_resolution_clock::time_point begin, end;
  static const char *algo_list[2] = {"jacobi", "sor"};
  graphic::GraphicContext context{"Assignment 4 - P-Thread Implementation"};
  auto grid = hdist::Grid{static_cast<size_t>(current_state.room_size),
                          current_state.border_temp, current_state.source_temp,
                          static_cast<size_t>(current_state.source_x),
                          static_cast<size_t>(current_state.source_y)};

  context.run([&](graphic::GraphicContext *context [[maybe_unused]],
                  SDL_Window *) {
    auto io = ImGui::GetIO();
    pthread_mutex_t lock_on_line_remain;
    pthread_mutex_t lock_on_stablized;

    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("Assignment 4 - P-Thread Implementation", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600, "%d");
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

    // TODO: there is problem with GUI looping in finished and stablized.

    if (!finished) {

      // finished = hdist::calculate(current_state, grid);

      bool stabilized = true;

      // initialize pthread variables

      int line_remain = current_state.room_size - pthread_nums;

      std::vector<struct Pthread_Arg> argp(pthread_nums);
      std::vector<pthread_t> tids(pthread_nums);

      switch (current_state.algo) {
      case hdist::Algorithm::Jacobi:

        // create child threads
        for (int i = 0; i < pthread_nums; i++) {
          argp[i].state = &current_state;
          argp[i].grid = &grid;
          argp[i].stabilized = &stabilized;
          argp[i].line_remain = &line_remain;
          argp[i].lock_on_line_remain = &lock_on_line_remain;
          argp[i].lock_on_stablized = &lock_on_stablized;
          argp[i].start_index = current_state.room_size - i - 1;

          pthread_attr_t attr;
          pthread_attr_init(&attr);
          pthread_create(&tids[i], &attr, pthreadJacobi, &argp[i]);
        }

        // join child processes
        for (int i = 0; i < pthread_nums; i++) {
          pthread_join(tids[i], NULL);
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
        draw_list->AddRectFilled(
            ImVec2(x, y),
            ImVec2(x + current_state.block_size, y + current_state.block_size),
            color);
        y += current_state.block_size;
      }
      x += current_state.block_size;
      y = p.y + current_state.block_size;
    }
    ImGui::End();
  });
}
