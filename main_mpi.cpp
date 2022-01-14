#include <cstring>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <mpi.h>
#include <nbody/body.hpp>
#include <omp.h>

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_tag = 0;
  int stop_tag = 1;
  int stop_flag = 0;

  static float gravity = 100;
  static float space = 800;
  static float radius = 5;
  static const int bodies = 200;
  static float elapse = 0.05; // manually set
  static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
  static float max_mass = 50;
  static float current_space = space;
  static float current_max_mass = max_mass;
  static int current_bodies = bodies;

  // struct buffer to send the whole struct
  struct My_Buffer {
    double x[bodies];
    double y[bodies];
    double vx[bodies];
    double vy[bodies];
    double ax[bodies];
    double ay[bodies];
    double m[bodies];
  };

  MPI_Datatype MPI_Pool;
  MPI_Datatype type_list[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                               MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
  int block_lens[7] = {bodies, bodies, bodies, bodies, bodies, bodies, bodies};
  MPI_Aint offsets[7];
  offsets[0] = offsetof(My_Buffer, x);
  offsets[1] = offsetof(My_Buffer, y);
  offsets[2] = offsetof(My_Buffer, vx);
  offsets[3] = offsetof(My_Buffer, vy);
  offsets[4] = offsetof(My_Buffer, ax);
  offsets[5] = offsetof(My_Buffer, ay);
  offsets[6] = offsetof(My_Buffer, m);
  MPI_Type_create_struct(7, block_lens, offsets, type_list, &MPI_Pool);
  MPI_Type_commit(&MPI_Pool);

  size_t sub_size = bodies / mpi_size;
  size_t start_index = mpi_rank * sub_size;

  BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
  struct My_Buffer buffer;

  if (mpi_rank == 0) {
    graphic::GraphicContext context{"Assignment 3 MPI Version"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]],
                    SDL_Window *) {
      auto io = ImGui::GetIO();
      ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
      ImGui::SetNextWindowSize(io.DisplaySize);
      ImGui::Begin("Assignment 3 MPI Version", nullptr,
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                       ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
      ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
      ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
      ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
      ImGui::DragFloat("Elapse", &elapse, 0.05, 0.001, 10, "%f");
      ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
      ImGui::ColorEdit4("Color", &color.x);
      if (current_space != space || current_bodies != bodies ||
          current_max_mass != max_mass) {
        space = current_space;
        // bodies = current_bodies;
        max_mass = current_max_mass;
        pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
      }
      {
        const ImVec2 p = ImGui::GetCursorScreenPos();

        for (int i = 1; i < mpi_size; i++) {
          MPI_Send(&stop_flag, 1, MPI_INT, i, stop_tag, MPI_COMM_WORLD);
        }

        // pool.update_for_tick(elapse, gravity, space, radius);
        pool.ax.assign(pool.size(), 0);
        pool.ay.assign(pool.size(), 0);

        for (int i = 0; i < bodies; i++) {
          buffer.x[i] = pool.x[i];
          buffer.y[i] = pool.y[i];
          buffer.vx[i] = pool.vx[i];
          buffer.vy[i] = pool.vy[i];
          buffer.ax[i] = pool.ax[i];
          buffer.ay[i] = pool.ay[i];
          buffer.m[i] = pool.m[i];
        }

        for (int i = 1; i < mpi_size; i++) {
          MPI_Send(&buffer, 1, MPI_Pool, i, mpi_tag, MPI_COMM_WORLD);
        }

        for (size_t i = start_index; i < start_index + sub_size; ++i) {
          for (size_t j = i + 1; j < pool.size(); ++j) {
            // update acceleration
            pool.check_and_update(pool.get_body(i), pool.get_body(j), radius,
                                  gravity);
          }
        }

        for (size_t i = start_index; i < start_index + sub_size; ++i) {
          // update position and velocity according to acceleration
          pool.get_body(i).update_for_tick(elapse, space, radius);
        }

        for (int i = 1; i < mpi_size; i++) {
          MPI_Recv(&buffer, 1, MPI_Pool, i, mpi_tag, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          for (size_t j = i * sub_size; j < i * sub_size + sub_size; j++) {
            pool.x[j] = buffer.x[j];
            pool.y[j] = buffer.y[j];
            pool.vx[j] = buffer.vx[j];
            pool.vy[j] = buffer.vy[j];
            pool.ax[j] = buffer.ax[j];
            pool.ay[j] = buffer.ay[j];
            pool.m[j] = buffer.m[j];
          }
        }

        for (size_t i = 0; i < pool.size(); ++i) {
          auto body = pool.get_body(i);
          auto x = p.x + static_cast<float>(body.get_x());
          auto y = p.y + static_cast<float>(body.get_y());
          draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
        }
      }
      ImGui::End();

      if (context->finished) {
        stop_flag = 1;
        for (int i = 1; i < mpi_size; i++) {
          MPI_Send(&stop_flag, 1, MPI_INT, i, stop_tag, MPI_COMM_WORLD);
        }
      }
    });
  }

  // Else, for child Processes
  else {
    while (true) {

      MPI_Recv(&stop_flag, 1, MPI_INT, 0, stop_tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      if (stop_flag == 1) {
        break;
      }

      MPI_Recv(&buffer, 1, MPI_Pool, 0, mpi_tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      for (int i = 0; i < bodies; i++) {
        pool.x[i] = buffer.x[i];
        pool.y[i] = buffer.y[i];
        pool.vx[i] = buffer.vx[i];
        pool.vy[i] = buffer.vy[i];
        pool.ax[i] = buffer.ax[i];
        pool.ay[i] = buffer.ay[i];
        pool.m[i] = buffer.m[i];
      }

      for (size_t i = start_index; i < start_index + sub_size; ++i) {
        for (size_t j = i + 1; j < pool.size(); ++j) {
          // update acceleration
          pool.check_and_update(pool.get_body(i), pool.get_body(j), radius,
                                gravity);
        }
      }

      for (size_t i = start_index; i < start_index + sub_size; ++i) {
        // update position and velocity according to acceleration
        pool.get_body(i).update_for_tick(elapse, space, radius);
      }

      for (size_t i = start_index; i < start_index + sub_size; i++) {
        buffer.x[i] = pool.x[i];
        buffer.y[i] = pool.y[i];
        buffer.vx[i] = pool.vx[i];
        buffer.vy[i] = pool.vy[i];
        buffer.ax[i] = pool.ax[i];
        buffer.ay[i] = pool.ay[i];
        buffer.m[i] = pool.m[i];
      }

      MPI_Send(&buffer, 1, MPI_Pool, 0, mpi_tag, MPI_COMM_WORLD);
    }
  }
  MPI_Type_free(&MPI_Pool);
  MPI_Finalize();
}
