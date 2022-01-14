#include <cstring>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <nbody/body.hpp>

template <typename... Args> void UNUSED(Args &&...args [[maybe_unused]]) {}

__global__ void cudaCheckBodies(double *x, double *y, double *vx, 
                                double *vy, double *ax, double *ay, 
                                double *m, double *args, const int bodies)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  double elapse = args[0];
  double space = args[1];
  double radius = args[2];
  double COLLISION_RATIO = args[3];
  if (idx < bodies){
    vx[idx] += ax[idx] * elapse;
    vy[idx] += ay[idx] * elapse;

    bool flag = false;
    if (x[idx] <= radius) {
      flag = true;
      x[idx] = radius + radius * COLLISION_RATIO;
      vx[idx] = -vx[idx];
    } else if (x[idx] >= space - radius) {
      flag = true;
      x[idx] = space - radius - radius * COLLISION_RATIO;
      vx[idx] = -vx[idx];
    }

    if (y[idx] <= radius) {
      flag = true;
      y[idx] = radius + radius * COLLISION_RATIO;
      vy[idx] = -vy[idx];
    } else if (y[idx] >= space - radius) {
      flag = true;
      y[idx] = space - radius - radius * COLLISION_RATIO;
      vy[idx] = -vy[idx];
    }
    if (flag) {
      ax[idx] = 0;
      ay[idx] = 0;
    }

    x[idx] += vx[idx] * elapse;
    y[idx] += vy[idx] * elapse;

    flag = false;
    if (x[idx] <= radius) {
      flag = true;
      x[idx] = radius + radius * COLLISION_RATIO;
      vx[idx] = -vx[idx];
    } else if (x[idx] >= space - radius) {
      flag = true;
      x[idx] = space - radius - radius * COLLISION_RATIO;
      vx[idx] = -vx[idx];
    }

    if (y[idx] <= radius) {
      flag = true;
      y[idx] = radius + radius * COLLISION_RATIO;
      vy[idx] = -vy[idx];
    } else if (y[idx] >= space - radius) {
      flag = true;
      y[idx] = space - radius - radius * COLLISION_RATIO;
      vy[idx] = -vy[idx];
    }
    if (flag) {
      ax[idx] = 0;
      ay[idx] = 0;
    }
    
  }
}

int main(int argc, char **argv) {

    // TODO: 
    // - change all attributes into dynamic arrays then send with cudaMalloc
    // - rewrite all functions in device functions
    // - the first nest for loop might not be paralleled as for preventing data racing
    // - so we may only parallel the second for loop
    // 
    // - to prevent data race, we need to investigate co-collision case

  UNUSED(argc, argv);
  static float gravity = 100;
  static float space = 800;
  static float radius = 5;
  static const int bodies = 20;
  static float elapse = 0.1;
  static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
  static float max_mass = 50;
  static const int THREAD_NUMS_PER_BLOCK = 24;

  static float current_space = space;
  static float current_max_mass = max_mass;
  static int current_bodies = bodies;
  BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
  graphic::GraphicContext context{"Assignment 3 CUDA version"};
  context.run([&](graphic::GraphicContext *context [[maybe_unused]],
                  SDL_Window *) {
    auto io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("Assignment 3 CUDA version", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
    ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
    ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
    ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
    ImGui::DragFloat("Elapse", &elapse, 0.1, 0.001, 10, "%f");
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

      // pool.update_for_tick(elapse, gravity, space, radius);

      // allocate host memory space
      // 7 attributes, radius, gravity, elapse
      // cuda dont support STL, we only have thrust vector
      double *x =  new double[bodies];
      double *y =  new double[bodies];
      double *vx = new double[bodies];
      double *vy = new double[bodies];
      double *ax = new double[bodies];
      double *ay = new double[bodies];
      double *m = new double[bodies];
      double *args = new double[4]; // stores elapse, space, radius, collision ratio

      // allocate device memory space
      double *d_x, *d_y, *d_vx, *d_vy, *d_ax, *d_ay, *d_m, *d_args;
      cudaMalloc(&d_x, bodies*sizeof(double)); //d_x means device_x_coordinate
      cudaMalloc(&d_y, bodies*sizeof(double));
      cudaMalloc(&d_vx, bodies*sizeof(double));
      cudaMalloc(&d_vy, bodies*sizeof(double));
      cudaMalloc(&d_ax, bodies*sizeof(double));
      cudaMalloc(&d_ay, bodies*sizeof(double));
      cudaMalloc(&d_m, bodies*sizeof(double));
      cudaMalloc(&d_args, 4*sizeof(double));

      // initialize host data
      pool.ax.assign(pool.size(), 0);
      pool.ay.assign(pool.size(), 0);
      for (size_t i = 0; i < pool.size(); ++i) {
        for (size_t j = i + 1; j < pool.size(); ++j) {
          // update acceleration
          pool.check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
      }
      for (int i = 0; i < bodies; i++){
        x[i] = pool.x[i];
        y[i] = pool.y[i];
        vx[i] = pool.vx[i];
        vy[i] = pool.vy[i];
        ax[i] = pool.ax[i];
        ay[i] = pool.ay[i];
        m[i] = pool.m[i];
      }
      args[0] = elapse;
      args[1] = space;
      args[2] = radius;
      args[3] = pool.COLLISION_RATIO;

      // copy host data to device data

      cudaMemcpy(d_x, x, bodies*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, y, bodies*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vx, vx, bodies*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vy, vy, bodies*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ax, ax, bodies*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ay, ay, bodies*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_m, m, bodies*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_args, args, 4*sizeof(double), cudaMemcpyHostToDevice);

      // call kernel
      cudaCheckBodies<<<(bodies + THREAD_NUMS_PER_BLOCK - 1) / THREAD_NUMS_PER_BLOCK, THREAD_NUMS_PER_BLOCK>>>
      (x, y, vx, vy, ax, ay, m, args, bodies);

      // copy data back to host
      cudaMemcpy(x, d_x, bodies*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, bodies*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(vx, d_vx, bodies*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, d_vy, bodies*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(ax, d_ax, bodies*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(ay, d_ay, bodies*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(m, d_m, bodies*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(args, d_args, 4*sizeof(double), cudaMemcpyDeviceToHost);

      for (int i = 0; i < bodies; i++){
        pool.x[i] = x[i];
        pool.y[i] = y[i];
        pool.vx[i] = vx[i];
        pool.vy[i] = vy[i];
        pool.ax[i] = ax[i];
        pool.ay[i] = ay[i];
        pool.m[i] = m[i];
      }
      

      // for (size_t i = 0; i < pool.size(); ++i) {
      //   // update position and velocity according to acceleration
      //   pool.get_body(i).update_for_tick(elapse, space, radius);
      // }

      // display only needs x and y
      for (size_t i = 0; i < pool.size(); ++i) {
        auto body = pool.get_body(i);
        auto x = p.x + static_cast<float>(body.get_x());
        auto y = p.y + static_cast<float>(body.get_y());
        draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
      }
    }
    ImGui::End();
  });
}
