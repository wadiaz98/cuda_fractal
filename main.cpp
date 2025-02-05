#include <iostream>
#include <fmt/core.h>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

unsigned int _bswap32(unsigned int a) {
    return
        ((a & 0X000000FF) << 24) |
        ((a & 0X0000FF00) <<  8) |
        ((a & 0x00FF0000) >>  8) |
        ((a & 0xFF000000) >> 24);
}


#define WIDTH 1600
#define HEIGHT 900

const double x_min = -2;
const double x_max = 1;
const double y_min = -1;
const double y_max = 1;
int maxIteration = 100;

const int PALETTE_SIZE = 16;
const unsigned int color_ramp[] = {
    _bswap32(0xFF1010FF),
    _bswap32(0xEF1019FF),
    _bswap32(0xE01123FF),
    _bswap32(0xD1112DFF),
    _bswap32(0xC11237FF),
    _bswap32(0xB21341FF),
    _bswap32(0xA3134BFF),
    _bswap32(0x931455FF),
    _bswap32(0x84145EFF),
    _bswap32(0x751568FF),
    _bswap32(0x651672FF),
    _bswap32(0x56167CFF),
    _bswap32(0x471786FF),
    _bswap32(0x371790FF),
    _bswap32(0x28189AFF),
    _bswap32(0x1919A4FF)
};

static unsigned int* host_pixel_buffer = nullptr;
static unsigned int* device_pixel_buffer = nullptr;

//Macro: los macros deben ser escritas en una solo linea, caso contrario se usa "\ 'aqui va el codigo' \"
#define CHECK(expr){                    \
    auto error = expr;                  \
    if (error != cudaSuccess) {         \
        fmt::print("{}: {} in {} at line {}\n", (int)error, (char*)cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);             \
    }                                   \
}

unsigned int divergente(double cx, double cy, int max_iterations) {

    int iter = 0;

    double vx = cx;
    double vy = cy;

    while(iter<max_iterations && (vx*vx+vy*vy)<=4) {
        //Zn+1 = Zn^2 + C
        double tx = vx * vx - vy * vy + cx; //vx^2-vy^2+cx
        double ty = 2 * vx * vy + cy; // 2 vx vy + cy

        vx = tx;
        vy = ty;

        iter++;
    }

    if(iter>0 && iter<max_iterations) {
        // diverge
        int color_idx = iter % PALETTE_SIZE;
        return color_ramp[color_idx];
    }

    if((vx*vx+vy*vy)>4) {
        return color_ramp[0];
    }

    // convergente
    return 0;
}

void mandelbrotCpu()
{
    double dx = (x_max-x_min) / WIDTH;
    double dy = (y_max-y_min) / HEIGHT;

    for(int i=0; i<WIDTH; i++) {
        for(int j=0;j<HEIGHT;j++) {
            double x = x_min + i*dx;
            double y = y_max - j*dy;

            // C = X+Yi
            unsigned int color = divergente(x,y, maxIteration);
            host_pixel_buffer[j*WIDTH + i] = color;
        }
    }
}

void mandelbrotCpuOMP()
{
    double dx = (x_max-x_min) / WIDTH;
    double dy = (y_max-y_min) / HEIGHT;

#pragma omp parallel for default(none) shared(host_pixel_buffer, dx, dy, color_ramp, maxIteration)
    for(int i=0; i<WIDTH; i++) {
        for(int j=0;j<HEIGHT;j++) {
            double x = x_min + i*dx;
            double y = y_max - j*dy;

            // C = X+Yi
            unsigned int color = divergente(x,y, maxIteration);
            host_pixel_buffer[j*WIDTH + i] = color;
        }
    }
}

int main()
{

    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    fmt::println("Device: {}", props.name);
    fmt::println(" Total global memory: {}GB", props.totalGlobalMem);
    fmt::println("  Multiprocessors: {}", props.multiProcessorCount);
    fmt::println("  Max threads per multiprocessors: {}", props.maxThreadsPerMultiProcessor);
    fmt::println("  Max threads per block: {}", props.maxThreadsPerBlock);

    //---inicializar
    size_t buffer_size = WIDTH * HEIGHT * sizeof(unsigned int);

    host_pixel_buffer = (unsigned int*)malloc(buffer_size);

    // inicializar a 0 todo el buffer para que no se llene de bsaura
    memset(host_pixel_buffer, 0, buffer_size);

    CHECK(cudaMalloc(&device_pixel_buffer, buffer_size));

    //--Inicializacion de la interfaz de la grafica
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Mandelbort CUDA");

    mandelbrotCpu();

    //--textura
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);
    texture.update((const sf::Uint8*)host_pixel_buffer);


    sf::Sprite sprite;
    sprite.setTexture(texture);

    //--para textos
    sf::Font font;
    font.loadFromFile("arial.ttf");

    sf::Text text;
    {
        text.setFont(font);
        text.setString("Mandelbrot Set");
        text.setCharacterSize(24);
        text.setFillColor(sf::Color::White);
        text.setStyle(sf::Text::Bold);
        text.setPosition(10, 10);
    }

    //--FPS
    sf::Clock clock;
    int frames = 0;
    int fps = 0;


    while (window.isOpen())
    {
        // Process events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Close window: exit
            if (event.type == sf::Event::Closed)
                window.close();
        }

        //--Regenerar la imagen
        mandelbrotCpu();
        texture.update((const sf::Uint8*)host_pixel_buffer);

        //--contar FPS
        frames++;
        if (clock.getElapsedTime().asSeconds()>=1)
        {
            fps = frames;
            frames = 0;
            clock.restart();
        }

        auto msg = fmt::format("Mandelbrot Set, Iterations={}, FPS: {}", maxIteration, fps);

        text.setString(msg);

        // Clear screen
        window.clear();
        {
            window.draw(sprite);
            window.draw(text);
        }

        // Update the window
        window.display();
    }


    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.