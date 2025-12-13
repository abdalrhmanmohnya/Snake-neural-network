# Snake-neural-network
Teaching a clanker to play Snake

# Running it yourself
The install location for pytorch C++ was really janky, so this MIGHT NOT WORK for other people's machines. It was also built on linux & probably won't work on windows/mac. 
```
git clone https://github.com/Ethan-Blesch/Snake-neural-network
cd Snake-neural-network
g++ -std=c++17 thread.cpp -o snake_rl -I/usr/local/libtorch/include -I/usr/local/libtorch/include/torch/csrc/api/include -L/usr/local/libtorch/lib -D_GLIBCXX_USE_CXX11_ABI=0 -ltorch -lc10 -lpthread -Wl,-rpath,/usr/local/libtorch/lib $(sdl2-config --cflags --libs) -w
./snake_rl
```
