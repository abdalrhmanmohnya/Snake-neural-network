#include <SDL2/SDL.h>
#include <deque>
#include <random>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <torch/torch.h>
#include <csignal>

volatile std::sig_atomic_t force_render_next = 0;

void signal_handler(int signum){
    force_render_next = 1;
}


using vec_t = std::vector<float>;

const int CELL = 20;
const int COLS = 20;
const int ROWS = 20;
const int WINDOW_W = COLS * CELL;
const int WINDOW_H = ROWS * CELL;
const int TICKS_PER_SECOND = 1000;
const int INPUT_SIZE = 12;
const int HIDDEN_SIZE = 64;
const int OUTPUT_SIZE = 4;
const float LEARNING_RATE = 0.0003f;
const float GAMMA = 0.95f;
const float EPSILON_START = 1.0f;
const float EPSILON_END = 0.01f;
const float EPSILON_DECAY = 0.995f;
const int REPLAY_BUFFER_SIZE = 10000;
const int BATCH_SIZE = 64;
const int TRAIN_EVERY = 4;

enum class Dir { UP=-1, DOWN=1, LEFT=-2, RIGHT=2 };

struct Point {
    int x, y;
    bool operator==(Point const& o) const { return x==o.x && y==o.y; }
};

struct Experience {
    vec_t state;
    int action;
    float reward;
    vec_t next_state;
    bool done;
};

struct QNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    QNetImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(INPUT_SIZE, HIDDEN_SIZE));
        fc2 = register_module("fc2", torch::nn::Linear(HIDDEN_SIZE, HIDDEN_SIZE));
        fc3 = register_module("fc3", torch::nn::Linear(HIDDEN_SIZE, OUTPUT_SIZE));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(QNet);

class SnakeGame {
public:
    SnakeGame(bool render);
    ~SnakeGame();
    bool init();
    void train(int episodes);

private:
    void reset();
    int serializeDirection(const int signedDir) const;
    int unpackDirection(const int unsignedDir) const;
    float step(int action, bool shouldPrint);
    void render();
    void render_network();
    void render_network_dynamic(const vec_t& state);
    void spawn_food();
    bool inside_grid(const Point &p) const;
    bool snake_contains(const Point &p) const;
    vec_t get_state() const;
    int select_action(const vec_t& state);
    void train_step();

    SDL_Window* window=nullptr;
    SDL_Renderer* renderer=nullptr;
    SDL_Window* net_window=nullptr;
    SDL_Renderer* net_renderer=nullptr;
    SDL_Window* dynamic_window=nullptr;
    SDL_Renderer* dynamic_renderer=nullptr;

    std::deque<Point> snake;
    Dir dir = Dir::RIGHT;
    Point food{0,0};
    bool game_over=false;
    int score=0;
    int steps_without_food=0;

    std::mt19937 rng;

    QNet q_network;
    QNet target_network;
    torch::optim::Adam optimizer;

    std::deque<Experience> replay_buffer;
    float epsilon = EPSILON_START;
    bool render_mode;

    int episode_count=0;
    int total_steps=0;
};

SnakeGame::SnakeGame(bool render)
: render_mode(render),
  q_network(QNet()),
  target_network(QNet()),
  optimizer(q_network->parameters(), torch::optim::AdamOptions(LEARNING_RATE))
{
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());

    torch::NoGradGuard no_grad;
    for(size_t i=0;i<q_network->parameters().size();i++){
        target_network->parameters()[i].copy_(q_network->parameters()[i]);
    }
}

SnakeGame::~SnakeGame(){
    if(dynamic_renderer) SDL_DestroyRenderer(dynamic_renderer);
    if(dynamic_window) SDL_DestroyWindow(dynamic_window);
    if(net_renderer) SDL_DestroyRenderer(net_renderer);
    if(net_window) SDL_DestroyWindow(net_window);
    if(renderer) SDL_DestroyRenderer(renderer);
    if(window) SDL_DestroyWindow(window);
    if(render_mode) SDL_Quit();
}

bool SnakeGame::init(){
    if(render_mode){
        if(SDL_Init(SDL_INIT_VIDEO)!=0){
            std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
            return false;
        }
        window = SDL_CreateWindow("Snake AI Training",
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);

        net_window = SDL_CreateWindow("Q-Network Visualization",
                                      SDL_WINDOWPOS_CENTERED+WINDOW_W+50,
                                      SDL_WINDOWPOS_CENTERED,
                                      800, 600, SDL_WINDOW_SHOWN);
        net_renderer = SDL_CreateRenderer(net_window,-1,SDL_RENDERER_ACCELERATED);

        dynamic_window = SDL_CreateWindow("Dynamic Q-Network Activity",
                                          SDL_WINDOWPOS_CENTERED+WINDOW_W+50,
                                          SDL_WINDOWPOS_CENTERED+650,
                                          800, 600, SDL_WINDOW_SHOWN);
        dynamic_renderer = SDL_CreateRenderer(dynamic_window,-1,SDL_RENDERER_ACCELERATED);
    }

    reset();
    return true;
}

void SnakeGame::reset(){
    snake.clear();
    int x=COLS/2, y=ROWS/2;
    snake.push_back({x,y});
    snake.push_back({x-1,y});
    snake.push_back({x-2,y});
    dir=Dir::RIGHT;
    score=0;
    game_over=false;
    steps_without_food=0;
    spawn_food();
}

void SnakeGame::spawn_food(){
    std::uniform_int_distribution<int> dx(0,COLS-1), dy(0,ROWS-1);
    Point p;
    do{
        p.x=dx(rng);
        p.y=dy(rng);
    }while(snake_contains(p));
    food=p;
}


int SnakeGame::serializeDirection(const int signedDir) const{
    switch(signedDir){
        case -1:
            return 0;
        case 1: 
            return 1;
        case -2:
            return 2;
        case 2:
            return 3;
    }
}
int SnakeGame::unpackDirection(const int unsignedDir) const{
    switch(unsignedDir){
        case 0:
            return -1;
        case 1: 
            return 1;
        case 2:
            return -2;
        case 3:
            return 2;
    }
}

bool SnakeGame::inside_grid(const Point&p) const{
    return p.x>=0 && p.x<COLS && p.y>=0 && p.y<ROWS;
}

bool SnakeGame::snake_contains(const Point&p) const{
    for(auto&s:snake) if(s==p) return true;
    return false;
}

vec_t SnakeGame::get_state() const{
    vec_t s(INPUT_SIZE,0.0f);
    Point h=snake.front();

    int du=0; for(int dy=1;dy<=h.y;dy++){Point c{h.x,h.y-dy};du=dy; if(snake_contains(c)) break;} s[0]=du/(float)ROWS;
    int dd=0; for(int dy=1;dy<ROWS-h.y;dy++){Point c{h.x,h.y+dy};dd=dy; if(snake_contains(c)) break;} s[1]=dd/(float)ROWS;
    int dl=0; for(int dx=1;dx<=h.x;dx++){Point c{h.x-dx,h.y};dl=dx; if(snake_contains(c)) break;} s[2]=dl/(float)COLS;
    int dr=0; for(int dx=1;dx<COLS-h.x;dx++){Point c{h.x+dx,h.y};dr=dx; if(snake_contains(c)) break;} s[3]=dr/(float)COLS;

    s[4] = (food.y<h.y && food.x==h.x)? (h.y-food.y)/(float)ROWS:0;
    s[5] = (food.y>h.y && food.x==h.x)? (food.y-h.y)/(float)ROWS:0;
    s[6] = (food.x<h.x && food.y==h.y)? (h.x-food.x)/(float)COLS:0;
    s[7] = (food.x>h.x && food.y==h.y)? (food.x-h.x)/(float)COLS:0;

    s[8+SnakeGame::serializeDirection((int)dir)]=1.0f;
    return s;
}




int SnakeGame::select_action(const vec_t& state){
    std::uniform_real_distribution<float> d(0.0f,1.0f);
    if(d(rng)<epsilon){std::uniform_int_distribution<int>a(0,3); return a(rng);}    
    torch::Tensor inp = torch::tensor(state).reshape({1,INPUT_SIZE});
    torch::Tensor out = q_network->forward(inp);
    auto max_idx = out.argmax(1).item<int>();
    return max_idx;
}

float SnakeGame::step(int action, bool shouldPrint){
    if(game_over) return 0;
    Dir old = dir;
    dir=(Dir)SnakeGame::unpackDirection(action);
    Point h=snake.front(); Point nh=h;

    if((int)old + (int)dir == 0){
        dir = old;
    }

    if(dir==Dir::UP) nh.y--; else if(dir==Dir::DOWN) nh.y++; else if(dir==Dir::LEFT) nh.x--; else nh.x++;

    float reward=-0.01f;
    steps_without_food++;

    if(!inside_grid(nh)){
        if(shouldPrint){
            std::cout << "Hit wall" << "\n";
        }
        game_over=true; return -1.0f;
        
    } 
    if(snake_contains(nh)){
        if(shouldPrint){
            std::cout << "Hit self" << "\n";
        }
        game_over=true; return -1.0f;

    }

    if(steps_without_food > 2000){
        game_over = true; return -0.5f;
    }

    bool ate=(nh==food);
    snake.push_front(nh);
    if(!ate) snake.pop_back(); else{
        reward=1.0f; score++;  spawn_food();
    }

    return reward;
}

void SnakeGame::train_step(){
    if(replay_buffer.size()<BATCH_SIZE) return;
    std::uniform_int_distribution<size_t>d(0,replay_buffer.size()-1);

    std::vector<torch::Tensor> states, targets;
    for(int i=0;i<BATCH_SIZE;i++){
        const auto &e=replay_buffer[d(rng)];
        torch::Tensor s = torch::tensor(e.state);
        torch::Tensor q = q_network->forward(s);
        torch::Tensor t = q.detach().clone();  // <-- detach here!
        
        if(e.done) 
            t[e.action] = e.reward;
        else{
            torch::Tensor ns = torch::tensor(e.next_state);
            auto nq = target_network->forward(ns);
            float mx = nq.max().item<float>();
            t[e.action] = e.reward + GAMMA*mx;
        }
        states.push_back(s);
        targets.push_back(t);
    }

    torch::Tensor batch_s = torch::stack(states);
    torch::Tensor batch_t = torch::stack(targets);

    optimizer.zero_grad();
    torch::Tensor out = q_network->forward(batch_s);
    torch::Tensor loss = torch::mse_loss(out, batch_t);
    loss.backward();
    optimizer.step();
    static int c=0; c++;
    if(c%50==0){
        torch::NoGradGuard no_grad;
        for(size_t i=0;i<q_network->parameters().size();i++){
            target_network->parameters()[i].copy_(q_network->parameters()[i]);
        }
    }
}

void SnakeGame::render(){
    if(!render_mode) return;
    SDL_SetRenderDrawColor(renderer,25,25,25,255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer,40,40,40,255);
    for(int gx=0;gx<=COLS;gx++){ SDL_Rect r{gx*CELL,0,1,WINDOW_H}; SDL_RenderFillRect(renderer,&r);}    
    for(int gy=0;gy<=ROWS;gy++){ SDL_Rect r{0,gy*CELL,WINDOW_W,1}; SDL_RenderFillRect(renderer,&r);}    

    SDL_Rect fr{food.x*CELL+2,food.y*CELL+2,CELL-4,CELL-4};
    SDL_SetRenderDrawColor(renderer,200,50,50,255);
    SDL_RenderFillRect(renderer,&fr);

    bool first=true;
    for(auto&s:snake){SDL_Rect r{s.x*CELL+1,s.y*CELL+1,CELL-2,CELL-2};
        if(first){SDL_SetRenderDrawColor(renderer,90,200,90,255); first=false;} else SDL_SetRenderDrawColor(renderer,30,160,30,255);
        SDL_RenderFillRect(renderer,&r);
    }
    SDL_RenderPresent(renderer);
}

void SnakeGame::render_network(){
    if(!render_mode) return;

    SDL_SetRenderDrawColor(net_renderer,0,0,0,255);
    SDL_RenderClear(net_renderer);

    const int win_w = 800;
    const int win_h = 600;
    const int padding_x = 40;
    const int padding_y = 25;

    const int layers[] = {INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    const int num_layers = 4;

    std::vector<std::vector<SDL_Point>> neuron_pos(num_layers);
    for(int l=0;l<num_layers;l++){
        int n = layers[l];
        neuron_pos[l].resize(n);
        for(int i=0;i<n;i++){
            neuron_pos[l][i] = { padding_x + l*(win_w - 2*padding_x)/(num_layers-1),
                                 padding_y + i*(win_h - 2*padding_y)/(n-1) };
        }
    }

    auto draw_layer_connections = [&](torch::nn::Linear layer, int l){
        auto w = layer->weight.detach();
        int in = w.size(1), out = w.size(0);
        for(int i=0;i<out;i++){
            for(int j=0;j<in;j++){
                float val = w[i][j].item<float>();
                int r = val>0 ? std::min(255,int(val*255)) : 0;
                int g = val<0 ? std::min(255,int(-val*255)) : 0;
                SDL_SetRenderDrawColor(net_renderer, r, g, 0, 255);
                SDL_RenderDrawLine(net_renderer,
                                   neuron_pos[l][j].x, neuron_pos[l][j].y,
                                   neuron_pos[l+1][i].x, neuron_pos[l+1][i].y);
            }
        }
    };

    draw_layer_connections(q_network->fc1, 0);
    draw_layer_connections(q_network->fc2, 1);
    draw_layer_connections(q_network->fc3, 2);

    for(int l=0;l<num_layers;l++){
        for(auto &p: neuron_pos[l]){
            SDL_Rect r{p.x-3, p.y-3, 6, 6};
            SDL_SetRenderDrawColor(net_renderer,255,255,255,255);
            SDL_RenderFillRect(net_renderer,&r);
        }
    }

    SDL_RenderPresent(net_renderer);
}

void SnakeGame::render_network_dynamic(const vec_t& state){
    if(!render_mode) return;
    SDL_SetRenderDrawColor(dynamic_renderer,0,0,0,255);
    SDL_RenderClear(dynamic_renderer);

    const int win_w = 800;
    const int win_h = 600;
    const int padding_x = 40;
    const int padding_y = 25;
    const int layers[] = {INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    const int num_layers = 4;

    std::vector<std::vector<SDL_Point>> neuron_pos(num_layers);
    for(int l=0;l<num_layers;l++){
        int n = layers[l];
        neuron_pos[l].resize(n);
        for(int i=0;i<n;i++){
            neuron_pos[l][i] = { padding_x + l*(win_w - 2*padding_x)/(num_layers-1),
                                 padding_y + i*(win_h - 2*padding_y)/(n-1) };
        }
    }

    torch::Tensor inp = torch::tensor(state).reshape({1,INPUT_SIZE});
    torch::Tensor h1 = torch::relu(q_network->fc1->forward(inp));
    torch::Tensor h2 = torch::relu(q_network->fc2->forward(h1));
    torch::Tensor out = q_network->fc3->forward(h2);
    std::vector<torch::Tensor> activations = {inp.flatten(), h1.flatten(), h2.flatten(), out.flatten()};

    auto draw_connections = [&](torch::nn::Linear layer, int l){
        auto w = layer->weight.detach();
        int in = w.size(1), out_n = w.size(0);
        for(int i=0;i<out_n;i++){
            for(int j=0;j<in;j++){    
                float val = w[i][j].item<float>() * activations[l][j].item<float>();
                float intensity = std::tanh(std::abs(val)) * 255.0f; // pulse effect
                int r = val>0 ? intensity : 0;
                int g = val<0 ? intensity : 0;
                SDL_SetRenderDrawColor(dynamic_renderer,0,0,std::min(255, abs(int(intensity)) * 3), 255);
                SDL_RenderDrawLine(dynamic_renderer,
                                   neuron_pos[l][j].x, neuron_pos[l][j].y,
                                   neuron_pos[l+1][i].x, neuron_pos[l+1][i].y);
            }
        }
    };

    draw_connections(q_network->fc1,0);
    draw_connections(q_network->fc2,1);
    draw_connections(q_network->fc3,2);

    for(int l=0;l<num_layers;l++){
        for(int i=0;i<layers[l];i++){
            float act = activations[l][i].item<float>();
            int intensity = std::min(255,std::max(0,int(act*255)));
            SDL_SetRenderDrawColor(dynamic_renderer,intensity,intensity,255,255);
            SDL_Rect r{neuron_pos[l][i].x-5,neuron_pos[l][i].y-5,10,10};
            SDL_RenderFillRect(dynamic_renderer,&r);
        }
    }

    SDL_RenderPresent(dynamic_renderer);
}

void SnakeGame::train(int episodes){
    int total_score=0, max_score=0;

    for(int ep=0; ep<episodes; ep++){
        episode_count = ep;

        reset();
        auto state = get_state();
        int ep_score = 0;
        int steps = 0;

        while(!game_over && steps < 1000){
            int a = select_action(state);
            float r = step(a, (ep % 100 == 0));
            auto ns = get_state();

            replay_buffer.push_back({state, a, r, ns, game_over});
            if(replay_buffer.size() > REPLAY_BUFFER_SIZE) replay_buffer.pop_front();

            if(total_steps % TRAIN_EVERY == 0 && replay_buffer.size() >= BATCH_SIZE)
                train_step();

            state = ns;
            ep_score = score;
            steps++;
            total_steps++;
            if(force_render_next || ep % 100 == 0){
                render();
                render_network();
                render_network_dynamic(state); // <-- dynamic firing window
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                SDL_Event e; while(SDL_PollEvent(&e)){ if(e.type==SDL_QUIT) return;}            
            }
        }

        total_score += ep_score;
        max_score = std::max(max_score, ep_score);
        epsilon = std::max(EPSILON_END, epsilon * EPSILON_DECAY);

        if(ep % 10 == 0){
            float avg = total_score / 10.0f;
            std::cout << "Episode " << ep 
                      << " | Avg Score: " << avg 
                      << " | Max: " << max_score 
                      << " | Eps: " << epsilon 
                      << " | Buffer: " << replay_buffer.size() << "\n";
            total_score = 0;
        }
    }
}

int main(int argc,char**argv){
    std::signal(SIGINT, signal_handler); 
    bool render=true;
    SnakeGame g(render);
    if(!g.init()) return 1;
    std::cout<<"Starting training..."<<std::endl;
    g.train(10000);
    return 0;
}