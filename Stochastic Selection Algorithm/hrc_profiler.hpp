#pragma once

#include <chrono>
#include <string>

class hrc_profiler {
public:
	hrc_profiler(std::string profiler_name) : name(profiler_name) {}
	~hrc_profiler() {}

	__forceinline bool flag() { return total != std::chrono::duration<double, std::milli>::zero(); }
	__forceinline void start() { start_t = std::chrono::high_resolution_clock::now(); }
	__forceinline void stop() { total += std::chrono::high_resolution_clock::now() - start_t; }
	__forceinline std::string print_time() { return name + ": " + std::to_string(total.count()) + " ms\n"; }

private:
	std::string name;
	std::chrono::time_point<std::chrono::high_resolution_clock> start_t;
	std::chrono::duration<double, std::milli> total = std::chrono::duration<double, std::milli>::zero();
};
