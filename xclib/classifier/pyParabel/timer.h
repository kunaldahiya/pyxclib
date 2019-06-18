#pragma once

#include <iostream>
#include <chrono>

#include "config.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;


/* ----------------------------- timer resource ------------------------------------ */

class Timer
{
	private:
		high_resolution_clock::time_point start_time;
		high_resolution_clock::time_point stop_time;

	public:
		Timer()
		{
		}

		void tic()
		{
			start_time = high_resolution_clock::now();
		}

		_float toc()
		{
			stop_time = high_resolution_clock::now();
			_float elapsed_time = duration_cast<microseconds>(stop_time - start_time).count() / 1000000.0;
			return elapsed_time;
		}
};

