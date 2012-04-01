/*
 * 3BA26 Mandelbrot Project
 * 
 * Using techniques we've covered in class, accelerate the rendering of
 * the M set.
 * 
 * Hints
 * 
 * 1) Vectorize
 * 2) Use threads
 * 3) Load Balance
 * 4) Profile and Optimise
 * 
 * Potential FAQ.
 * 
 * Q1) Why when I zoom in far while palying with the code, why does the image begin to render all blocky?
 * A1) In order to render at increasing depths we must use increasingly higher precision floats
 * 	   We quickly run out of precision with 32 bits floats. Change all floats to doubles if you want
 * 	   dive deeper. Eventually you will however run out of precision again and need to integrate an
 * 	   infinite precision math library or use other techniques.
 * 
 * Q2) Why do some frames render much faster than others?
 * A2) Frames with a lot of black, i.e, frames showing a lot of set M, show pixels that ran until the 
 *     maximum number of iterations was reached before bailout. This means more CPU time was consumed
 */



#include <iostream>
#include <cmath>
#include <omp.h>
#include "xmmintrin.h"

#include "Screen.h"

#define TIMING
#ifdef TIMING
#include <sys/time.h>
#endif



/*
 * You can't change these values to accelerate the rendering.
 * Feel free to play with them to render different images though.
 */
const int 	MAX_ITS = 1000;			//Max Iterations before we assume the point will not escape
const int 	HXRES = 700; 			// horizontal resolution	
const int 	HYRES = 700;			// vertical resolution
const int 	MAX_DEPTH = 40;		// max depth of zoom
const float ZOOM_FACTOR = 1.02;		// zoom between each frame

/* Change these to zoom into different parts of the image */
const float PX = -0.702295281061;	// Centre point we'll zoom on - Real component
const float PY = +0.350220783400;	// Imaginary component


/*
 * The palette. Modifying this can produce some really interesting renders.
 * The colours are arranged R1,G1,B1, R2, G2, B2, R3.... etc.
 * RGB values are 0 to 255 with 0 being darkest and 255 brightest
 * 0,0,0 is black
 * 255,255,255 is white
 * 255,0,0 is bright red
 */
unsigned char pal[]={
	255,100,4,
	240,26,4,
	220,124,4,
	156,71,4,
	72,20,4,
	251,180,4,
	180,74,4,
	180,70,4,
	164,91,4,
	100,28,4,
	191,82,4,
	47,5,4,
	138,39,4,
	81,27,4,
	192,89,4,
	61,27,4,
	216,148,4,
	71,14,4,
	142,48,4,
	196,102,4,
	58,9,4,
	132,45,4,
	95,15,4,
	92,21,4,
	166,59,4,
	244,178,4,
	194,121,4,
	120,41,4,
	53,14,4,
	80,15,4,
	23,3,4,
	249,204,4,
	97,25,4,
	124,30,4,
	151,57,4,
	104,36,4,
	239,171,4,
	131,57,4,
	111,23,4,
	4,2,4};
const int PAL_SIZE = 40;  //Number of entries in the palette 

char memtable[700][700];


/* 
 * Return true if the point cx,cy is a member of set M.
 * iterations is set to the number of iterations until escape.
 */
bool member(float cx, float cy, int& iterations)
{
	if (memtable[(int)cx][(int)cy] == (char)1) {
		memtable[(int)cx][(int)cy] = (char)0;
		return true;
	}
	float x = 0.0;
	float y = 0.0;
	iterations = 0;
	while (true) {
		float xtemp = x*x - y*y + cx;
		y = 2*x*y + cy;
		x = xtemp;
		iterations+=1;
		if(!((x*x + y*y < 4) && (iterations < MAX_ITS)))break;

		xtemp = x*x -y*y + cx;
		y = 2*x*y + cy;
		x = xtemp;
		iterations+=1;
		if(!((x*x + y*y < 4) && (iterations < MAX_ITS)))break;

		xtemp = x*x -y*y + cx;
		y = 2*x*y + cy;
		x = xtemp;
		iterations+=1;
		if(!((x*x + y*y < 4) && (iterations < MAX_ITS)))break;


		xtemp = x*x -y*y + cx;
		y = 2*x*y + cy;
		x = xtemp;
		iterations+=1;
		if(!((x*x + y*y < 4) && (iterations < MAX_ITS)))break;


		xtemp = x*x -y*y + cx;
		y = 2*x*y + cy;
		x = xtemp;
		iterations+=1;
		if(!((x*x + y*y < 4) && (iterations < MAX_ITS)))break;
	}

	//return (iterations == MAX_ITS);
	if (iterations == MAX_ITS) {
		memtable[(int)cx][(int)cy] = (char)1;
		return true;
	} else {
		return false;
	}
}



int main()
{	
	int hx, hy;
	
	__m128 hrexes  = _mm_set1_ps((float)HXRES);
	//__m128 hreyes  = _mm_set1_ps((float)HYRES);

        __m128 offsets = _mm_set1_ps(-0.5f);
        __m128 pf = _mm_set1_ps(4.0f);
	__m128 pxs = _mm_set1_ps(PX);
        //__m128 pys = _mm_set1_ps(PY);

	float m=1.0; /* initial  magnification		*/
	__m128 ms = _mm_set1_ps(m);

	/* Create a screen to render to */
	Screen *screen;
	screen = new Screen(HXRES, HYRES);

	int depth=0;

#ifdef TIMING
  struct timeval start_time;
  struct timeval stop_time;
  long long total_time = 0;
#endif

	while (depth < MAX_DEPTH) {
#ifdef TIMING
	        /* record starting time */
	        gettimeofday(&start_time, NULL);
#endif
		__m128 rlf  = _mm_div_ps(pf, ms);
		__m128 lfr  = _mm_div_ps(pxs, rlf);
                #pragma omp parallel for schedule(dynamic, 20)
		for (hy=0; hy<HYRES; hy++) {
			float cy = ((((float)hy/(float)HYRES) -0.5 + (PY/(4.0/m)))*(4.0f/m));
                        for (hx=0; hx<HXRES; hx+=4) {
				int iterations;

				/* 
				 * Translate pixel coordinates to complex plane coordinates centred
				 * on PX, PY
				 */
				float results[4];

				__m128 hxs = _mm_setr_ps((float)hx,(float)hx+1,(float)hx+2,(float)hx+3);
				hxs = _mm_div_ps(hxs, hrexes);
			  	hxs = _mm_add_ps(hxs,offsets);	
				hxs = _mm_add_ps(hxs, lfr);
				hxs = _mm_mul_ps(hxs, rlf);


				_mm_store_ps(&results[0], hxs);

				#pragma omp parallel sections 
				{
				#pragma omp section 
				if (!member(results[0], cy, iterations)) {
					int i=(iterations%40) - 1;
					int b = i*3;
					screen->putpixel(hx, hy, pal[b], pal[b+1], pal[b+2]);
				} else {

					screen->putpixel(hx, hy, 0, 0, 0);
				}
			
				#pragma omp section 
				if (!member(results[1], cy, iterations)) {
					int i=(iterations%40) - 1;
					int b = i*3;
					screen->putpixel(hx+1, hy, pal[b], pal[b+1], pal[b+2]);
				} else {

					screen->putpixel(hx+1, hy, 0, 0, 0);
				}
				#pragma omp section 
				if (!member(results[2], cy, iterations)) {
					int i=(iterations%40) - 1;
					int b = i*3;
					screen->putpixel(hx+2, hy, pal[b], pal[b+1], pal[b+2]);
				} else {

					screen->putpixel(hx+2, hy, 0, 0, 0);
				}
				#pragma omp section 
				if (!member(results[3], cy, iterations)) {
					int i=(iterations%40) - 1;
					int b = i*3;
					screen->putpixel(hx+3, hy, pal[b], pal[b+1], pal[b+2]);
				} else {

					screen->putpixel(hx+3, hy, 0, 0, 0);
				}
				}//pragma omp sections
			}
		}
#ifdef TIMING
		gettimeofday(&stop_time, NULL);
		total_time += (stop_time.tv_sec - start_time.tv_sec) * 1000000L + (stop_time.tv_usec - start_time.tv_usec);
#endif
		/* Show the rendered image on the screen */
		//screen->flip();
		std::cout << "Render done " << depth++ << " " << m << std::endl;

		/* Zoom in */
		m *= ZOOM_FACTOR;
		ms = _mm_set1_ps(m);
	}
	
	sleep(10);
#ifdef TIMING
	std::cout << "Total executing time " << total_time << " microseconds\n";
#endif
	std::cout << "Clean Exit"<< std::endl;

}
