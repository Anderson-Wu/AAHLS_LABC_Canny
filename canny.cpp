#include <iostream>
#include <cmath>
using namespace std;

#include "canny.hpp"
#include "common.hpp"

#include "hls_stream.h"




struct window {
	DTYPE pix[FILTER_V_SIZE][FILTER_H_SIZE];
};

struct windowf {
	Float pix[FILTER_V_SIZE][FILTER_H_SIZE];
};


void ReadFromMem(DTYPE width,DTYPE height,DTYPE stride,DTYPE *src,hls::stream<DTYPE> &pixel_stream )
{
#pragma HLS INTERFACE m_axi depth=16384 port=src
    DTYPE pad[130*130];
#pragma HLS DEPENDENCE variable=pad inter false
#pragma HLS DEPENDENCE variable=pad intra false
    int n = 0;
    int index = 0;
    int temp;
    int tempup[130];
    int tempdown[130];
    int templeft[130];
    int tempright[130];
    for(int i = 1;i <= height;i++){
        for(int j = 1;j <= width;j++){
#pragma HLS PIPELINE
        	pad[i*(130)+j] =src[n];
            if(i == 2){
            	tempup[j] =src[n];
            }
            if(j == 2){
            	templeft[i] =src[n];
            	//pad[i*(128+2)] =src[n];
            }
            if(i == 127){
            	tempdown[j] = src[n];
            }
            if(j == 127){
            	tempright[i] =src[n];
            	//pad[i*(128+2)+129] = src[n];
            }
            n++;
        }
    }
    //129*(128+2)+j
    	for(int i = 1;i <=width;i++){
    		pad[i] = tempup[i];
    		pad[129*130+i] = tempdown[i];
    	}

    	for(int i = 1;i <=height;i++){
    		pad[i*(128+2)] = templeft[i];
    		pad[i*(128+2)+129] = tempright[i];
    	}
    	pad[0] = pad[2*(128+2)+2];

    	pad[129] = pad[2*(128+2)+127];

    	pad[129*(128+2)] = pad[127*(128+2)+2];


    	pad[129*(128+2)+129] =  pad[127*(128+2)+127];


    while(index <16900){
    	pixel_stream.write( pad[index]);
    	index++;
    }
}




void Window2DG(
		DTYPE	       width,
		DTYPE        height,
        hls::stream<DTYPE>      &pixel_stream,
        hls::stream<window>  &window_stream)
{
    // Line buffers - used to store [FILTER_V_SIZE-1] entire lines of pixels
    DTYPE LineBuffer[FILTER_V_SIZE-1][MAX_IMAGE_WIDTH+2];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

    // Sliding window of [FILTER_V_SIZE][FILTER_H_SIZE] pixels
    window Window;

    height+=2;
    width+=2;

    unsigned col_ptr = 0;
    unsigned ramp_up = 2*width*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2+1;
    unsigned num_pixels = width*height;
    unsigned num_iterations = num_pixels + ramp_up;
    num_iterations = 16900;
    int mod1 = 390;
    int mod2 = 391;

    // Iterate until all pixels have been processed
    update_window: for (int n=0; n<num_iterations; n++)
    {
#pragma HLS PIPELINE II=1

        // Read a new pixel from the input stream
        DTYPE new_pixel =pixel_stream.read();
        // Shift the window and add a column of new pixels from the line buffer
        for(int i = 0; i < FILTER_V_SIZE; i++) {
            for(int j = 0; j < FILTER_H_SIZE-1; j++) {
                Window.pix[i][j] = Window.pix[i][j+1];
            }
            Window.pix[i][FILTER_H_SIZE-1] = (i<FILTER_V_SIZE-1) ? LineBuffer[i][col_ptr] : new_pixel;
        }

        // Shift pixels in the column of pixels in the line buffer, add the newest pixel
        for(int i = 0; i < FILTER_V_SIZE-2; i++) {
            LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
        }
        LineBuffer[FILTER_V_SIZE-2][col_ptr] = new_pixel;

        // Update the line buffer column pointer
        if (col_ptr==(width-1)) {
            col_ptr = 0;
        } else {
            col_ptr++;
        }

        // Write output only when enough pixels have been read the buffers and ramped-up
        if (n>=ramp_up && n != mod1 && n != mod2) {
            window_stream.write(Window);
        }

		if(n == mod1){
			mod1+=130;
		}
		if(n == mod2){
			mod2+=130;
		}

    }
}


void Window2DM(
		DTYPE	       width,
		DTYPE        height,
        hls::stream<DTYPE>      &pixel_stream,
        hls::stream<window>  &window_stream)
{
    // Line buffers - used to store [FILTER_V_SIZE-1] entire lines of pixels
    DTYPE LineBuffer[FILTER_V_SIZE-1][MAX_IMAGE_WIDTH+2];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

    // Sliding window of [FILTER_V_SIZE][FILTER_H_SIZE] pixels
    window Window;

    height+=2;
    width+=2;

    unsigned col_ptr = 0;
    unsigned ramp_up = 2*width*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2+1;
    unsigned num_pixels = width*height;
    unsigned num_iterations = num_pixels + ramp_up;
    num_iterations = 16900;
    int mod1 = 390;
    int mod2 = 391;
    // Iterate until all pixels have been processed
    update_window: for (int n=0; n<num_iterations; n++)
    {

#pragma HLS PIPELINE II=1

        // Read a new pixel from the input stream
        DTYPE new_pixel =pixel_stream.read();
        // Shift the window and add a column of new pixels from the line buffer
        for(int i = 0; i < FILTER_V_SIZE; i++) {
            for(int j = 0; j < FILTER_H_SIZE-1; j++) {
                Window.pix[i][j] = Window.pix[i][j+1];
            }
            Window.pix[i][FILTER_H_SIZE-1] = (i<FILTER_V_SIZE-1) ? LineBuffer[i][col_ptr] : new_pixel;
        }


        // Shift pixels in the column of pixels in the line buffer, add the newest pixel
        for(int i = 0; i < FILTER_V_SIZE-2; i++) {
            LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
        }
        LineBuffer[FILTER_V_SIZE-2][col_ptr] = new_pixel;

        // Update the line buffer column pointer
        if (col_ptr==(width-1)) {
            col_ptr = 0;
        } else {
            col_ptr++;
        }

        // Write output only when enough pixels have been read the buffers and ramped-up
        if (n>=ramp_up && n!= mod1 && n!=mod2) {
            window_stream.write(Window);
        }

		if(n == mod1){
			mod1+=130;
		}
		if(n == mod2){
			mod2+=130;
		}



    }
}

void Window2DD(
		DTYPE	       width,
		DTYPE        height,
        hls::stream<DTYPE>      &pixel_stream,
        hls::stream<window>  &window_stream)
{
    // Line buffers - used to store [FILTER_V_SIZE-1] entire lines of pixels
   DTYPE LineBuffer[FILTER_V_SIZE-1][MAX_IMAGE_WIDTH+2];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

    // Sliding window of [FILTER_V_SIZE][FILTER_H_SIZE] pixels
    window Window;

    height+=2;
    width+=2;

    unsigned col_ptr = 0;
    unsigned ramp_up = 2*width*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2+1;
    unsigned num_pixels = width*height;
    unsigned num_iterations = num_pixels + ramp_up;
    num_iterations = 16900;
    int mod1 = 390;
    int mod2 = 391;
    // Iterate until all pixels have been processed
    update_window: for (int n=0; n<num_iterations; n++)
    {

#pragma HLS PIPELINE II=1

        // Read a new pixel from the input stream
        DTYPE new_pixel =pixel_stream.read();

        // Shift the window and add a column of new pixels from the line buffer
        for(int i = 0; i < FILTER_V_SIZE; i++) {
            for(int j = 0; j < FILTER_H_SIZE-1; j++) {
                Window.pix[i][j] = Window.pix[i][j+1];
            }
            Window.pix[i][FILTER_H_SIZE-1] = (i<FILTER_V_SIZE-1) ? LineBuffer[i][col_ptr] : new_pixel;
        }


        // Shift pixels in the column of pixels in the line buffer, add the newest pixel
        for(int i = 0; i < FILTER_V_SIZE-2; i++) {
            LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
        }
        LineBuffer[FILTER_V_SIZE-2][col_ptr] = new_pixel;

        // Update the line buffer column pointer
        if (col_ptr==(width-1)) {
            col_ptr = 0;
        } else {
            col_ptr++;
        }

        // Write output only when enough pixels have been read the buffers and ramped-up
        if (n>=ramp_up && n!= mod1 && n!= mod2) {
            window_stream.write(Window);
        }

		if(n == mod1){
			mod1+=130;
		}
		if(n == mod2){
			mod2+=130;
		}


    }
}


void Window2DS(
		DTYPE	       width,
		DTYPE        height,
        hls::stream<Float>      &pixel_stream,
        hls::stream<windowf>  &window_stream)
{
    // Line buffers - used to store [FILTER_V_SIZE-1] entire lines of pixels
    Float LineBuffer[FILTER_V_SIZE-1][MAX_IMAGE_WIDTH+2];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

    // Sliding window of [FILTER_V_SIZE][FILTER_H_SIZE] pixels
    windowf Window;

    height+=2;
    width+=2;

    unsigned col_ptr = 0;
    unsigned ramp_up = 2*width*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2+1;
    unsigned num_pixels = width*height;
    unsigned num_iterations = num_pixels + ramp_up;
    num_iterations = 16900;
    int mod1 = 390;
    int mod2 = 391;
    // Iterate until all pixels have been processed
    update_window: for (int n=0; n<num_iterations; n++)
    {

#pragma HLS PIPELINE II=1

        // Read a new pixel from the input stream
        Float new_pixel =  pixel_stream.read();

        // Shift the window and add a column of new pixels from the line buffer
        for(int i = 0; i < FILTER_V_SIZE; i++) {
            for(int j = 0; j < FILTER_H_SIZE-1; j++) {
                Window.pix[i][j] = Window.pix[i][j+1];
            }
            Window.pix[i][FILTER_H_SIZE-1] = (i<FILTER_V_SIZE-1) ? LineBuffer[i][col_ptr] : new_pixel;
        }


        // Shift pixels in the column of pixels in the line buffer, add the newest pixel
        for(int i = 0; i < FILTER_V_SIZE-2; i++) {
            LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
        }
        LineBuffer[FILTER_V_SIZE-2][col_ptr] = new_pixel;

        // Update the line buffer column pointer
        if (col_ptr==(width-1)) {
            col_ptr = 0;
        } else {
            col_ptr++;
        }

        // Write output only when enough pixels have been read the buffers and ramped-up
        if (n>=ramp_up && n!= mod1 && n!=mod2) {
            window_stream.write(Window);

        }

		if(n == mod1){
			mod1+=130;
		}
		if(n == mod2){
			mod2+=130;
		}

    }
}




void Filter2DG(
        DTYPE       width,
		DTYPE       height,
        Float   coeffs[][3],
        hls::stream<window> &window_stream,
		hls::stream<Float>     &pixel_stream )
{



#pragma HLS ARRAY_PARTITION variable=coeffs complete dim=0



    height +=2;
    width+=2;
    Float temp[130*130];
    int index=0;
    int writeindex=0;
    Float tempup[130];
    Float tempdown[130];
    Float templeft[130];
    Float tempright[130];
    apply_filter: for (int y = 1; y < height-1; y++)
    {
        for (int x = 1; x < width-1; x++)
        {
#pragma HLS PIPELINE II=1
            // Read a 2D window of pixels
            window w = window_stream.read();
            // Apply filter to the 2D window
            Float sum = 0;
            for(int row=0; row<FILTER_V_SIZE; row++)
            {
                for(int col=0; col<FILTER_H_SIZE; col++)
                {
                    Float pixel;
                    int xoffset = (x+col-(FILTER_H_SIZE/2));
                    int yoffset = (y+row-(FILTER_V_SIZE/2));
                    // Deal with boundary conditions : clamp pixels to 0 when outside of image

                    pixel = w.pix[row][col];

                    sum += pixel*coeffs[row][col];
                }
            }
            sum = sum/253;
            //pixel_stream.write(sum);
            temp[y*(128+2)+x] =sum;



            if(y == 2){
            	tempup[x] =sum;
            }


            if(x == 2){
            	templeft[y] =sum;
            }

            if(y == 127){
            	tempdown[x] =sum;
            }

            if(x == 127){
            	tempright[y] = sum;
            }

        }
    }


		for(int i = 1;i <=width;i++){
			temp[i] = tempup[i];
			temp[129*130+i] = tempdown[i];
		}

		for(int i = 1;i <=height;i++){
			temp[i*(128+2)] = templeft[i];
			temp[i*(128+2)+129] = tempright[i];
		}



    	temp[0] = temp[2*(128+2)+2];

    	temp[129] =  temp[2*(128+2)+127];

    	temp[129*(128+2)] =  temp[127*(128+2)+2];

    	temp[129*(128+2)+129] =   temp[127*(128+2)+127];

    while(index < 16900){
    	pixel_stream.write(temp[index]);
    	index++;
    }
}




void Filter2DS(
        DTYPE       width,
		DTYPE       height,
        Float   coeffs[][3],
        hls::stream<windowf> &window_stream,
		hls::stream<DTYPE>     &pixel_stream )
{



#pragma HLS ARRAY_PARTITION variable=coeffs complete dim=0


    // Process the incoming stream of pixel windows

    height +=2;
    width+=2;
    Float temp[130*130];
    Float shift=0.5;
    int index=0;
    int writeindex=0;
    int start = 0;
    apply_filter: for (int y = 1; y < height-1; y++)
    {
    	start = y*130;
        for (int x = 1; x < width-1; x++)
        {
#pragma HLS PIPELINE II=1
            // Read a 2D window of pixels
            windowf w = window_stream.read();

            // Apply filter to the 2D window
            Float sum = 0;
            for(int row=0; row<FILTER_V_SIZE; row++)
            {
                for(int col=0; col<FILTER_H_SIZE; col++)
                {
                    Float pixel;
                    int xoffset = (x+col-(FILTER_H_SIZE/2));
                    int yoffset = (y+row-(FILTER_V_SIZE/2));

                    pixel = w.pix[row][col];

                    sum += pixel*coeffs[row][col];
                }
            }


            temp[start+x] =sum;

        }

    }
    while(index < 16900){
    	if(temp[index] >=0){
    		temp[index]+=shift;
    	}
    	else{
    		temp[index]-=shift;
    	}
    	pixel_stream.write(DTYPE(temp[index]));
    	index++;
    }
}

void NMS(
        DTYPE       width,
		DTYPE       height,
        hls::stream<window> &Magnitude_Direction_window_stream,
		hls::stream<window> &gradient_window_stream,
		hls::stream<DTYPE> &nms_output_stream
		)
{



    height +=2;
    width+=2;
    Float temp[130*130];
    int index=0;
    int g1,g2;
    int res;
    apply_filter: for (int y = 1; y < height-1; y++)
    {
        for (int x = 1; x < width-1; x++)
        {
			#pragma HLS PIPELINE II=1
            // Read a 2D window of pixels
            window w = Magnitude_Direction_window_stream.read();
            window g = gradient_window_stream.read();

            if (g.pix[1][1]==2){
            g1 = w.pix[0][2];
            g2 = w.pix[2][0];
            }

			else if (g.pix[1][1]==1){
				g1 = w.pix[0][1];
				g2 = w.pix[2][1];
			}

			else if (g.pix[1][1]==0){
				g1 = w.pix[0][0];
				g2 = w.pix[2][2];
			}

			else{
				g1 = w.pix[1][0];
				g2 = w.pix[1][2];
			}
            if(w.pix[1][1]  < g1 || w.pix[1][1]  < g2){
            	res = 0;
            }
            else{
            	res = w.pix[1][1];
            }
        	nms_output_stream.write(res);
        }
    }
}


void Split(
        DTYPE       width,
		DTYPE       height,
        hls::stream<Float> &input_stream,
		hls::stream<Float>     &sobelx_stream,
		hls::stream<Float>     &sobely_stream){

	int numiterations = 130*130;
	for(int n = 0;n < numiterations;n++){
#pragma HLS PIPELINE II=1
		Float new_pixel = input_stream.read();
		sobelx_stream.write(new_pixel);
		sobely_stream.write(new_pixel);
	}

}


void  Threshold(int upperThresh, int lowerThresh ,hls::stream<DTYPE> &nms_output_stream,hls::stream<DTYPE> &threshold_output_stream){
	for(int i = 0;i < 16384;i++){
#pragma HLS PIPELINE II=1
		DTYPE val = nms_output_stream.read();
		if (val >= upperThresh*upperThresh)
			threshold_output_stream.write(255);
		else if (val > lowerThresh*lowerThresh )
			threshold_output_stream.write(127);
		else{
			threshold_output_stream.write(0);
		}
	}
}

void Magnitude_Direction(
        hls::stream<DTYPE> &sobelx_output_stream,
		hls::stream<DTYPE>     &sobely_output_stream,
		hls::stream<DTYPE>     &Magnitude_Direction_output_stream,
		hls::stream<DTYPE>     &gradient_output_stream){

	int numiterations = 130*130;
	for(int n = 0;n < numiterations;n++){
#pragma HLS PIPELINE II=1
		DTYPE x_grad = sobelx_output_stream.read();
		DTYPE y_grad = sobely_output_stream.read();
		DTYPE res = x_grad*x_grad + y_grad*y_grad;
		Magnitude_Direction_output_stream.write(res);
		Float coeff225 = 6.625;
		Float coeff675 = 38.625;
		DTYPE grad;
        if (x_grad >= 0 && y_grad >=0){
            Float gx225 = coeff225*x_grad;
            Float gx675 = coeff675*x_grad;
            y_grad = y_grad*16;

            if (y_grad >gx225 && y_grad <= gx675){
            	grad = 0;
            }
            else if (y_grad <= gx225){
            	grad = 3;
            }
            else if(y_grad > gx675){
            	grad = 1;
            }
        }


        if (x_grad <= 0 && y_grad <=0){
            x_grad = -x_grad;
            y_grad = -y_grad;
            Float gx225 = coeff225*x_grad;
            Float gx675 = coeff675*x_grad;
            y_grad = y_grad*16;

            if (y_grad >gx225 && y_grad <= gx675){
            	grad = 0;
            }
            else if (y_grad <= gx225){
            	grad = 3;
            }
            else if(y_grad > gx675){
            	grad = 1;
            }
        }


        if (x_grad <= 0 && y_grad >=0){
            x_grad = -x_grad;
            Float gx225 = coeff225*x_grad;
            Float gx675 = coeff675*x_grad;
            y_grad = y_grad*16;

            if (y_grad >gx225 && y_grad <= gx675){
            	grad = 2;
            }
            else if (y_grad <= gx225){
            	grad = 3;
            }
            else if(y_grad > gx675){
            	grad = 1;
            }
        }


        if (x_grad >= 0 && y_grad <=0){
            y_grad = -y_grad;
            Float gx225 = 6.625*x_grad;
            Float gx675 = 38.625*x_grad;
            y_grad = y_grad*16;

            if (y_grad >gx225 && y_grad <= gx675){
            	grad = 2;
            }
            else if (y_grad <= gx225){
            	grad = 3;
            }
            else if(y_grad > gx675){
            	grad = 1;
            }
        }
        gradient_output_stream.write(grad);
	}

}



void WriteToMem(
		DTYPE       width,
		DTYPE       height,
		DTYPE       stride,
        hls::stream<DTYPE>     &pixel_stream,
		DTYPE       *dst)
{
#pragma HLS INTERFACE m_axi depth=16384 port=dst

    write_image: for (int n = 0; n < 16384; n++) {
#pragma HLS PIPELINE II=1
        DTYPE pix = pixel_stream.read();
        dst[n] = pix;
    }
}



void canny(DTYPE* src, DTYPE* dst, int upperThresh, int lowerThresh)
{

#pragma HLS DATAFLOW
#pragma HLS INTERFACE m_axi depth=16384 port=src
#pragma HLS INTERFACE m_axi depth=16384 port=dst
	// Stream of pixels from kernel input to filter, and from filter to output
    hls::stream<Float,2>    coefs_stream;
    hls::stream<DTYPE,64>      pixel_stream;
    hls::stream<window,5>  window_stream; // Set FIFO depth to 0 to minimize resources
    hls::stream<DTYPE,64>     output_stream;
    hls::stream<Float,64>     gaussian_output_stream;
    hls::stream<Float,64>     sobelx_stream;
    hls::stream<Float,64>     sobely_stream;
    hls::stream<windowf,3>  sobelx_window_stream; // Set FIFO depth to 0 to minimize resources
    hls::stream<windowf,3>  sobely_window_stream; // Set FIFO depth to 0 to minimize resources
    hls::stream<DTYPE,64>     sobelx_output_stream;
    hls::stream<DTYPE,64>     sobely_output_stream;
    hls::stream<DTYPE,64>     Magnitude_Direction_output_stream;
    hls::stream<DTYPE,64>     gradient_output_stream;
    hls::stream<window,5>     Magnitude_Direction_window_stream;
    hls::stream<window,5>     gradient_window_stream;
    hls::stream<DTYPE,64>     nms_output_stream;


    int stride=128;
    int width = 128;
    int height = 128;

    Float gaussianCoeffs[3][3]={
    		{24,30,24},
    		{30,37,30},
    		{24,30,24}
    };



    Float sobelxCoeffs[3][3]={
    		{-1,0,1},
    		{-2,0,2},
    		{-1,0,1}
    };

    Float sobelyCoeffs[3][3]={
    		{-1,-2,-1},
    		{0,0,0},
    		{1,2,1}
    };


    ReadFromMem(width, height, stride, src, pixel_stream);


    Window2DG(width, height, pixel_stream, window_stream);


    Filter2DG(width, height, gaussianCoeffs, window_stream, gaussian_output_stream);


    Split(width, height, gaussian_output_stream, sobelx_stream,sobely_stream);

    Window2DS(width, height, sobelx_stream, sobelx_window_stream);

    Filter2DS(width, height, sobelxCoeffs, sobelx_window_stream, sobelx_output_stream);


    Window2DS(width, height, sobely_stream, sobely_window_stream);
    Filter2DS(width, height, sobelyCoeffs, sobely_window_stream, sobely_output_stream);

    Magnitude_Direction(sobelx_output_stream,sobely_output_stream,Magnitude_Direction_output_stream,gradient_output_stream);

    Window2DM(width, height, Magnitude_Direction_output_stream, Magnitude_Direction_window_stream);

    Window2DD(width, height, gradient_output_stream, gradient_window_stream);

    NMS(width, height,Magnitude_Direction_window_stream,gradient_window_stream,nms_output_stream);

    Threshold(upperThresh, lowerThresh,nms_output_stream,output_stream);

    WriteToMem(width, height, stride, output_stream, dst);



}
