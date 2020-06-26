// pure c++ neural network

///////////////////////////
//    author: grafstor
//    date: 26.06.20
///////////////////////////

// verison 1.0

#include <iostream>
#include <math.h>

void print_1d_array(float* a, int size)
{
    for (int i = 0; i<size; i++)
    	std::cout << a[i] << " ";
	std::cout << std::endl;
}

void print_2d_array(float** a, int size_x, int size_y)
{
    for (int i = 0; i<size_x; i++)
    {
	    for (int j = 0; j<size_y; j++)
	    	std::cout << a[i][j] << " ";

    	std::cout << std::endl;
    }
}

float* dot_feed_forword(float a[][3], float b[])
{
	float* result = new float[4];

	for (int i = 0; i<4; i++)
	{		
		float summary = 0;

		for (int p = 0; p<3; p++)
			summary += a[i][p] * b[p];

		result[i] = summary;
	}
	return result;
}

float* dot_backprop(float** x, float* delta)
{
	float* result = new float[3];

	for (int i = 0; i<3; i++)
	{		
		float summary = 0;

		for (int p = 0; p<4; p++)
			summary += x[i][p] * delta[p];

		result[i] = summary;
	}
	return result;
}

float* sigmoid(float* x)
{
	float* result = new float[4];

	const float E = 2.71828182846;

	for (int i = 0; i<4; i++)
		result[i] = 1 / (1 + pow(E, -x[i]));

	return result;
}

float* sigmoid_derivative(float* x)
{
	float* result = new float[4];

	for (int i = 0; i<4; i++)
		result[i] = x[i] * (1 - x[i]);

	return result;
}

float* crossentropy(float* y, float* p)
{
	float* result = new float[4];

	for (int i = 0; i<4; i++)
		result[i] = y[i] - p[i];

	return result;
}

float* count_delta(float* error, float* dsigmoid)
{
	float* result = new float[4];

	for (int i = 0; i<4; i++)
		result[i] = error[i] * dsigmoid[i];

	return result;
}

float** reverse_shape(float a[][3])
{
	float** result = 0;
	result = new float*[3];

	for (int i = 0; i<3; i++)
	{
		result[i] = new float[4];
		for (int j = 0; j<4; j++)
			result[i][j] = a[j][i];
	}

	return result;
}

void print_weights(float weights[])
{
	float* to_print = new float[3];

	for (int i = 0; i<3; i++)
		to_print[i] = weights[i];

	print_1d_array(to_print, 3);
}

int main()
{
	float train_x[4][3] = {{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}};
	float weights[3] = {0.23, -0.46, 0.53};
	float train_y[4] = {0, 1, 1, 0};

	for (int epoch = 0; epoch<10000; epoch++)
	{
		float* dot_product = dot_feed_forword(train_x, weights);
		float* output = sigmoid(dot_product);

		float* error = crossentropy(train_y, output);

		float* dsigmoid = sigmoid_derivative(output);
		float* delta = count_delta(error, dsigmoid);
		float** T_train_x = reverse_shape(train_x);

		float* adjustment = dot_backprop(T_train_x, delta);

		// change weights
		for (int i = 0; i<3; i++)
			weights[i] += adjustment[i];
	}

	// test neural network
	float test[1][3] = {{1, 0, 1}};

	float* dot_product = dot_feed_forword(test, weights);
	float* output = sigmoid(dot_product);

	std::cout << "test result: " << std::endl;
	print_1d_array(output, 1);

	std::cout << "weights: " << std::endl;
	print_weights(weights);

	return 0;
}
